"""Single-process-per-conv forward pass for chesapeakersc U-Net.

Drives ./build/gdalnn_conv (one subprocess per conv layer) and a small
handful of `gdal raster ...` CLIs for max-pool / upsample / concat / residual
adds / softmax. Replaces the ~2185-subprocess chunked-diagonal pipeline with
~40 subprocess calls total.

Usage:
    python predict_cpp.py samples/1717_image.tif samples/1717_gdal_probs.tif
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ENV = "/projects/bgtj/isaaccorley/envs/ftw-tile"
os.environ["PATH"] = f"{ENV}/bin:" + os.environ.get("PATH", "")
os.environ["PROJ_DATA"] = f"{ENV}/share/proj"
os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"]

ROOT = Path(__file__).parent.resolve()
GDALNN = str(ROOT / "build" / "gdalnn_conv")
W = np.load(ROOT / "model_weights.npz")
NTH = int(os.environ.get("GDALNN_THREADS", "16"))

COFLAGS = ["--co", "BIGTIFF=IF_SAFER", "--co", "INTERLEAVE=BAND"]

NCALLS = 0
T_SUB = 0.0
KEEP_WORK = False


def sh(*cmd):
    """Run a subprocess; account it; raise on nonzero."""
    global NCALLS, T_SUB
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("CMD:", " ".join(cmd[:30]))
        print(e.stderr.decode(errors="replace"))
        raise
    NCALLS += 1
    T_SUB += time.time() - t0


# ------ raw-weight cache (write .bin once per layer) ------
class WeightDir:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, Path] = {}

    def get(self, key: str) -> Path:
        p = self.cache.get(key)
        if p is not None:
            return p
        arr = np.ascontiguousarray(W[key].astype(np.float16))
        p = self.root / (key.replace(".", "_") + ".bin")
        arr.tofile(p)
        self.cache[key] = p
        return p


def conv(in_tif: Path, out_tif: Path, *, name: str,
         kernel_key: str,
         bn_a_key: str | None = None, bn_b_key: str | None = None,
         bias_key: str | None = None,
         relu: bool = True, stride: int = 1,
         padding: int | None = None,
         weights: WeightDir):
    """Run one full conv layer via gdalnn_conv."""
    k = W[kernel_key]
    Cout, Cin, kH, kW = k.shape
    if padding is None:
        padding = kH // 2  # PyTorch "same"-style for odd kernels
    args = [GDALNN,
            "--in", str(in_tif),
            "--kernel", str(weights.get(kernel_key)),
            "--kernel-shape", f"{Cout},{Cin},{kH},{kW}",
            "--stride", str(stride),
            "--padding", str(padding),
            "--out", str(out_tif),
            "--threads", str(NTH),
            *COFLAGS]
    if relu:
        args.append("--relu")
    if bn_a_key is not None:
        args += ["--bn-a", str(weights.get(bn_a_key)),
                 "--bn-b", str(weights.get(bn_b_key))]
    if bias_key is not None:
        args += ["--bias", str(weights.get(bias_key))]
    sh(*args)


# --- light numpy ops for non-conv pieces (kept out of subprocess count) ---
def _read(tif: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif) as src:
        return src.read(), src.profile.copy()


def _write(tif: Path, arr: np.ndarray, profile: dict):
    p = profile.copy()
    p.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
             dtype="float16", driver="GTiff", compress=None,
             tiled=False, BIGTIFF="IF_SAFER")
    with rasterio.open(tif, "w", **p) as dst:
        dst.write(arr.astype(np.float16))


def preprocess(in_tif: Path, out_tif: Path):
    """x / 255 -> Float16 raster (first 4 bands)."""
    with rasterio.open(in_tif) as src:
        a = src.read()[:4].astype(np.float32) / 255.0
        p = src.profile.copy()
    _write(out_tif, a, p)


def maxpool3x3_s2(in_tif: Path, out_tif: Path):
    """nn.MaxPool2d(3, 2, 1) on non-negative input.

    Zero-pad by 1, 3x3 max (numpy stride-tricks), stride-2 sample.
    """
    a, p = _read(in_tif)
    a = a.astype(np.float32)
    C, H, W_ = a.shape
    pad = np.zeros((C, H + 2, W_ + 2), dtype=np.float32)
    pad[:, 1:1+H, 1:1+W_] = a
    # 3x3 max via 9-fold maximum on shifted views
    m = pad[:, 0:H, 0:W_]
    for dy in range(3):
        for dx in range(3):
            if dy == 0 and dx == 0: continue
            m = np.maximum(m, pad[:, dy:dy+H, dx:dx+W_])
    out = m[:, ::2, ::2]
    _write(out_tif, out, p)


def upsample2x(in_tif: Path, out_tif: Path):
    a, p = _read(in_tif)
    out = a.repeat(2, axis=1).repeat(2, axis=2)
    _write(out_tif, out, p)


def concat(tifs: list[Path], out_tif: Path):
    arrs, profs = zip(*[_read(t) for t in tifs])
    out = np.concatenate(arrs, axis=0)
    _write(out_tif, out, profs[0])


def add_relu(a_tif: Path, b_tif: Path, out_tif: Path, *, relu: bool = True):
    a, p = _read(a_tif)
    b, _ = _read(b_tif)
    a = a.astype(np.float32); b = b.astype(np.float32)
    out = a + b
    if relu:
        np.maximum(out, 0, out=out)
    _write(out_tif, out, p)


def softmax_2cls(in_tif: Path, out_tif: Path):
    a, p = _read(in_tif)
    a = a.astype(np.float32)
    m = a.max(axis=0, keepdims=True)
    e = np.exp(a - m)
    pr = e / e.sum(axis=0, keepdims=True)
    p2 = p.copy()
    p2.update(count=2, dtype="float16", driver="GTiff",
              compress="deflate", tiled=True,
              blockxsize=256, blockysize=256)
    with rasterio.open(out_tif, "w", **p2) as dst:
        dst.write(pr.astype(np.float16))


# ------- net -------
def basic_block(in_tif: Path, out_tif: Path, *, layer: int, block: int,
                stride: int, workdir: Path, weights: WeightDir):
    p = f"l{layer}.b{block}"
    c1 = workdir / f"{p}_c1.tif"
    conv(in_tif, c1, name=f"{p}_c1",
         kernel_key=f"{p}.conv1.kernel",
         bn_a_key=f"{p}.conv1.bn_a", bn_b_key=f"{p}.conv1.bn_b",
         relu=True, stride=stride, weights=weights)
    c2 = workdir / f"{p}_c2.tif"
    conv(c1, c2, name=f"{p}_c2",
         kernel_key=f"{p}.conv2.kernel",
         bn_a_key=f"{p}.conv2.bn_a", bn_b_key=f"{p}.conv2.bn_b",
         relu=False, weights=weights)
    if not KEEP_WORK: c1.unlink(missing_ok=True)

    if block == 0 and layer > 1:
        skip = workdir / f"{p}_ds.tif"
        conv(in_tif, skip, name=f"{p}_ds",
             kernel_key=f"{p}.ds.kernel",
             bn_a_key=f"{p}.ds.bn_a", bn_b_key=f"{p}.ds.bn_b",
             relu=False, stride=stride, weights=weights)
    else:
        skip = in_tif
    add_relu(c2, skip, out_tif, relu=True)
    if not KEEP_WORK:
        c2.unlink(missing_ok=True)
        if skip != in_tif: skip.unlink(missing_ok=True)


def decoder_block(in_tif: Path, skip_tif: Path | None, out_tif: Path, *,
                  idx: int, workdir: Path, weights: WeightDir):
    up = workdir / f"d{idx}_up.tif"
    upsample2x(in_tif, up)
    if skip_tif is not None:
        cat = workdir / f"d{idx}_cat.tif"
        concat([up, skip_tif], cat)
        if not KEEP_WORK: up.unlink(missing_ok=True)
        layer_in = cat
    else:
        layer_in = up
    c1 = workdir / f"d{idx}_c1.tif"
    conv(layer_in, c1, name=f"d{idx}_c1",
         kernel_key=f"d{idx}.c1.kernel",
         bn_a_key=f"d{idx}.c1.bn_a", bn_b_key=f"d{idx}.c1.bn_b",
         relu=True, weights=weights)
    if not KEEP_WORK: layer_in.unlink(missing_ok=True)
    conv(c1, out_tif, name=f"d{idx}_c2",
         kernel_key=f"d{idx}.c2.kernel",
         bn_a_key=f"d{idx}.c2.bn_a", bn_b_key=f"d{idx}.c2.bn_b",
         relu=True, weights=weights)
    if not KEEP_WORK: c1.unlink(missing_ok=True)


def forward(in_tif: Path, out_tif: Path, *, workdir: Path):
    weights = WeightDir(workdir / "weights")
    pre = workdir / "00_pre.tif"
    preprocess(in_tif, pre)

    s1 = workdir / "01_stem.tif"
    conv(pre, s1, name="stem",
         kernel_key="stem.kernel",
         bn_a_key="stem.bn_a", bn_b_key="stem.bn_b",
         relu=True, stride=2, padding=3, weights=weights)
    if not KEEP_WORK: pre.unlink(missing_ok=True)

    mp = workdir / "02_mp.tif"
    maxpool3x3_s2(s1, mp)
    l1a = workdir / "02a.tif"; basic_block(mp, l1a, layer=1, block=0, stride=1, workdir=workdir, weights=weights)
    if not KEEP_WORK: mp.unlink(missing_ok=True)
    l1b = workdir / "02b.tif"; basic_block(l1a, l1b, layer=1, block=1, stride=1, workdir=workdir, weights=weights)
    if not KEEP_WORK: l1a.unlink(missing_ok=True)

    l2a = workdir / "03a.tif"; basic_block(l1b, l2a, layer=2, block=0, stride=2, workdir=workdir, weights=weights)
    l2b = workdir / "03b.tif"; basic_block(l2a, l2b, layer=2, block=1, stride=1, workdir=workdir, weights=weights)
    if not KEEP_WORK: l2a.unlink(missing_ok=True)

    l3a = workdir / "04a.tif"; basic_block(l2b, l3a, layer=3, block=0, stride=2, workdir=workdir, weights=weights)
    l3b = workdir / "04b.tif"; basic_block(l3a, l3b, layer=3, block=1, stride=1, workdir=workdir, weights=weights)
    if not KEEP_WORK: l3a.unlink(missing_ok=True)

    l4a = workdir / "05a.tif"; basic_block(l3b, l4a, layer=4, block=0, stride=2, workdir=workdir, weights=weights)
    l4b = workdir / "05b.tif"; basic_block(l4a, l4b, layer=4, block=1, stride=1, workdir=workdir, weights=weights)
    if not KEEP_WORK: l4a.unlink(missing_ok=True)

    d0 = workdir / "d0.tif"; decoder_block(l4b, l3b, d0, idx=0, workdir=workdir, weights=weights)
    if not KEEP_WORK: l4b.unlink(missing_ok=True); l3b.unlink(missing_ok=True)
    d1 = workdir / "d1.tif"; decoder_block(d0, l2b, d1, idx=1, workdir=workdir, weights=weights)
    if not KEEP_WORK: d0.unlink(missing_ok=True); l2b.unlink(missing_ok=True)
    d2 = workdir / "d2.tif"; decoder_block(d1, l1b, d2, idx=2, workdir=workdir, weights=weights)
    if not KEEP_WORK: d1.unlink(missing_ok=True); l1b.unlink(missing_ok=True)
    d3 = workdir / "d3.tif"; decoder_block(d2, s1, d3, idx=3, workdir=workdir, weights=weights)
    if not KEEP_WORK: d2.unlink(missing_ok=True); s1.unlink(missing_ok=True)
    d4 = workdir / "d4.tif"; decoder_block(d3, None, d4, idx=4, workdir=workdir, weights=weights)
    if not KEEP_WORK: d3.unlink(missing_ok=True)

    logits = workdir / "logits.tif"
    conv(d4, logits, name="head", kernel_key="head.kernel",
         bias_key="head.bias", relu=False, padding=1, weights=weights)
    if not KEEP_WORK: d4.unlink(missing_ok=True)

    softmax_2cls(logits, out_tif)
    if not KEEP_WORK: logits.unlink(missing_ok=True)


def _workdir_base() -> Path:
    for p in (os.environ.get("SLURM_TMPDIR"), "/dev/shm", "/tmp"):
        if p and Path(p).is_dir() and os.access(p, os.W_OK):
            return Path(p)
    return Path.cwd()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()
    global KEEP_WORK; KEEP_WORK = args.keep_work

    in_tif = Path(args.input).resolve()
    out_tif = Path(args.output or in_tif.with_suffix(".cpp_probs.tif")).resolve()
    base = ROOT if args.keep_work else _workdir_base()
    workdir = base / f"work_cpp_{int(time.time())}_{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[input]  {in_tif.name}")
    print(f"[output] {out_tif.name}")
    print(f"[work]   {workdir}")

    t0 = time.time()
    try:
        forward(in_tif, out_tif, workdir=workdir)
    finally:
        if not args.keep_work:
            shutil.rmtree(workdir, ignore_errors=True)
    dt = time.time() - t0
    print(f"\n[done] {dt:.1f}s  ({NCALLS} gdalnn_conv subprocesses, "
          f"avg {1000 * T_SUB / max(NCALLS, 1):.0f} ms each, "
          f"sub-total = {T_SUB:.1f}s)")


if __name__ == "__main__":
    main()
