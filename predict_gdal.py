"""Run chesapeakersc U-Net (smp.Unet + resnet18, in=4 out=2) entirely via
`gdal` CLI subprocesses, using the inert weights in model_weights.npz.

This is the deliberately-silly variant: the only actual math runs inside
`gdal raster neighbors|calc|reproject|stack`.  Python only orchestrates.

Strategy per conv layer (Cout, Cin, kH, kW):
  - For each input channel ic, run ONE `gdal raster neighbors --band ic+1`
    with all Cout kernels => produces a Cout-band intermediate raster.
  - Linearly accumulate the Cin intermediates with band-wise add `calc`s.
  - Final fused BN+ReLU (or bias) in one `calc` with Cout expressions.
  - Stride 2 -> follow with `reproject --size W/2,H/2 -r nearest`.

Strided convs use "full-res conv + nearest downsample" which is
mathematically equivalent to stride-2 conv (alignment within half a pixel).

Usage:  predict_gdal.py <input_naip_512.tif> [<output_probs.tif>]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ENV = "/projects/bgtj/isaaccorley/envs/ftw-tile"
os.environ["PATH"] = f"{ENV}/bin:" + os.environ.get("PATH", "")
os.environ["PROJ_DATA"] = f"{ENV}/share/proj"
os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"]

ROOT = Path(__file__).parent.resolve()
W = np.load(ROOT / "model_weights.npz")

# Parallelism: ThreadPool fan-out for independent gdal subprocesses.
# 8 workers × ~4 internal threads/process is plenty for our 64-CPU box.
NUM_WORKERS = 8
POOL = ThreadPoolExecutor(max_workers=NUM_WORKERS)

# Stats
_LOCK = threading.Lock()
NCALLS = 0
T_GDAL = 0.0


def sh(*cmd):
    global NCALLS, T_GDAL
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("CMD:", " ".join(cmd[:40]))
        print(e.stderr.decode(errors="replace"))
        raise
    dt = time.time() - t0
    with _LOCK:
        NCALLS += 1
        T_GDAL += dt


def parallel(fn_args):
    """Run a list of (fn, args, kwargs) tuples in parallel; return list of results."""
    futures = [POOL.submit(fn, *a, **kw) for fn, a, kw in fn_args]
    return [f.result() for f in futures]


def kstr(k: np.ndarray) -> str:
    return json.dumps(k.tolist()).replace(" ", "")


def get_size(tif: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["gdal", "raster", "info", "-f", "json", str(tif)]).decode()
    info = json.loads(out)
    w, h = info["size"]
    return int(w), int(h)


CHUNK = 32  # default input-channel chunk size for the diagonal-trick fusion
MAX_INTERMEDIATE_BYTES = 800_000_000  # ~800 MB cap on per-chunk neighbors raster

# Workdir base: prefer fast-tmpfs to avoid disk I/O on every intermediate.
def _workdir_base() -> Path:
    for p in (os.environ.get("SLURM_TMPDIR"), "/dev/shm", "/tmp"):
        if p and Path(p).is_dir() and os.access(p, os.W_OK):
            return Path(p)
    return Path.cwd()

# All write ops: BIGTIFF (avoid 4 GB TIFF format limit) + minimal metadata
# (intermediates inherit the input's CRS/geotransform; GeoTIFF tags add I/O
# overhead so we skip writing them) + per-band interleave (better for many
# many-banded intermediates).
COFLAGS = [
    "--co", "BIGTIFF=IF_SAFER",
    "--co", "INTERLEAVE=BAND",
]


# ----------------------------------------------------------------------------
# Geometric helpers (numpy via rasterio).  These are not "pure gdal CLI" but
# they isolate the parts where PyTorch and `gdal raster ...` disagree by
# half a pixel or a padding mode -- doing them numerically here gives us
# exact PyTorch semantics.
# ----------------------------------------------------------------------------
def pad_zeros(in_tif: Path, out_tif: Path, pad: int) -> Path:
    """Zero-pad a multi-band raster by `pad` rows/cols on each side."""
    if pad == 0:
        shutil.copy(str(in_tif), str(out_tif))
        return out_tif
    with rasterio.open(in_tif) as src:
        data = src.read()
        prof = src.profile.copy()
    out = np.pad(data, ((0, 0), (pad, pad), (pad, pad)),
                 mode="constant", constant_values=0)
    prof.update(height=out.shape[1], width=out.shape[2])
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(out)
    return out_tif


def crop_center(in_tif: Path, out_tif: Path, pad: int) -> Path:
    """Inverse of pad_zeros: drop `pad` rows/cols from each side."""
    if pad == 0:
        shutil.copy(str(in_tif), str(out_tif))
        return out_tif
    with rasterio.open(in_tif) as src:
        H, W = src.height, src.width
        data = src.read(window=Window(pad, pad, W - 2 * pad, H - 2 * pad))
        prof = src.profile.copy()
    prof.update(height=H - 2 * pad, width=W - 2 * pad)
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(data)
    return out_tif


def stride2_downsample(in_tif: Path, out_tif: Path) -> Path:
    """Sample every other pixel starting at (0, 0) — matches PyTorch
    `conv2d(stride=2)` output sampling exactly."""
    with rasterio.open(in_tif) as src:
        data = src.read()
        prof = src.profile.copy()
    out = data[:, ::2, ::2]
    prof.update(height=out.shape[1], width=out.shape[2])
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(out)
    return out_tif


def nearest_2x_up(in_tif: Path, out_tif: Path) -> Path:
    """Replicate each pixel as a 2x2 block — matches PyTorch
    `nn.Upsample(scale_factor=2, mode='nearest')` exactly."""
    with rasterio.open(in_tif) as src:
        data = src.read()
        prof = src.profile.copy()
    out = data.repeat(2, axis=1).repeat(2, axis=2)
    prof.update(height=out.shape[1], width=out.shape[2])
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(out)
    return out_tif


def pick_chunk_size(Cin: int, Cout: int, spatial_hw: tuple[int, int]) -> int:
    """Choose K so the neighbors output (K*K*Cout bands at Float16) stays under
    both the memory budget and the 65535-band TIFF format limit."""
    H, W_ = spatial_hw
    per_band = max(1, H * W_ * 2)  # Float16 = 2 bytes
    mem_cap = max(1, int((MAX_INTERMEDIATE_BYTES / (Cout * per_band)) ** 0.5))
    band_cap = max(1, int((65000 / Cout) ** 0.5))  # leave a small safety margin
    return min(CHUNK, Cin, mem_cap, band_cap)


def _run_chunk_pipeline(in_tif: Path, kernel: np.ndarray, c: int, k: int,
                         Cout: int, work: Path) -> Path:
    """One chunk's `select+neighbors` pipeline call."""
    kernel_args = []
    for oc in range(Cout):
        for ic_in_chunk in range(k):
            kernel_args.extend(["--kernel", kstr(kernel[oc, c + ic_in_chunk])])
    band_args = []
    for ic_in_chunk in range(k):
        band_args.extend(["-b", str(c + ic_in_chunk + 1)])
    nb = work / f"nb_{c}.tif"
    sh("gdal", "raster", "pipeline", "--quiet",
       "!", "read", str(in_tif),
       "!", "select", *band_args,
       "!", "neighbors", *kernel_args, "--method", "sum", "--ot", "Float16",
       "!", "write", "--overwrite", *COFLAGS, str(nb))
    return nb


def _run_chunk_diag(nb: Path, c: int, k: int, Cout: int, work: Path) -> Path:
    """One chunk's diagonal-pick calc: extract Cout bands from K*K*Cout intermediate."""
    partial = work / f"part_{c}.tif"
    calc_args = []
    for oc in range(Cout):
        terms = [f"X[{ic_in_chunk * (k * Cout + 1) + oc * k + 1}]"
                 for ic_in_chunk in range(k)]
        calc_args.extend(["--calc", " + ".join(terms)])
    sh("gdal", "raster", "calc", "-q", "--overwrite",
       *calc_args, "--flatten",
       "-i", f"X={nb}",
       "--ot", "Float16", *COFLAGS,
       str(partial))
    nb.unlink(missing_ok=True) if not KEEP_WORK else None
    return partial


def _sum_two(a: Path, b: Path, out: Path, Cout: int,
              bn_a: np.ndarray | None = None,
              bn_b: np.ndarray | None = None,
              bias: np.ndarray | None = None,
              relu: bool = False) -> Path:
    """sum + optional fused bias/BN/ReLU per output channel."""
    calc_args = []
    for oc in range(Cout):
        term = f"X[{oc + 1}] + Y[{oc + 1}]"
        if bias is not None:
            term = f"(({term}) + ({float(bias[oc])}))"
        if bn_a is not None:
            term = f"(({float(bn_a[oc])}) * ({term}) + ({float(bn_b[oc])}))"
        if relu:
            term = f"max(0, {term})"
        calc_args.extend(["--calc", term])
    sh("gdal", "raster", "calc", "-q", "--overwrite",
       *calc_args, "--flatten",
       "-i", f"X={a}", "-i", f"Y={b}",
       "--ot", "Float16", *COFLAGS,
       str(out))
    a.unlink(missing_ok=True) if not KEEP_WORK else None; b.unlink(missing_ok=True) if not KEEP_WORK else None
    return out


def _bnrelu_inplace(in_p: Path, out_p: Path, Cout: int,
                    bn_a: np.ndarray | None,
                    bn_b: np.ndarray | None,
                    bias: np.ndarray | None,
                    relu: bool) -> Path:
    calc_args = []
    for oc in range(Cout):
        term = f"X[{oc + 1}]"
        if bias is not None:
            term = f"(({term}) + ({float(bias[oc])}))"
        if bn_a is not None:
            term = f"(({float(bn_a[oc])}) * ({term}) + ({float(bn_b[oc])}))"
        if relu:
            term = f"max(0, {term})"
        calc_args.extend(["--calc", term])
    sh("gdal", "raster", "calc", "-q", "--overwrite",
       *calc_args, "--flatten",
       "-i", f"X={in_p}",
       "--ot", "Float16", *COFLAGS,
       str(out_p))
    in_p.unlink(missing_ok=True) if not KEEP_WORK else None
    return out_p


def conv_layer(in_tif: Path, out_tif: Path, name: str,
               kernel: np.ndarray, *,
               bn_a: np.ndarray | None = None,
               bn_b: np.ndarray | None = None,
               bias: np.ndarray | None = None,
               relu: bool = True,
               stride: int = 1,
               workdir: Path,
               chunk_size: int | None = None) -> Path:
    """Parallel chunked-diagonal conv -> [bias/BN] -> [ReLU] -> [stride 2].

    1. All chunks run their `select+neighbors` pipelines in parallel.
    2. All chunks run their diagonal-pick `calc` in parallel.
    3. Partials reduced via parallel pairwise sums (last sum fuses BN+ReLU/bias).
    4. Optional stride-2 nearest downsample.
    """
    Cout, Cin, kH, kW = kernel.shape
    work = workdir / name
    work.mkdir(parents=True, exist_ok=True)

    # PyTorch conv with kernel size kH uses padding = kH // 2 (zero pad).
    # GDAL `neighbors` edge-replicates at the raster boundary -- mismatch.
    # Fix: zero-pad the raster ourselves, run neighbors on the padded raster,
    # then crop the output back to the original spatial size.
    pad = kH // 2
    if pad > 0:
        padded_in = work / "padded_in.tif"
        pad_zeros(in_tif, padded_in, pad)
        conv_in = padded_in
    else:
        conv_in = in_tif

    Wi, Hi = get_size(conv_in)
    K = chunk_size or pick_chunk_size(Cin, Cout, (Hi, Wi))

    chunks = []
    for c in range(0, Cin, K):
        chunks.append((c, min(K, Cin - c)))

    # Phase 1+2: each chunk runs pipeline -> diagonal-pick.
    # Bundle both stages into one function so a worker handles both end-to-end
    # without coming back to the scheduler between them.
    def _chunk(c: int, k: int) -> Path:
        nb = _run_chunk_pipeline(conv_in, kernel, c, k, Cout, work)
        return _run_chunk_diag(nb, c, k, Cout, work)

    fn_args = [(_chunk, (c, k), {}) for (c, k) in chunks]
    partials = parallel(fn_args)

    # Phase 3: parallel pairwise reduction (binary tree).  The very last sum
    # in the tree fuses bias / BN+ReLU.
    needs_final = (bn_a is not None) or (bias is not None) or relu
    round_idx = 0
    while len(partials) > 1:
        next_partials: list[Path] = []
        pair_tasks: list[tuple] = []
        odd_one: Path | None = None
        for i in range(0, len(partials) - 1, 2):
            a, b = partials[i], partials[i + 1]
            out_p = work / f"r{round_idx}_{i//2}.tif"
            # If this is the final reduction (will produce a single result),
            # fuse the BN+ReLU/bias INTO this sum.
            is_final = (len(partials) == 2 and needs_final)
            kw = {}
            if is_final:
                kw = dict(bn_a=bn_a, bn_b=bn_b, bias=bias, relu=relu)
            pair_tasks.append((_sum_two, (a, b, out_p, Cout), kw))
        if len(partials) % 2 == 1:
            odd_one = partials[-1]
        results = parallel(pair_tasks)
        next_partials = results + ([odd_one] if odd_one else [])
        partials = next_partials
        round_idx += 1

    final = partials[0]

    # If there was only one chunk (no pairwise reduction), apply BN+ReLU here.
    if needs_final and round_idx == 0:
        new = work / "fin.tif"
        final = _bnrelu_inplace(final, new, Cout, bn_a, bn_b, bias, relu)

    # Crop padded boundary back to original spatial size.
    if pad > 0:
        cropped = work / "cropped.tif"
        crop_center(final, cropped, pad)
        if not KEEP_WORK:
            final.unlink(missing_ok=True)
        final = cropped

    # Stride 2: pick every other pixel starting at (0, 0) -- exact match
    # for PyTorch `conv2d(stride=2)` output positions.
    if stride == 2:
        stride2_downsample(final, out_tif)
        if not KEEP_WORK:
            final.unlink(missing_ok=True)
    else:
        shutil.move(str(final), str(out_tif))

    if not KEEP_WORK:
        shutil.rmtree(work, ignore_errors=True)
    return out_tif


def maxpool3x3_stride2(in_tif: Path, out_tif: Path, *, workdir: Path) -> Path:
    """3x3 maxpool stride 2 with padding 1, matching `nn.MaxPool2d(3, 2, 1)`.

    Post-ReLU activations are non-negative, so zero-padding is equivalent
    to PyTorch's -inf padding for max purposes (max(0, anything>=0) = the
    other value if positive, else 0).
    """
    # Zero-pad by 1 row/col on each side
    padded = workdir / f"{out_tif.stem}_padded.tif"
    pad_zeros(in_tif, padded, 1)
    # 3x3 sliding max
    full = workdir / f"{out_tif.stem}_full.tif"
    k = kstr(np.ones((3, 3)))
    sh("gdal", "raster", "neighbors", "-q", "--overwrite",
       "--kernel", k,
       "--method", "max",
       "--ot", "Float16", *COFLAGS,
       str(padded), str(full))
    if not KEEP_WORK:
        padded.unlink(missing_ok=True)
    # Crop back to original H, W
    cropped = workdir / f"{out_tif.stem}_cropped.tif"
    crop_center(full, cropped, 1)
    if not KEEP_WORK:
        full.unlink(missing_ok=True)
    # Pick every other pixel for stride-2 sampling
    stride2_downsample(cropped, out_tif)
    if not KEEP_WORK:
        cropped.unlink(missing_ok=True)
    return out_tif


def upsample2x(in_tif: Path, out_tif: Path) -> Path:
    """Nearest 2x upsample: each pixel becomes a 2x2 block, no smoothing.
    Matches `nn.Upsample(scale_factor=2, mode='nearest')` exactly."""
    return nearest_2x_up(in_tif, out_tif)


def concat(tifs: list[Path], out_tif: Path) -> Path:
    sh("gdal", "raster", "stack", "-q", "--overwrite", *COFLAGS,
       *[str(t) for t in tifs], str(out_tif))
    return out_tif


def add_relu(a_tif: Path, b_tif: Path, out_tif: Path, *, relu: bool = True) -> Path:
    """Multi-band residual add (and optional ReLU)."""
    Wa, Ha = get_size(a_tif)
    # need band count
    info = json.loads(subprocess.check_output(
        ["gdal", "raster", "info", "-f", "json", str(a_tif)]).decode())
    n_bands = len(info["bands"])
    calc_args = []
    for oc in range(n_bands):
        expr = f"X[{oc + 1}] + Y[{oc + 1}]"
        if relu:
            expr = f"max(0, {expr})"
        calc_args.extend(["--calc", expr])
    sh("gdal", "raster", "calc", "-q", "--overwrite",
       *calc_args, "--flatten",
       "-i", f"X={a_tif}",
       "-i", f"Y={b_tif}",
       "--ot", "Float16", *COFLAGS,
       str(out_tif))
    return out_tif


def preprocess(in_tif: Path, out_tif: Path) -> Path:
    """Normalize: x / 255.0  (matches T.Normalize(mean=[0], std=[255]))."""
    calc_args = []
    for ic in range(4):
        calc_args.extend(["--calc", f"X[{ic + 1}] / 255.0"])
    sh("gdal", "raster", "calc", "-q", "--overwrite",
       *calc_args, "--flatten",
       "-i", f"X={in_tif}",
       "--ot", "Float16", *COFLAGS,
       str(out_tif))
    return out_tif


def softmax_2cls(in_tif: Path, out_tif: Path) -> Path:
    """Numerically-stable 2-class softmax across bands."""
    mx = "max(X[1], X[2])"
    den = f"(exp(X[1] - ({mx})) + exp(X[2] - ({mx})))"
    sh("gdal", "raster", "calc", "-q", "--overwrite",
       "--calc", f"exp(X[1] - ({mx})) / {den}",
       "--calc", f"exp(X[2] - ({mx})) / {den}",
       "--flatten",
       "-i", f"X={in_tif}",
       "--ot", "Float16",
       "--co", "TILED=YES", "--co", "COMPRESS=DEFLATE",
       "--co", "BLOCKXSIZE=256", "--co", "BLOCKYSIZE=256",
       str(out_tif))
    return out_tif


def fix_geo(in_tif: Path, out_tif: Path) -> Path:
    """Normalize geotransform: copy file + force neg NS via `gdal raster edit`."""
    shutil.copy(str(in_tif), str(out_tif))
    Wi, Hi = get_size(out_tif)
    # Note: --bbox arranges things so NS is negative
    sh("gdal", "raster", "edit", "-q",
       "--bbox", f"0,0,{Wi},{Hi}",
       str(out_tif))
    return out_tif


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
def basic_block(in_tif: Path, out_tif: Path, *, layer: int, block: int,
                stride: int, workdir: Path) -> Path:
    """ResNet-18 BasicBlock: conv-BN-ReLU -> conv-BN -> + skip -> ReLU."""
    p = f"l{layer}.b{block}"

    # main path
    c1 = workdir / f"{p}_c1.tif"
    conv_layer(in_tif, c1, f"{p}_c1",
               kernel=W[f"{p}.conv1.kernel"],
               bn_a=W[f"{p}.conv1.bn_a"],
               bn_b=W[f"{p}.conv1.bn_b"],
               relu=True, stride=stride, workdir=workdir)

    c2 = workdir / f"{p}_c2.tif"
    conv_layer(c1, c2, f"{p}_c2",
               kernel=W[f"{p}.conv2.kernel"],
               bn_a=W[f"{p}.conv2.bn_a"],
               bn_b=W[f"{p}.conv2.bn_b"],
               relu=False, workdir=workdir)
    c1.unlink(missing_ok=True) if not KEEP_WORK else None

    # skip path
    if block == 0 and layer > 1:
        skip = workdir / f"{p}_ds.tif"
        conv_layer(in_tif, skip, f"{p}_ds",
                   kernel=W[f"{p}.ds.kernel"],
                   bn_a=W[f"{p}.ds.bn_a"],
                   bn_b=W[f"{p}.ds.bn_b"],
                   relu=False, stride=stride, workdir=workdir)
    else:
        skip = in_tif

    add_relu(c2, skip, out_tif, relu=True)
    c2.unlink(missing_ok=True) if not KEEP_WORK else None
    if skip != in_tif:
        skip.unlink(missing_ok=True) if not KEEP_WORK else None
    return out_tif


def decoder_block(in_tif: Path, skip_tif: Path | None, out_tif: Path, *,
                  idx: int, workdir: Path) -> Path:
    """smp Unet decoder block: upsample -> [concat skip] -> conv-BN-ReLU x 2."""
    up = workdir / f"d{idx}_up.tif"
    upsample2x(in_tif, up)

    if skip_tif is not None:
        cat = workdir / f"d{idx}_cat.tif"
        concat([up, skip_tif], cat)
        up.unlink(missing_ok=True) if not KEEP_WORK else None
        layer_in = cat
    else:
        layer_in = up

    c1 = workdir / f"d{idx}_c1.tif"
    conv_layer(layer_in, c1, f"d{idx}_c1",
               kernel=W[f"d{idx}.c1.kernel"],
               bn_a=W[f"d{idx}.c1.bn_a"], bn_b=W[f"d{idx}.c1.bn_b"],
               relu=True, workdir=workdir)
    layer_in.unlink(missing_ok=True) if not KEEP_WORK else None

    conv_layer(c1, out_tif, f"d{idx}_c2",
               kernel=W[f"d{idx}.c2.kernel"],
               bn_a=W[f"d{idx}.c2.bn_a"], bn_b=W[f"d{idx}.c2.bn_b"],
               relu=True, workdir=workdir)
    c1.unlink(missing_ok=True) if not KEEP_WORK else None
    return out_tif


def forward(in_tif: Path, out_tif: Path, *, workdir: Path):
    # preprocess (divide by 255)
    pre = workdir / "00_pre.tif"
    fixed = workdir / "00_fixed.tif"
    fix_geo(in_tif, fixed)
    preprocess(fixed, pre)
    fixed.unlink(missing_ok=True) if not KEEP_WORK else None

    # encoder
    # Stage 1 (stride 2): conv1 + bn1 + ReLU
    s1 = workdir / "01_stem.tif"  # stride 2, 64ch
    conv_layer(pre, s1, "stem",
               kernel=W["stem.kernel"],
               bn_a=W["stem.bn_a"], bn_b=W["stem.bn_b"],
               relu=True, stride=2, workdir=workdir)
    pre.unlink(missing_ok=True) if not KEEP_WORK else None

    # Stage 2: maxpool (stride 2) + layer1 (stride 1)
    mp = workdir / "02_mp.tif"
    maxpool3x3_stride2(s1, mp, workdir=workdir)
    l1a = workdir / "02a.tif"
    basic_block(mp, l1a, layer=1, block=0, stride=1, workdir=workdir)
    mp.unlink(missing_ok=True) if not KEEP_WORK else None
    l1b = workdir / "02b.tif"
    basic_block(l1a, l1b, layer=1, block=1, stride=1, workdir=workdir)
    l1a.unlink(missing_ok=True) if not KEEP_WORK else None
    # l1b is the feature at stride 4 (after maxpool + layer1)

    # Stage 3: layer2 (stride 2 in first block)
    l2a = workdir / "03a.tif"
    basic_block(l1b, l2a, layer=2, block=0, stride=2, workdir=workdir)
    l2b = workdir / "03b.tif"
    basic_block(l2a, l2b, layer=2, block=1, stride=1, workdir=workdir)
    l2a.unlink(missing_ok=True) if not KEEP_WORK else None

    # Stage 4: layer3
    l3a = workdir / "04a.tif"
    basic_block(l2b, l3a, layer=3, block=0, stride=2, workdir=workdir)
    l3b = workdir / "04b.tif"
    basic_block(l3a, l3b, layer=3, block=1, stride=1, workdir=workdir)
    l3a.unlink(missing_ok=True) if not KEEP_WORK else None

    # Stage 5: layer4 (deepest)
    l4a = workdir / "05a.tif"
    basic_block(l3b, l4a, layer=4, block=0, stride=2, workdir=workdir)
    l4b = workdir / "05b.tif"  # final encoder feature at stride 32
    basic_block(l4a, l4b, layer=4, block=1, stride=1, workdir=workdir)
    l4a.unlink(missing_ok=True) if not KEEP_WORK else None

    # smp Unet decoder takes features (rev order): [l4b, l3b, l2b, l1b, s1]
    # block 0: in=l4b (512ch), skip=l3b (256ch) -> 256ch
    d0 = workdir / "d0.tif"
    decoder_block(l4b, l3b, d0, idx=0, workdir=workdir)
    l4b.unlink(missing_ok=True) if not KEEP_WORK else None; l3b.unlink(missing_ok=True) if not KEEP_WORK else None

    # block 1: in=d0, skip=l2b (128) -> 128
    d1 = workdir / "d1.tif"
    decoder_block(d0, l2b, d1, idx=1, workdir=workdir)
    d0.unlink(missing_ok=True) if not KEEP_WORK else None; l2b.unlink(missing_ok=True) if not KEEP_WORK else None

    # block 2: in=d1, skip=l1b (64) -> 64
    d2 = workdir / "d2.tif"
    decoder_block(d1, l1b, d2, idx=2, workdir=workdir)
    d1.unlink(missing_ok=True) if not KEEP_WORK else None; l1b.unlink(missing_ok=True) if not KEEP_WORK else None

    # block 3: in=d2, skip=s1 (64) -> 32
    d3 = workdir / "d3.tif"
    decoder_block(d2, s1, d3, idx=3, workdir=workdir)
    d2.unlink(missing_ok=True) if not KEEP_WORK else None; s1.unlink(missing_ok=True) if not KEEP_WORK else None

    # block 4: in=d3, no skip -> 16
    d4 = workdir / "d4.tif"
    decoder_block(d3, None, d4, idx=4, workdir=workdir)
    d3.unlink(missing_ok=True) if not KEEP_WORK else None

    # segmentation head: 3x3 conv 16->2 with bias, no BN no ReLU
    logits = workdir / "logits.tif"
    conv_layer(d4, logits, "head",
               kernel=W["head.kernel"],
               bias=W["head.bias"],
               relu=False, workdir=workdir)
    d4.unlink(missing_ok=True) if not KEEP_WORK else None

    # softmax across 2 classes -> 2-band float32 probability raster
    softmax_2cls(logits, out_tif)
    logits.unlink(missing_ok=True) if not KEEP_WORK else None
    return out_tif


KEEP_WORK = False


def maybe_unlink(p: Path):
    if not KEEP_WORK:
        p.unlink(missing_ok=True) if not KEEP_WORK else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default=None)
    p.add_argument("--keep-work", action="store_true",
                   help="don't delete intermediate work dir or layer outputs")
    args = p.parse_args()
    global KEEP_WORK
    KEEP_WORK = args.keep_work

    in_tif = Path(args.input).resolve()
    out_tif = Path(args.output or in_tif.with_suffix(".gdal_probs.tif")).resolve()
    # Use fast tmpfs for the work dir (cuts NFS I/O on every intermediate).
    # If --keep-work is set, fall back to ROOT so the dir survives for profiling.
    base = ROOT if args.keep_work else _workdir_base()
    workdir = base / f"work_{int(time.time())}_{os.getpid()}"
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
    print(f"\n[done] {dt:.1f}s  ({NCALLS} gdal subprocesses, "
          f"avg {1000 * T_GDAL / max(NCALLS, 1):.0f} ms each, "
          f"wall = {dt:.1f}s, gdal = {T_GDAL:.1f}s)")


if __name__ == "__main__":
    main()
