"""Numpy/rasterio helpers for the non-conv pieces of the forward pass.

These run in-process (no subprocess overhead). The conv-BN-ReLU primitive
is shelled out to the `gdal-unet-conv` C++ binary in `conv_runner.py`.
"""

import os
import time
from pathlib import Path

import numpy as np
import rasterio


# ---------- raw .bin weight cache ----------
class WeightDir:
    """Materialise weights from a loaded .npz into raw Float16 .bin files.

    The C++ binary reads kernels / BN / bias as raw little-endian Float16
    buffers; we write each key once and reuse the cached path.
    """

    def __init__(self, root: Path, npz):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._npz = npz
        self.cache: dict[str, Path] = {}

    def get(self, key: str) -> Path:
        p = self.cache.get(key)
        if p is not None:
            return p
        arr = np.ascontiguousarray(self._npz[key].astype(np.float16))
        p = self.root / (key.replace(".", "_") + ".bin")
        arr.tofile(p)
        self.cache[key] = p
        return p

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._npz[key].shape)


# ---------- raster io ----------
def read(tif: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif) as src:
        return src.read(), src.profile.copy()


def write(tif: Path, arr: np.ndarray, profile: dict):
    p = profile.copy()
    p.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
             dtype="float16", driver="GTiff", compress=None,
             tiled=False, BIGTIFF="IF_SAFER")
    with rasterio.open(tif, "w", **p) as dst:
        dst.write(arr.astype(np.float16))


# ---------- ops ----------
def preprocess(in_tif: Path, out_tif: Path):
    """x / 255 -> Float16 raster (first 4 bands)."""
    with rasterio.open(in_tif) as src:
        a = src.read()[:4].astype(np.float32) / 255.0
        p = src.profile.copy()
    write(out_tif, a, p)


def maxpool3x3_s2(in_tif: Path, out_tif: Path):
    """nn.MaxPool2d(3, 2, 1) on non-negative input."""
    a, p = read(in_tif)
    a = a.astype(np.float32)
    C, H, W_ = a.shape
    pad = np.zeros((C, H + 2, W_ + 2), dtype=np.float32)
    pad[:, 1:1 + H, 1:1 + W_] = a
    m = pad[:, 0:H, 0:W_]
    for dy in range(3):
        for dx in range(3):
            if dy == 0 and dx == 0:
                continue
            m = np.maximum(m, pad[:, dy:dy + H, dx:dx + W_])
    out = m[:, ::2, ::2]
    write(out_tif, out, p)


def upsample2x(in_tif: Path, out_tif: Path):
    """Nearest-neighbour 2x upsample (np.repeat -> matches PyTorch nearest)."""
    a, p = read(in_tif)
    out = a.repeat(2, axis=1).repeat(2, axis=2)
    write(out_tif, out, p)


def concat(tifs: list[Path], out_tif: Path):
    arrs, profs = zip(*[read(t) for t in tifs])
    out = np.concatenate(arrs, axis=0)
    write(out_tif, out, profs[0])


def add_relu(a_tif: Path, b_tif: Path, out_tif: Path, *, relu: bool = True):
    a, p = read(a_tif)
    b, _ = read(b_tif)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    out = a + b
    if relu:
        np.maximum(out, 0, out=out)
    write(out_tif, out, p)


def softmax_2cls(in_tif: Path, out_tif: Path):
    a, p = read(in_tif)
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


# ---------- workdir helpers ----------
def workdir_base() -> Path:
    for p in (os.environ.get("SLURM_TMPDIR"), "/dev/shm", "/tmp"):
        if p and Path(p).is_dir() and os.access(p, os.W_OK):
            return Path(p)
    return Path.cwd()


def make_workdir(base: Path) -> Path:
    wd = base / f"work_cpp_{int(time.time())}_{os.getpid()}"
    wd.mkdir(parents=True, exist_ok=True)
    return wd
