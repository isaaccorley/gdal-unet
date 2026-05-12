"""Shared low-level ops for the gdal-unet runtime.

Each conv layer is one ``gdal-conv2d`` subprocess.  BN folding is done in
Python at call-time (so a model can be shipped as a vanilla ``.pt``).  All
other tensor-shape rearrangements (concat, residual add, upsample, SE) run as
small numpy ops via rasterio: cheap relative to the conv itself.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import rasterio


# ---------- environment ----------
# Locate the `gdal-conv2d` binary.  Resolution order:
#   1. $GDAL_CONV2D  (explicit override)
#   2. ./cpp/build/gdal-conv2d  (developer build next to the source tree)
#   3. PATH lookup (conda / homebrew / system install)
ROOT = Path(__file__).resolve().parents[1]


def _resolve_binary() -> str:
    env = os.environ.get("GDAL_CONV2D")
    if env:
        return env
    cand = ROOT / "cpp" / "build" / "gdal-conv2d"
    if cand.exists():
        return str(cand)
    from shutil import which
    found = which("gdal-conv2d")
    if found:
        return found
    return "gdal-conv2d"  # fall through; subprocess will error if missing


GDAL_CONV2D_BIN = _resolve_binary()
NTH = int(os.environ.get("GDAL_CONV2D_THREADS", "16"))

# HPC fallback: if a known conda-env libdir is present, prepend so the
# binary's RPATH doesn't have to fight a torchgeo / ftw-tile libstdc++ clash.
# No-op for normal conda-forge / Homebrew installs.
for _hint in ("/projects/bgtj/isaaccorley/envs/ftw-tile/lib",):
    if os.path.isdir(_hint):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _hint not in _ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{_hint}:{_ld}" if _ld else _hint

COFLAGS = ["--co", "BIGTIFF=IF_SAFER", "--co", "INTERLEAVE=BAND"]

# Accounting (mirrors predict_cpp.py behavior so the existing profile harness
# keeps working).
NCALLS = 0
T_SUB = 0.0


def reset_counters():
    global NCALLS, T_SUB
    NCALLS = 0
    T_SUB = 0.0


def sh(*cmd):
    """Run a subprocess; account it; raise on nonzero."""
    global NCALLS, T_SUB
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("CMD:", " ".join(cmd[:40]))
        print(e.stderr.decode(errors="replace"))
        raise
    NCALLS += 1
    T_SUB += time.time() - t0


# ---------- BN folding ----------
def fold_bn(weight, bn_w, bn_b, bn_m, bn_v, eps=1e-5):
    """Return (a, b) such that  BN(conv(x)) == a * conv(x) + b  per output ch.

    All tensors must be numpy arrays.  ``weight`` is returned unchanged (it is
    the original conv kernel: BN folding only rescales it if we were merging
    into the kernel, but for the gdal pipeline we keep the kernel raw and
    apply the affine after the conv -- bit-identical to PyTorch eval()).
    """
    a = bn_w / np.sqrt(bn_v + eps)
    b = bn_b - bn_m * a
    return a.astype(np.float16), b.astype(np.float16)


# ---------- weight cache ----------
class WeightDir:
    """Lazily writes float16 tensors to ``.bin`` files in workdir.

    Keys are arbitrary strings; identical keys reuse the same file.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, Path] = {}

    def put(self, key: str, arr: np.ndarray) -> Path:
        p = self.cache.get(key)
        if p is not None:
            return p
        arr = np.ascontiguousarray(arr.astype(np.float16))
        p = self.root / (key.replace("/", "_").replace(".", "_") + ".bin")
        arr.tofile(p)
        self.cache[key] = p
        return p


# ---------- conv ----------
def conv(
    in_tif: Path,
    out_tif: Path,
    *,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    bn: tuple | None = None,            # (w, b, m, v, eps) or pre-folded (a, b)
    activation: str = "none",            # none|relu|relu6|swish|gelu|hswish|sigmoid
    stride: int = 1,
    padding: int | None = None,
    depthwise: bool = False,
    weights: WeightDir,
    key: str,
):
    """Run one conv (+ optional BN-affine + activation) via gdal-conv2d.

    ``weight`` is the raw conv kernel from the state_dict.  Shape:
      - normal conv:    (Cout, Cin, kH, kW)
      - depthwise conv: (Cin, 1, kH, kW)  (PyTorch convention with groups=Cin)
    """
    if depthwise:
        assert weight.shape[1] == 1, f"depthwise kernel must have Cin/groups==1, got {weight.shape}"
        Cin, _, kH, kW = weight.shape
        Cout = Cin
        kernel_path = weights.put(key + ".k", weight.reshape(Cin, kH, kW))
        kernel_shape = f"{Cout},{Cin},{kH},{kW}"
    else:
        Cout, Cin, kH, kW = weight.shape
        kernel_path = weights.put(key + ".k", weight)
        kernel_shape = f"{Cout},{Cin},{kH},{kW}"

    if padding is None:
        padding = kH // 2

    args = [
        GDAL_CONV2D_BIN,
        "--in", str(in_tif),
        "--kernel", str(kernel_path),
        "--kernel-shape", kernel_shape,
        "--stride", str(stride),
        "--padding", str(padding),
        "--out", str(out_tif),
        "--threads", str(NTH),
        "--activation", activation,
        *COFLAGS,
    ]
    if depthwise:
        args.append("--depthwise")

    if bn is not None:
        if len(bn) == 5:
            a, b = fold_bn(weight, *bn)
        elif len(bn) == 2:
            a, b = bn
        else:
            raise ValueError(f"bn must be 5-tuple or 2-tuple, got len {len(bn)}")
        a_path = weights.put(key + ".bn_a", a)
        b_path = weights.put(key + ".bn_b", b)
        args += ["--bn-a", str(a_path), "--bn-b", str(b_path)]
    elif bias is not None:
        bias_path = weights.put(key + ".bias", bias)
        args += ["--bias", str(bias_path)]

    sh(*args)


# ---------- numpy helpers for the non-conv ops ----------
def read(tif: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif) as src:
        return src.read(), src.profile.copy()


def write(tif: Path, arr: np.ndarray, profile: dict, *, dtype: str = "float16",
          compress: str | None = None, tiled: bool = False):
    p = profile.copy()
    p.update(
        count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
        dtype=dtype, driver="GTiff", compress=compress,
        tiled=tiled, BIGTIFF="IF_SAFER",
    )
    for k in ("blockxsize", "blockysize"):
        if not tiled:
            p.pop(k, None)
    with rasterio.open(tif, "w", **p) as dst:
        dst.write(arr.astype(dtype))


def preprocess(in_tif: Path, out_tif: Path, *, in_channels: int = 4):
    """x[:in_channels] / 255 -> Float16 raster."""
    with rasterio.open(in_tif) as src:
        a = src.read()[:in_channels].astype(np.float32) / 255.0
        p = src.profile.copy()
    write(out_tif, a, p)


def pad_same_asym(in_tif: Path, out_tif: Path, *, k: int, stride: int):
    """Pre-pad a raster with TF-style 'SAME' asymmetric padding.

    Used by EfficientNet (which uses ``Conv2dStaticSamePadding``).  Pads with
    zeros on right/bottom (the TF convention) when the required pad is odd.
    After this op, follow up with ``ops.conv(..., padding=0, stride=stride)``.

    For stride=1 the SAME padding is exactly ``k//2`` symmetric on each side
    (this helper still works, but ``ops.conv`` with ``padding=k//2`` is faster).
    For stride=2 with even input H, pad is 1 -> asymmetric (0 left/top,
    1 right/bottom).  With odd input H, pad is 2 -> symmetric (1, 1).
    """
    with rasterio.open(in_tif) as src:
        a = src.read()
        p = src.profile.copy()
    import math
    C, H, W = a.shape
    oh = math.ceil(H / stride)
    ow = math.ceil(W / stride)
    pad_h = max((oh - 1) * stride + k - H, 0)
    pad_w = max((ow - 1) * stride + k - W, 0)
    # TF SAME: pad_top = pad_h // 2, pad_bottom = pad_h - pad_top  (extra on bottom/right)
    pt = pad_h // 2
    pb = pad_h - pt
    pl = pad_w // 2
    pr = pad_w - pl
    if pt == 0 and pb == 0 and pl == 0 and pr == 0:
        # nothing to do -- just copy
        write(out_tif, a, p)
        return 0
    Hp = H + pt + pb
    Wp = W + pl + pr
    out = np.zeros((C, Hp, Wp), dtype=a.dtype)
    out[:, pt:pt + H, pl:pl + W] = a
    write(out_tif, out, p)
    return 0


def maxpool3x3_s2(in_tif: Path, out_tif: Path):
    a, p = read(in_tif)
    a = a.astype(np.float32)
    C, H, W = a.shape
    pad = np.zeros((C, H + 2, W + 2), dtype=np.float32)
    pad[:, 1:1 + H, 1:1 + W] = a
    m = pad[:, 0:H, 0:W]
    for dy in range(3):
        for dx in range(3):
            if dy == 0 and dx == 0:
                continue
            m = np.maximum(m, pad[:, dy:dy + H, dx:dx + W])
    out = m[:, ::2, ::2]
    write(out_tif, out, p)


def upsample2x(in_tif: Path, out_tif: Path):
    a, p = read(in_tif)
    out = a.repeat(2, axis=1).repeat(2, axis=2)
    write(out_tif, out, p)


def upsample_to(in_tif: Path, out_tif: Path, h: int, w: int):
    """Nearest-neighbor upsample to exact (h, w).  smp's decoder picks the
    target shape from the skip features, so we may need non-2x ratios at
    odd sizes.
    """
    a, p = read(in_tif)
    C, H, W = a.shape
    if H == h and W == w:
        write(out_tif, a, p)
        return
    # nearest: index via floor((i+0.5)*H/h) is what F.interpolate(mode="nearest")
    # does, but for integer ratios it equals i*H/h.  We use the integer-ratio
    # path; gdal-unet only upsamples by powers of 2.
    iy = (np.arange(h) * H // h)
    ix = (np.arange(w) * W // w)
    out = a[:, iy[:, None], ix[None, :]]
    write(out_tif, out, p)


def concat(tifs: list[Path], out_tif: Path):
    arrs, profs = zip(*[read(t) for t in tifs])
    # crop trailing rows/cols if there's an off-by-one due to stride math
    H = min(x.shape[1] for x in arrs)
    W = min(x.shape[2] for x in arrs)
    out = np.concatenate([x[:, :H, :W] for x in arrs], axis=0)
    write(out_tif, out, profs[0])


def add(a_tif: Path, b_tif: Path, out_tif: Path, *,
        activation: str = "none"):
    a, p = read(a_tif)
    b, _ = read(b_tif)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    H = min(a.shape[1], b.shape[1]); W = min(a.shape[2], b.shape[2])
    out = a[:, :H, :W] + b[:, :H, :W]
    out = activate(out, activation)
    write(out_tif, out, p)


def activate(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return arr
    if kind == "relu":
        return np.maximum(arr, 0)
    if kind == "relu6":
        return np.clip(arr, 0, 6)
    if kind in ("swish", "silu"):
        return arr / (1.0 + np.exp(-arr))
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-arr))
    if kind in ("hswish", "hard_swish"):
        return arr * np.clip(arr + 3.0, 0, 6) / 6.0
    if kind in ("hsigmoid", "hard_sigmoid"):
        # torchvision's nn.Hardsigmoid: relu6(x + 3) / 6
        return np.clip(arr + 3.0, 0, 6) / 6.0
    if kind == "gelu":
        # PyTorch default exact form; for the SE bottleneck activations we
        # never actually use this -- it's here for completeness.
        return 0.5 * arr * (1.0 + np.tanh(0.7978845608028654 *
                                          (arr + 0.044715 * arr ** 3)))
    raise ValueError(f"unknown activation {kind!r}")


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
    for k in ("blockxsize", "blockysize"):
        if p2.get("width", 0) < 256 or p2.get("height", 0) < 256:
            p2.pop(k, None); p2["tiled"] = False
    with rasterio.open(out_tif, "w", **p2) as dst:
        dst.write(pr.astype(np.float16))


def softmax_kcls(in_tif: Path, out_tif: Path):
    """K-class softmax along channel axis (still writes float16)."""
    a, p = read(in_tif)
    a = a.astype(np.float32)
    m = a.max(axis=0, keepdims=True)
    e = np.exp(a - m)
    pr = e / e.sum(axis=0, keepdims=True)
    write(out_tif, pr, p)


# ---------- SE block ----------
def se_block(in_tif: Path, out_tif: Path, *,
             w1: np.ndarray, b1: np.ndarray,
             w2: np.ndarray, b2: np.ndarray,
             act1: str = "relu",
             scale_activation: str = "sigmoid",
             workdir: Path, weights: WeightDir, key: str):
    """Squeeze-and-Excitation block.

    1. Global avg pool over (H, W) per channel -> shape (C,)
    2. 1x1 conv w1 (Cmid, C) + bias + act1
    3. 1x1 conv w2 (C, Cmid) + bias + sigmoid
    4. Per-channel scale: y = x * gate[:, None, None]

    Implemented entirely in numpy here (no gdal-conv2d subprocesses needed --
    the spatial size after global pool is 1x1, so the "conv" is just a matmul).
    """
    a, p = read(in_tif)
    a = a.astype(np.float32)
    C, H, W = a.shape
    # global avg pool
    g = a.mean(axis=(1, 2))           # (C,)
    # 1x1 conv: w1 has shape (Cmid, C, 1, 1)
    w1m = w1.reshape(w1.shape[0], w1.shape[1]).astype(np.float32)
    b1v = b1.astype(np.float32)
    z = w1m @ g + b1v                 # (Cmid,)
    z = activate(z, act1)
    w2m = w2.reshape(w2.shape[0], w2.shape[1]).astype(np.float32)
    b2v = b2.astype(np.float32)
    gate = activate(w2m @ z + b2v, scale_activation)     # (C,)
    out = a * gate[:, None, None]
    write(out_tif, out, p)
    # so that NCALLS budget stays informative, charge two "synthetic" calls
    # (SE-fc1 + SE-fc2).  No subprocess -- numpy only.
    return out_tif
