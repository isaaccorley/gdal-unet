#!/usr/bin/env bash
# Verify each new gdal-conv2d mode against a numpy/PyTorch reference on 8x8 inputs.
# Exits nonzero on first mismatch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="${GDAL_CONV2D:-$ROOT/cpp/build/gdal-conv2d}"
PYTHON="${PYTHON:-python3}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

cat > _ref.py <<'PY'
import os, sys, numpy as np, rasterio
import torch, torch.nn.functional as F

WORK = os.environ["WORK"]
np.random.seed(0)

def write(path, arr):
    arr = arr.astype(np.float16)
    C, H, W = arr.shape
    with rasterio.open(path, "w", driver="GTiff",
                       count=C, height=H, width=W, dtype="float16") as dst:
        dst.write(arr)

def read(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

def approx(a, b, tol=2e-2):
    # fp16 round-trips through tiff can be ~1e-3; we use a moderate tol.
    d = np.abs(a - b).max()
    return d, d < tol

if sys.argv[1] == "make":
    # Make 8x8 inputs for each mode
    x = np.random.randn(3, 8, 8).astype(np.float32)
    write(f"{WORK}/in_3x8x8.tif", x)

    y = np.random.randn(3, 8, 8).astype(np.float32)
    write(f"{WORK}/in2_3x8x8.tif", y)

    z = np.random.randn(2, 8, 8).astype(np.float32)
    write(f"{WORK}/in_2x8x8.tif", z)

elif sys.argv[1] == "scale":
    x = read(f"{WORK}/in_3x8x8.tif")
    got = read(f"{WORK}/out.tif")
    ref = x * 0.25
    d, ok = approx(got, ref)
    print(f"scale: maxdiff={d:.4g} ok={ok}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "maxpool":
    x = read(f"{WORK}/in_3x8x8.tif")
    got = read(f"{WORK}/out.tif")
    ref = F.max_pool2d(torch.from_numpy(x).unsqueeze(0), 3, 2, 1).squeeze(0).numpy()
    d, ok = approx(got, ref)
    print(f"maxpool: maxdiff={d:.4g} ok={ok} shape={got.shape} ref={ref.shape}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "upsample":
    x = read(f"{WORK}/in_3x8x8.tif")
    got = read(f"{WORK}/out.tif")
    ref = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    d, ok = approx(got, ref)
    print(f"upsample: maxdiff={d:.4g} ok={ok}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "add":
    a = read(f"{WORK}/in_3x8x8.tif")
    b = read(f"{WORK}/in2_3x8x8.tif")
    got = read(f"{WORK}/out.tif")
    ref = np.maximum(a + b, 0)  # with --activation relu
    d, ok = approx(got, ref)
    print(f"add+relu: maxdiff={d:.4g} ok={ok}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "concat":
    a = read(f"{WORK}/in_3x8x8.tif")
    b = read(f"{WORK}/in_2x8x8.tif")
    got = read(f"{WORK}/out.tif")
    ref = np.concatenate([a, b], axis=0)
    d, ok = approx(got, ref)
    print(f"concat: maxdiff={d:.4g} ok={ok} shape={got.shape}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "softmax":
    x = read(f"{WORK}/in_3x8x8.tif")
    got = read(f"{WORK}/out.tif")
    m = x.max(axis=0, keepdims=True)
    e = np.exp(x - m)
    ref = e / e.sum(axis=0, keepdims=True)
    d, ok = approx(got, ref)
    print(f"softmax: maxdiff={d:.4g} ok={ok}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "conv":
    x = read(f"{WORK}/in_3x8x8.tif")
    # 4 outchannels, 3 inch, 3x3
    w = np.random.randn(4, 3, 3, 3).astype(np.float16)
    w.tofile(f"{WORK}/k.bin")
    bias = np.random.randn(4).astype(np.float16)
    bias.tofile(f"{WORK}/bias.bin")
    # save args for shell
    open(f"{WORK}/kshape", "w").write("4,3,3,3")
    got = read(f"{WORK}/out.tif")
    ref = F.conv2d(torch.from_numpy(x).unsqueeze(0),
                   torch.from_numpy(w.astype(np.float32)),
                   bias=torch.from_numpy(bias.astype(np.float32)),
                   stride=1, padding=1).squeeze(0).numpy()
    d, ok = approx(got, ref, tol=5e-2)
    print(f"conv: maxdiff={d:.4g} ok={ok}")
    sys.exit(0 if ok else 1)

elif sys.argv[1] == "conv-prep":
    # writes kernel + bias before running gdal-conv2d
    w = np.random.randn(4, 3, 3, 3).astype(np.float16)
    w.tofile(f"{WORK}/k.bin")
    bias = np.random.randn(4).astype(np.float16)
    bias.tofile(f"{WORK}/bias.bin")
PY

export WORK
echo "=== preparing inputs ==="
$PYTHON _ref.py make

# fix seed alignment: regenerate conv kernel
$PYTHON _ref.py conv-prep

echo "=== scale ==="
"$BIN" --mode scale --in "$WORK/in_3x8x8.tif" --scale 0.25 --out "$WORK/out.tif"
$PYTHON _ref.py scale

echo "=== maxpool ==="
"$BIN" --mode maxpool --in "$WORK/in_3x8x8.tif" --kernel-size 3 --stride 2 --padding 1 --out "$WORK/out.tif"
$PYTHON _ref.py maxpool

echo "=== upsample ==="
"$BIN" --mode upsample --in "$WORK/in_3x8x8.tif" --scale 2 --method nearest --out "$WORK/out.tif"
$PYTHON _ref.py upsample

echo "=== add ==="
"$BIN" --mode add --in "$WORK/in_3x8x8.tif" --in "$WORK/in2_3x8x8.tif" --activation relu --out "$WORK/out.tif"
$PYTHON _ref.py add

echo "=== concat ==="
"$BIN" --mode concat --in "$WORK/in_3x8x8.tif" --in "$WORK/in_2x8x8.tif" --out "$WORK/out.tif"
$PYTHON _ref.py concat

echo "=== softmax ==="
"$BIN" --mode softmax --in "$WORK/in_3x8x8.tif" --out "$WORK/out.tif"
$PYTHON _ref.py softmax

echo "=== conv ==="
"$BIN" --mode conv --in "$WORK/in_3x8x8.tif" --kernel "$WORK/k.bin" --kernel-shape 4,3,3,3 --bias "$WORK/bias.bin" --padding 1 --stride 1 --out "$WORK/out.tif"
$PYTHON _ref.py conv

echo "=== georeferencing ==="
$PYTHON - <<'PY'
import os, numpy as np, rasterio
from rasterio.transform import from_origin
WORK = os.environ["WORK"]
tr = from_origin(west=500000.0, north=4500000.0, xsize=10.0, ysize=10.0)
x = np.random.randn(3, 16, 16).astype(np.float16)
with rasterio.open(f"{WORK}/g_in.tif", "w", driver="GTiff",
                   count=3, height=16, width=16, dtype="float16",
                   crs="EPSG:32616", transform=tr) as d:
    d.write(x)
PY

# same-conv: out transform must equal in transform
"$BIN" --mode conv --in "$WORK/g_in.tif" --kernel "$WORK/k.bin" --kernel-shape 4,3,3,3 \
       --bias "$WORK/bias.bin" --padding 1 --stride 1 --out "$WORK/g_same.tif"

# strided downsample: pixel size doubles, origin shifts -0.5 input pixel
"$BIN" --mode conv --in "$WORK/g_in.tif" --kernel "$WORK/k.bin" --kernel-shape 4,3,3,3 \
       --bias "$WORK/bias.bin" --padding 1 --stride 2 --out "$WORK/g_stride.tif"

# 2x2 maxpool stride 2: pixel size doubles, origin unchanged
"$BIN" --mode maxpool --in "$WORK/g_in.tif" --kernel-size 2 --stride 2 --padding 0 \
       --out "$WORK/g_pool.tif"

# upsample x2: pixel size halves, origin unchanged
"$BIN" --mode upsample --in "$WORK/g_in.tif" --scale 2 --method nearest \
       --out "$WORK/g_up.tif"

$PYTHON - <<'PY'
import os, sys, rasterio
WORK = os.environ["WORK"]
t_in   = rasterio.open(f"{WORK}/g_in.tif").transform
t_same = rasterio.open(f"{WORK}/g_same.tif").transform
t_str  = rasterio.open(f"{WORK}/g_stride.tif").transform
t_pool = rasterio.open(f"{WORK}/g_pool.tif").transform
t_up   = rasterio.open(f"{WORK}/g_up.tif").transform
def approx(a, b, tol=1e-6):
    return all(abs(a[i] - b[i]) < tol for i in range(6))
ok = True
# same conv: identity
if not approx(t_same, t_in):
    print("FAIL same-conv:", t_same, "expected", t_in); ok = False
else: print("ok same-conv == input transform")
# strided conv K=3 S=2 P=1: shift = (3-2)/2 - 1 = -0.5 input px in each dim
expect_str = (20.0, 0.0, 500000.0 - 0.5*10.0, 0.0, -20.0, 4500000.0 - 0.5*(-10.0))
if not approx(t_str, expect_str):
    print("FAIL strided-conv:", t_str, "expected", expect_str); ok = False
else: print("ok strided-conv")
# maxpool 2x2 S=2 P=0: no shift, px size *= 2
expect_pool = (20.0, 0.0, 500000.0, 0.0, -20.0, 4500000.0)
if not approx(t_pool, expect_pool):
    print("FAIL maxpool:", t_pool, "expected", expect_pool); ok = False
else: print("ok maxpool")
# upsample x2: no shift, px size /= 2
expect_up = (5.0, 0.0, 500000.0, 0.0, -5.0, 4500000.0)
if not approx(t_up, expect_up):
    print("FAIL upsample:", t_up, "expected", expect_up); ok = False
else: print("ok upsample")
sys.exit(0 if ok else 1)
PY

echo "ALL PASS"
