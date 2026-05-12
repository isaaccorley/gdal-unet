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

echo "ALL PASS"
