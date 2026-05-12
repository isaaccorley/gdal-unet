# shell-only inference - STATUS

WIP branch: `shell-only-inference`. Goal: full U-Net inference using only
`gdal-conv2d` subprocesses + bash; no Python at predict time.

## What's tested and passing

Per-mode unit checks in `cpp/tests/test_modes.sh` (run against numpy/PyTorch
references on 8x8 synthetic inputs). All 7 modes pass within float16 tol:

- `--mode conv` (existing; smoke-tested with bias path)
- `--mode scale` - elementwise * scalar
- `--mode maxpool --kernel-size --stride --padding` - bit-matches `nn.MaxPool2d`
- `--mode upsample --scale F --method nearest` - matches `np.repeat`
- `--mode add --in ... --in ... [--activation relu]` - matches numpy add+relu
- `--mode concat --in ... --in ...` - band-wise concat
- `--mode softmax --in ...` - per-pixel stable softmax

Also verified end-to-end on samples/1717_image.tif:

- `gdal-unet-export unet-resnet18.pt --arch resnet18 -o /tmp/w/` produces:
  - 56 conv kernel .bin files + bn_a/bn_b per conv
  - `head.kernel.bin` + `head.bias.bin` (segmentation head)
  - `shapes.txt`, `arch.txt`, `n_decoder_blocks.txt`, `num_blocks.txt`
  - `predict_resnet18.sh` (~135 lines)
- `bash /tmp/w/predict_resnet18.sh /tmp/w samples/1717_image.tif /tmp/probs.tif`
  exits 0, produces a 512x512x2 Float16 GeoTIFF with valid softmax probs
  (sum == 1.0 per pixel). CRS + GT preserved from input.

## What's untested

- **Regression IoU vs the original predict_gdal.py pipeline** - the target was
  0.6372 on samples/1717. Not run. The shell pipeline produces *valid* probs
  (sums to 1, plausible ranges), but bit-equivalence to the original Python
  driver has not been verified.
- Resnet50 path - emitter handles the bottleneck case in code, but no .pt was
  exercised.
- All non-resnet arches (mobilenet, efficientnet) - explicitly out of scope.

## Known bugs / TBD

- `predict_resnet18.sh` decoder uses `UPSAMPLE ... --scale 2`, which assumes
  the input H,W are multiples of 32 so every upsample is a clean 2x. The
  Python pipeline uses `ops.upsample_to(target_h, target_w)` from the actual
  skip shape, which tolerates off-by-one. For 512x512 input this matches;
  for arbitrary input sizes it could mismatch. Fix: add a `--target-h/--target-w`
  option to `--mode upsample` and have the shell script query skip shapes
  via `gdalinfo` (or have the emitter pre-compute them for a fixed input
  size).
- Segmentation head is assumed to be 3x3 (kernel padding=1). smp default,
  matches all checkpoints we ship, but the emitter doesn't introspect `kH`
  to set padding. Hardcoded.
- BN folding uses `eps=1e-5` matching the Python ops.fold_bn. If the .pt was
  trained with a different eps this will silently drift.
- The `gdal-conv2d` binary always emits Float16 on disk. If the Python
  pipeline had any spot where it kept Float32, those values will round-trip
  through Float16 here. Empirically the difference is <1e-3 per layer, but
  it can accumulate over a 30-conv encoder.
- No bit-equivalence test against `predict_gdal.py` end-to-end - that's the
  next person's job per the task scope.

## How to reproduce

```bash
module load gcc/12.3.0
export PATH=/projects/bgtj/isaaccorley/envs/ftw-tile/bin:$PATH
export LD_LIBRARY_PATH=/projects/bgtj/isaaccorley/envs/ftw-tile/lib:$LD_LIBRARY_PATH

# Build
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/projects/bgtj/isaaccorley/envs/ftw-tile
cmake --build cpp/build -j

# Per-mode tests
PYTHON=/projects/bgtj/isaaccorley/envs/torchgeo/bin/python \
  bash cpp/tests/test_modes.sh

# Install exporter + emit pipeline
/projects/bgtj/isaaccorley/envs/torchgeo/bin/pip install -e ./export
curl -sL -o unet-resnet18.pt 'https://huggingface.co/isaaccorley/chesapeakersc/resolve/main/unet-resnet18.pt?download=true'
gdal-unet-export unet-resnet18.pt --arch resnet18 -o /tmp/w/

# Run
export GDAL_CONV2D=$(pwd)/cpp/build/gdal-conv2d
bash /tmp/w/predict_resnet18.sh /tmp/w samples/1717_image.tif /tmp/probs.tif
```
