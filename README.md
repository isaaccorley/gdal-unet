# gdal-unet

> **Multi-backbone branch.** This branch adds direct PyTorch `.pt` loading
> (no `.npz` step) and support for additional smp encoders.  See
> [`Multi-backbone usage`](#multi-backbone-usage) below.

## Multi-backbone usage

```bash
# Save a fresh smp model (any encoder smp supports)
python -c "
import torch, segmentation_models_pytorch as smp
m = smp.Unet(encoder_name='resnet50', encoder_weights=None,
             in_channels=4, classes=2)
torch.save(m.state_dict(), 'unet_r50.pt')"

# Run the gdal-CLI forward pass directly from the .pt (BN folding done
# at load time, kernel + (a, b) tensors materialized lazily per layer).
python -m gdal_unet.predict samples/1717_image.tif probs.tif \
    --ckpt unet_r50.pt --arch resnet50
```

Supported encoders on this branch:

| encoder | status | notes |
|---|---|---|
| `resnet18` | bit-close | original baseline |
| `resnet34` | bit-close | (same path as r18; auto-detected) |
| `resnet50`, `resnet101`, `resnet152` | bit-close | Bottleneck block |
| `mobilenet_v2` | bit-close | depthwise + 1x1 expand/project + ReLU6 |
| `mobilenet_v3_*` | stub | all primitives wired in `ops`, encoder TBD |
| `efficientnet-b*` | stub | swish + SE wired, MBConv encoder TBD |

Verify a backbone:

```bash
python tests/test_backbones.py resnet50 mobilenet_v2
```

The test instantiates a fresh smp.Unet, saves a `.pt`, runs both PyTorch and
the gdal-CLI pipeline on a synthetic 128x128 4-band input, and prints
per-stage cosine similarity + final softmax max-diff.

## New primitives in `gdalnn_conv`

| flag | meaning |
|---|---|
| `--activation {none,relu,relu6,swish,gelu,hswish,sigmoid}` | replaces single-purpose `--relu` (still accepted) |
| `--depthwise` | kernel shape becomes `(Cin, 1, kH, kW)`; per-band conv (groups=Cin) |
| `--stride N` | stride is now any positive integer (was 1 or 2) |

Squeeze-and-Excitation is implemented in `gdal_unet/ops.py:se_block` as a
numpy global-pool + two 1x1 conv "fc" layers + per-channel scale (cheap; no
extra subprocesses).



Run a U-Net (or any conv-based segmentation model) inference pipeline entirely through `gdal` CLI primitives. Loads pretrained PyTorch weights, runs the forward pass through `gdal raster ...` subprocesses (or, for speed, a small custom binary that uses GDAL for I/O). Outputs a probability GeoTIFF that matches the PyTorch reference bit-close.

Reference model: [`isaaccorley/chesapeakersc`](https://huggingface.co/isaaccorley/chesapeakersc) — `smp.Unet(resnet18, in_channels=4, classes=2)`, 14.3 M params, trained on NAIP RGBN for road segmentation. Same architecture pattern works for any conv-based U-Net (different encoder, different classes).

## Three back-ends, all bit-close to PyTorch

| | `predict_gdal.py` | `vrt/build_vrt.py` | `gdal-unet predict` |
|---|---|---|---|
| **Math runtime** | many `gdal raster pipeline / calc` per chunk | one `gdal_translate` per layer (VRT pixel functions) | `gdal-conv2d` (custom C++ binary) per layer |
| **Subprocesses / forward** | ~2185 | 31 | **31** |
| **Wall on 16-CPU node** | ~520 s | 131 s | **7.5 s** |
| **Peak RSS** | many GB on disk | 672 MB | **156 MB** |
| **Per-stage cosine vs PyTorch** | 1.0000 | 1.0000 | 1.0000 |
| **IoU vs ground truth** | 0.6372 | 0.6372 | 0.6372 |

PyTorch reference IoU on the same input: 0.6374. The C++ path is **~70× faster** than pure-CLI; the VRT path is **~4× faster** while staying in stock GDAL.

## What's here

```
gdal_unet/                       Python package (user-facing CLI)
  cli.py                         `gdal-unet` console entry point
  conv_runner.py                 shell wrapper around the C++ binary
  ops.py                         numpy/rasterio helpers (pad, pool, upsample, ...)
  backbones/resnet18.py          smp.Unet(resnet18) forward pass driver
pyproject.toml                   declares the `gdal-unet` entry-point

predict_gdal.py      pure-CLI chunked-diagonal forward pass     (the baseline)
export_weights.py    pull weights from a PyTorch .pt -> .npz (Float16, BN folded)
model_weights.npz    28 MB pretrained weights (chesapeakersc)
profile_layers.py    stage-by-stage cosine diff (used to debug)
samples/             5 NAIP patches + GT masks from the ChesapeakeRSC repo

vrt/
  build_vrt.py       generate one VRT per conv layer; gdal_translate per layer

cpp/
  src/gdal_conv2d.cpp  one conv-BN-ReLU layer, OpenMP-parallel, GDAL I/O
  CMakeLists.txt          portable build (Linux, macOS arm64/x86_64)
                          installs the `gdal-conv2d` binary

.github/workflows/build.yml   CI: build binaries for linux-{x86_64,arm64}, macos-{arm64,x86_64}
conda-recipe/                 conda-forge recipe (see "Deployment" below)
```

## Architecture (what the forward pass does)

```
input NAIP (4-band uint8)
  ├─ x / 255 -> Float16
  ├─ stem        : 7×7 conv stride 2 -> BN -> ReLU
  ├─ maxpool 3×3 stride 2
  ├─ layer1..layer4 : 8 ResNet-18 BasicBlocks
  │                   (conv -> BN -> ReLU -> conv -> BN -> +skip -> ReLU)
  ├─ decoder.0..decoder.4 : 5 smp U-Net decoder blocks
  │                         (upsample -> concat skip -> conv -> BN -> ReLU)*2
  ├─ head        : 3×3 conv -> 2-class logits
  └─ softmax     -> probs.tif (2-band Float16)
```

## Engineering notes (the interesting parts)

### Pure-CLI variant (`predict_gdal.py`)

- **Chunked-diagonal multi-channel conv.** `gdal raster neighbors` with K kernels on an N-band raster emits N·K bands (full cross-product). For a conv layer with Cout outputs, group Cin into chunks of K and pass K·Cout kernels per chunk; a follow-up `gdal raster calc` picks diagonal bands at indices `ic_in_chunk*(K·Cout+1) + oc·K + 1` and sums them. The final pairwise reduction over chunks fuses BN + ReLU + bias into the last sum.
- **Parallel chunks via `ThreadPoolExecutor`.** All chunks within a layer are independent; partials reduce in a parallel pairwise tree.
- **Float16 throughout.** `--ot Float16` on every gdal call. No measurable accuracy loss.
- **Boundary padding.** PyTorch `conv2d(padding=N)` zero-pads, `gdal raster neighbors` edge-replicates. We zero-pad the raster ourselves with rasterio+numpy before each `neighbors` call and crop the boundary off after.
- **Stride-2 alignment.** PyTorch samples at `(0, 0), (0, 2), …`; `gdal raster reproject -r nearest` samples at pixel centers (off by one). We `data[:, ::2, ::2]` via numpy instead. Same fix for `MaxPool2d(3, 2, 1)` and `Upsample(2, nearest)` → `np.repeat`.

### VRT variant (`vrt/build_vrt.py`)

- One VRT XML per conv layer: `<VRTDerivedRasterBand>` with `<PixelFunctionType>expression</PixelFunctionType>` (built-in muparser, no Python pixel functions needed) wrapping Cin `<KernelFilteredSource>` children. The expression is `bn_a * (B1 + B2 + ... + BCin) + bn_b`, with `max(0, …)` for ReLU.
- `gdal_translate` materializes one layer per call. Strides / maxpool / upsample / concat use the same numpy helpers as the pure-CLI variant.

### C++ binary (`cpp/src/gdal_conv2d.cpp`)

- 330 lines, OpenMP over output channels, scalar inner loops.
- Reads input via GDAL into a float32 buffer, kernel/BN/bias from raw `.bin` files (Float16), zero-pads, conv-BN-ReLU, optional stride-2 sampling, writes Float16 output preserving CRS/geotransform.
- Build: `cmake -S cpp -B cpp/build && cmake --build cpp/build`. Requires GDAL ≥ 3.11 and OpenMP. On macOS install `libomp` via Homebrew or conda.

## Usage

```bash
# one-time: pull the trained weights from the .pt checkpoint into model_weights.npz
python export_weights.py

# Option 1 — pure CLI (slow, no custom code)
python predict_gdal.py samples/1717_image.tif samples/1717_probs.tif

# Option 2 — VRT per layer (4× faster, still no custom code)
python vrt/build_vrt.py samples/1717_image.tif samples/1717_probs.tif

# Option 3 — C++ binary (70× faster, via the gdal-unet CLI)
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
pip install -e .
gdal-unet predict samples/1717_image.tif samples/1717_probs.tif \
    --arch unet-resnet18 --weights model_weights.npz
```

The `gdal-unet` CLI is a thin Python orchestrator; under the hood it
fires one `gdal-conv2d` subprocess per conv layer (~31 calls total).
You can point it at an alternate binary with `--binary` or `$GDAL_CONV2D`.

## Deployment

### Supported platforms

| Platform | Status |
|---|---|
| Linux x86_64 | ✅ CI-built |
| Linux arm64 | ✅ CI-built |
| macOS arm64 (Apple Silicon) | ✅ CI-built |
| macOS x86_64 (Intel) | ✅ CI-built |
| Windows | ❌ not built |

Windows is unsupported by default — the geospatial-ML-via-CLI audience overlaps very little with Windows users (most Windows geo-ML happens inside ArcGIS or via WSL2, which is Linux anyway). Adding it later is a small diff: add `windows-latest` to the matrix in `.github/workflows/build.yml` and remove `skip: true  # [win]` from `conda-recipe/meta.yaml`. Patches welcome if anyone actually needs it.

### Binary releases (GitHub Actions)

CI builds `gdal-conv2d` plus the Python package for the four supported platforms on every push and tag. See `.github/workflows/build.yml`. Tag-triggered runs (`v*`) attach the tarballs to a GitHub Release. End users download the right tarball and run:

```bash
tar -xzf gdal-unet-macos-arm64.tar.gz
cd gdal-unet-macos-arm64
./gdal-conv2d --help
pip install -e .                                  # installs the `gdal-unet` console script
gdal-unet predict in.tif out.tif --weights model_weights.npz
```

The binary RPATH is set to find `libgdal` next to it (`@loader_path/../lib` on macOS, `$ORIGIN/../lib` on Linux), so the user only needs GDAL installed on their system in a standard location.

### conda-forge (the right long-term home)

See [`conda-recipe/`](conda-recipe/) for the meta.yaml. Submission flow:

1. **Fork** `conda-forge/staged-recipes` on GitHub
2. Create branch and add the recipe at `recipes/gdal-unet/{meta.yaml,build.sh,bld.bat}` (the files in this repo's `conda-recipe/` directory)
3. Open a PR to `conda-forge/staged-recipes`
4. conda-forge CI builds for linux-64, linux-aarch64, osx-64, osx-arm64 (the recipe pins `libgdal >=3.11` and `llvm-openmp`/`libomp` for macOS)
5. A conda-forge maintainer reviews. Typical turn-around: 1–2 weeks.
6. On merge, conda-forge auto-creates `gdal-unet-feedstock` repo. From then on, **version bumps land via PR to the feedstock**; CI rebuilds and republishes.
7. End users: `conda install -c conda-forge gdal-unet`

### HuggingFace for weights

`model_weights.npz` (28 MB) lives next to the .pt checkpoint at [`isaaccorley/chesapeakersc`](https://huggingface.co/isaaccorley/chesapeakersc). The conda package doesn't bundle weights; the user downloads them once with `huggingface-cli download`.

## Why this exists

Because it's funny. Almost certainly impractical for real deployment compared to PyTorch / ONNX. But it works, and the parity story is solid (cosine 1.0000 at every stage, IoU off by 0.0002 from PyTorch).

The interesting outcome is that the lessons generalize:
- **Boundary padding** and **stride alignment** are non-trivial when emulating tensor ops on raster primitives.
- **Subprocess startup dominates** at ResNet18 scale (~2000 calls × 100 ms init = 200 s minimum). The 70× C++ speedup is entirely about collapsing that.
- **VRT pixel functions** can express most of a conv layer at zero custom code cost, but break at strided resampling — useful as a "no custom binary" option.
