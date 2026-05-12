# gdal-unet

Run a pretrained U-Net inference pipeline through one small C++ binary
(`gdal-conv2d`) plus a generated shell script. **No Python at deploy
time** — just bash and `gdal-conv2d` on PATH.

Reference model:
[`isaaccorley/chesapeakersc`](https://huggingface.co/isaaccorley/chesapeakersc)
— `smp.Unet(resnet18, in_channels=4, classes=2)`, 14.3 M params, trained on
NAIP RGBN for road segmentation. The same architecture pattern (and
`gdal-unet-export`) works for any conv-based U-Net.

## Two artifacts, two audiences

| Artifact | For | Install |
|---|---|---|
| `gdal-conv2d` (C++ binary) | end users running inference | tarball from the [latest release](https://github.com/isaaccorley/gdal-unet/releases/latest); `conda install -c conda-forge gdal-conv2d` is pending ([staged-recipes#33314](https://github.com/conda-forge/staged-recipes/pull/33314)) |
| `gdal-unet-export` (Python) | model authors converting checkpoints | `pip install git+https://github.com/isaaccorley/gdal-unet.git#subdirectory=export` |

## Inference flow

```bash
# One-time conversion (requires PyTorch):
gdal-unet-export model.pt --arch resnet18 -o weights/

# weights/ contains:
#   arch.txt
#   <layer>.kernel.bin   raw little-endian Float16
#   <layer>.bn_a.bin     pre-folded BN affine
#   <layer>.bn_b.bin
#   <layer>.bias.bin     for the head
#   predict_resnet18.sh  ~135 lines of bash

# Inference, no Python:
bash weights/predict_resnet18.sh weights/ input.tif output.tif
```

Every line of `predict_<arch>.sh` is a `gdal-conv2d --mode <X> ...`
invocation. One binary on PATH, libgdal as the only library dependency.

## Architecture (resnet18 example)

```
input NAIP (4-band uint8)
  ├─ scale   x / 255              -> Float16
  ├─ conv    7×7 stride 2 + BN + ReLU                   (stem)
  ├─ maxpool 3×3 stride 2
  ├─ conv-conv-add (× 8)                                (ResNet-18 BasicBlocks)
  ├─ upsample + concat + conv-conv (× 5)                (smp U-Net decoder)
  ├─ conv 3×3 + bias                                    (head)
  └─ softmax                                            -> probs.tif (2-band Float16)
```

## `gdal-conv2d` modes

| Mode | Args | Behavior |
|---|---|---|
| `conv` (default) | `--kernel --kernel-shape --bn-a --bn-b --bias --activation --stride --padding --depthwise` | multi-channel `nn.Conv2d` + optional bias/BN/activation/stride/padding/depthwise |
| `scale` | `--scale <float>` | elementwise multiply by scalar |
| `maxpool` | `--kernel-size --stride --padding` | bit-matches `nn.MaxPool2d(K, S, P)` |
| `upsample` | `--scale F --method nearest` | bit-matches `np.repeat` along H, W |
| `add` | repeated `--in`, optional `--activation` | elementwise sum across N rasters with optional fused activation |
| `concat` | repeated `--in` | band-wise concat |
| `softmax` | one `--in` | numerically-stable softmax across bands |

All modes write Float16 and preserve CRS+geotransform from the first input.
Inner conv loops parallelize over output channels via OpenMP.

## Repo layout

```
cpp/                                 the gdal-conv2d C++ binary
  src/gdal_conv2d.cpp
  CMakeLists.txt
  tests/test_modes.sh                per-mode parity vs numpy/PyTorch refs

export/                              gdal-unet-export Python tool
  gdal_unet_export/
    cli.py                           gdal-unet-export model.pt --arch ... -o weights/
    archs/resnet.py                  forward-graph walker
    sh_emit.py                       generates predict_<arch>.sh

samples/                             5 NAIP patches + GT masks (regression fixtures)
conda-recipe/                        conda-forge recipe for gdal-conv2d
.github/workflows/                   CI matrix + tag-based release

reference/                           historical, NOT shipped (see reference/README.md)
  predict_gdal.py                    original ~520 s pure-CLI baseline
  build_vrt.py                       131 s VRT-per-layer path
  profile_layers.py                  per-stage activation debugger
```

## End-to-end regression (sample 1717, chesapeakersc resnet18)

| Pipeline | Wall (16-CPU) | Subprocs | IoU vs GT |
|---|---|---|---|
| PyTorch reference (`smp.Unet`) | 0.08 s | — | 0.6374 |
| `reference/predict_gdal.py` (pure CLI) | 520 s | ~2185 | 0.6372 |
| `reference/build_vrt.py` (VRT per layer) | 131 s | 31 | 0.6372 |
| **`predict_resnet18.sh` (shipped)** | **5.5 s** | **~135** | **0.6372** |

All three GDAL-side pipelines match PyTorch bit-close (per-stage cosine
1.0000, max P(road) diff 2.5e-3 — entirely Float16 rounding).

## How we got to bit-close parity

The first naïve pass collapsed to IoU 0.21 vs PyTorch 0.64.
`reference/profile_layers.py` dumped per-stage activations and pinpointed two
boundary issues:

1. **PyTorch zero-pads; `gdal raster neighbors` edge-replicates.** Worst at the 7×7 stem; the wrong edge compounds via every residual skip. Fixed by zero-padding inside `gdal-conv2d` before each conv and cropping the boundary off after.
2. **Stride-2 alignment.** PyTorch samples at `(0, 0), (0, 2), …`; `gdal raster reproject -r nearest` samples at pixel centers (`(2i+1, 2j+1)`). Over 5 stride-2 stages = 32-pixel global shift at the bottleneck. Fixed by doing the strided sample at corner positions — now built into `gdal-conv2d` (and similarly for maxpool + nearest upsample).

After both fixes, the worst per-stage cosine to PyTorch jumped from 0.29 → 1.0000.

## Build

Requires GDAL ≥ 3.11 and OpenMP.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/gdal-conv2d --help
bash cpp/tests/test_modes.sh         # per-mode parity vs numpy/PyTorch
```

macOS: `brew install libomp` (or `conda install libomp`) first.
CMakeLists handles `@loader_path/../lib` / `$ORIGIN/../lib` RPATH portably.

## Supported platforms

| Platform | CI-built tarball | conda-forge build |
|---|---|---|
| Linux x86_64 | ✅ | ✅ |
| Linux arm64 | ✅ | ✅ (via osx/linux build) |
| macOS arm64 (Apple Silicon) | ✅ | ✅ |
| macOS x86_64 (Intel) | ❌ | ✅ |
| Windows | ❌ | ❌ |

macOS Intel and Windows are dropped from the GitHub Actions matrix (Intel-mac runner queue is multi-hour; Windows needs MSVC + libomp work). conda-forge does build osx-64 on its own infra so an Intel-mac conda install will still work once the conda-forge PR merges. Re-enabling Intel mac in the GH Actions matrix is one-line in `.github/workflows/build.yml`.

## License

MIT. See [`LICENSE`](LICENSE).
