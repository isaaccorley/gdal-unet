# gdal-unet

Run a pretrained U-Net inference pipeline through one small C++ binary
(`gdal-conv2d`) plus a generated shell script. **No Python at deploy
time** — just bash and `gdal-conv2d` on PATH.

Reference model:
[`isaaccorley/chesapeakersc`](https://huggingface.co/isaaccorley/chesapeakersc)
— `smp.Unet(resnet18, in_channels=4, classes=2)`, 14.3 M params, trained on
NAIP RGBN for road segmentation. The same architecture pattern (and
`gdal-unet`) works for any conv-based U-Net.

## Two artifacts, two audiences

| Artifact | For | Install |
|---|---|---|
| `gdal-conv2d` (C++ binary) | end users running inference | tarball from the [latest release](https://github.com/isaaccorley/gdal-unet/releases/latest); `conda install -c conda-forge gdal-conv2d` is pending ([staged-recipes#33314](https://github.com/conda-forge/staged-recipes/pull/33314)) |
| `gdal-unet` (Python) | model authors converting checkpoints | `pip install git+https://github.com/isaaccorley/gdal-unet.git#subdirectory=export` |

## Inference flow

```bash
# One-time conversion (requires PyTorch):
gdal-unet export model.pt --arch resnet18 -o weights/

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

## End-to-end regression (sample 1717, chesapeakersc resnet18)

| Pipeline | Wall (16-CPU) | Subprocs | IoU vs GT |
|---|---|---|---|
| PyTorch reference (`smp.Unet`) | 0.08 s | — | 0.6374 |
| `reference/predict_gdal.py` (pure CLI) | 520 s | ~2185 | 0.6372 |
| `reference/build_vrt.py` (VRT per layer) | 131 s | 31 | 0.6372 |
| **`predict_resnet18.sh` (shipped)** | **5.5 s** | **~135** | **0.6372** |

All three GDAL-side pipelines match PyTorch bit-close (per-stage cosine
1.0000, max P(road) diff 2.5e-3 — entirely Float16 rounding).

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

## Demo (web viewer)

The `web/` directory has a MapLibre-based viewer that overlays the road
classification (and optionally every U-Net intermediate stage) on the NAIP
basemap. One driver script runs the full pipeline end to end, idempotently —
each step skips work whose outputs already exist.

```bash
# basic: NAIP + classification only
bash scripts/run_demo.sh

# full: also renders + tiles all 53 intermediate U-Net stages (~15–20 min)
bash scripts/run_demo.sh --with-intermediates

# or just the intermediate viewer (assumes run_demo.sh already ran):
bash scripts/build_intermediate_viewer.sh
```

What each pipeline step does:

| Step | Script | Output |
|---|---|---|
| 1. Download NAIP crop | `scripts/download_naip.py` | `naip_md_4096.tif` |
| 2. PyTorch reference inference | `scripts/run_pytorch.py` | `pytorch_{probs,class}.tif` |
| 3. Export weights | `gdal-unet export` | `weights/` |
| 4. `gdal-conv2d` inference (all 53 stages) | `weights/predict_resnet18.sh` | `intermediate_output/*.tif` |
| 5. Argmax → class raster | `scripts/argmax_class.py` | `gdal_class.tif` |
| 6. Compare PyTorch vs gdal | `scripts/compare.py` | stdout report |
| 7. Colormap class raster | `gdaldem color-relief` | `classification_rgba.tif` |
| 8. Tile NAIP RGB base | `gdal raster tile` | `web/tiles/naip/` |
| 9. Tile classification overlay | `gdal raster tile` | `web/tiles/classification/` |
| 10. Render intermediates (PCA-RGB + top-8 channel heatmaps per stage) | `scripts/render_intermediate.py` | `intermediate_visuals/` |
| 11. Tile intermediates (parallel xargs) | `scripts/tile_intermediate.sh` | `web/tiles/intermediate/` |
| 12. Build layer manifest | `scripts/build_layers_manifest.py` | `web/layers.json` |

To serve:

```bash
python -m http.server 8000   # from repo root
# open http://localhost:8000/web/
```

See [`web/README.md`](web/README.md) for the viewer UI, per-channel scrubber,
keyboard navigation, and how to add custom overlays.

## Supported platforms

| Platform | CI-built tarball | conda-forge build |
|---|---|---|
| Linux x86_64 | ✅ | ✅ |
| Linux arm64 | ✅ | ✅ (via osx/linux build) |
| macOS arm64 (Apple Silicon) | ✅ | ✅ |
| macOS x86_64 (Intel) | ❌ | ✅ |
| Windows | ❌ | ❌ |

macOS Intel and Windows are dropped from the GitHub Actions matrix (Intel-mac runner queue is multi-hour; Windows needs MSVC + libomp work). conda-forge does build osx-64 on its own infra so an Intel-mac conda install will still work once the conda-forge PR merges. Re-enabling Intel mac in the GH Actions matrix is one-line in `.github/workflows/build.yml`.
