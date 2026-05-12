# gdal-nn

Run neural networks entirely with the `gdal` CLI.

The actual math runs inside `gdal raster neighbors | calc | reproject | stack`
subprocesses. Python is only used to orchestrate (load inert weights from
`.npz`, generate the right `--kernel "[[…]]"` and `--calc "…"` strings, fan
out subprocesses).

## What's here

| File | Purpose |
|---|---|
| `export_weights.py` | Pull weights from a PyTorch `.pt` checkpoint, fold BN into per-channel affine, save as `model_weights.npz` (Float16) |
| `predict_gdal.py`   | Orchestrator: load `model_weights.npz`, drive `gdal raster …` subprocesses to run the forward pass on a 4-band uint8 NAIP RGBN GeoTIFF |
| `run_predict.slurm` | SLURM job script for a CPU node |
| `evaluate.py`       | IoU / F1 vs ground-truth masks |
| `visualize.py`      | RGB / GT / P(road) / predicted-mask side-by-side PNG |

## Model

Reference model: [`isaaccorley/chesapeakersc`](https://huggingface.co/isaaccorley/chesapeakersc)
on HuggingFace — a `segmentation_models_pytorch.Unet(resnet18,
in_channels=4, classes=2)` road segmenter trained on NAIP RGBN over the
Chesapeake Bay watershed. 14.3 M parameters. Input: 4-band uint8 NAIP.
Output: 2-class softmax (background, road).

Transforms: `T.Normalize(mean=[0], std=[255])` — just divide by 255.

## Pipeline

```
input NAIP (4-band uint8)
  ├─ preprocess  : calc x/255 → 4-band Float16
  ├─ stem        : 7×7 conv stride 2 → BN → ReLU
  ├─ maxpool 3×3 stride 2
  ├─ layer1 .. layer4 : 8 ResNet-18 BasicBlocks
  │                     (conv → BN → ReLU → conv → BN → +skip → ReLU)
  ├─ decoder.0 .. decoder.4 : smp U-Net decoder blocks
  │                           (upsample → concat skip → conv → BN → ReLU)*2
  ├─ head        : 3×3 conv → 2-class logits
  └─ softmax     : exp(L_i - max) / Σ exp(L_j - max) → probs.tif
```

## Engineering tricks

1. **Multi-channel conv via chunked diagonal trick.** `gdal raster neighbors`
   with K kernels on N-band input emits N·K bands (full cross-product).
   For a conv layer with Cout output channels, we group Cin into chunks of
   K and pass K·Cout kernels per chunk. The "diagonal" bands at indices
   `ic_in_chunk * (K·Cout + 1) + oc·K + 1` are picked by a follow-up `calc`
   and summed.

2. **Parallel chunks.** All chunks within a layer are independent: each runs
   `select+neighbors` in a `gdal raster pipeline` call, then a `calc` to
   pick diagonals into a per-chunk partial sum. Partials reduce via a
   parallel pairwise binary tree (the final sum fuses BN + ReLU + bias).
   Driven by Python `concurrent.futures.ThreadPoolExecutor`.

3. **Float16 throughout.** Outputs and intermediates are written as
   `Float16` (muparser computes internally in Float64 but casts on write).
   Halves disk I/O vs Float32. No measurable accuracy loss for inference.

4. **Minimal TIFF metadata.** Intermediates use `PROFILE=BASELINE`
   (no GeoTIFF keys), `INTERLEAVE=BAND`, `BIGTIFF=IF_SAFER`. Saves
   per-file overhead at scale.

## Usage

```bash
# export weights once from the .pt checkpoint
python export_weights.py

# run inference on a NAIP patch
python predict_gdal.py samples/1717_image.tif samples/1717_gdal_probs.tif

# or submit to SLURM
sbatch run_predict.slurm
```

## Two implementations

| | `predict_gdal.py` (CLI-only) | `cpp/predict_cpp.py` (C++ plugin) |
|---|---|---|
| **Math runtime** | `gdal raster ...` subprocesses | `gdalnn_conv` (custom C++ binary) + a few `gdal raster ...` |
| **Subprocesses / forward pass** | ~2185 | **31** |
| **Wall time (16-CPU SLURM)** | ~520 s | **~7.5 s** |
| **Peak RSS** | ~6 GB intermediate tifs on disk | **156 MB** |
| **Per-stage cosine vs PyTorch** | 1.0000 | 1.0000 |
| **IoU vs GT** | 0.6372 | 0.6372 |

**The C++ path is ~70× faster** with identical numerical output. Source: [`cpp/src/gdalnn_conv.cpp`](cpp/src/gdalnn_conv.cpp) (330 LOC, OpenMP over output channels, scalar inner loops, GDAL Float16 I/O).

## Results so far

End-to-end forward pass on a 512×512 NAIP RGBN patch (`samples/1717_image.tif`):

| Metric | Value |
|---|---|
| Wall time (16-cpu node, 8-way ThreadPool) | **513 s** (~8.5 min) |
| Subprocesses | 2198 |
| Per-subprocess avg | 1.47 s |
| Cumulative gdal time | 3234 s |
| Parallel speedup | **6.3×** |
| Output | 2-band Float16 probability raster, sums to 1.0 |

### vs PyTorch reference

Same checkpoint loaded into `smp.Unet(resnet18, in=4, classes=2)`:

| Metric | PyTorch | GDAL CLI |
|---|---|---|
| road pixels (@0.5) | 10.83 % | 10.82 % |
| IoU vs GT | 0.6374 | **0.6372** |
| argmax agreement | — | **99.99 %** |
| max P(road) diff | — | 2.9e-3 |
| mean P(road) diff | — | 1.1e-4 |

**The pipeline reproduces PyTorch bit-close.** The remaining ~10
disagreeing pixels out of 262144 are entirely from Float16 quantization
at exactly-0.5 probabilities.

### How we got there: profile-then-debug

A first naïve pass collapsed to **IoU 0.21** (vs PyTorch 0.64) even
though all the math was structurally right. Running `profile_layers.py`
to dump stage-by-stage activations from PyTorch and from each
GDAL-intermediate tif uncovered the divergence by stage (cosine
similarity to PyTorch):

```
                     before fix     after fix
pre   (input/255)     1.0000         1.0000
stem  (7x7 s=2)       0.7979   →     1.0000
mp    (3x3 s=2)       0.8395   →     1.0000
l1                    0.6803   →     1.0000
l2                    0.3963   →     1.0000
l3                    0.2921   →     1.0000   ← worst before
l4                    0.4827   →     1.0000
d0..d4                0.51-0.73 →    1.0000
logits                0.8569   →     1.0000
```

Two root causes, both at the boundary between `gdal raster ...` and
PyTorch's tensor semantics:

1. **Boundary padding.** PyTorch `conv2d(padding=N)` zero-pads;
   `gdal raster neighbors` edge-replicates. For a 7×7 stem conv with
   padding 3, that's a 3-pixel boundary error that **compounds through
   the 5 residual stages** because the same wrong edge gets added back
   via every skip connection. Fix: explicitly zero-pad the raster
   (via `rasterio` + `np.pad`) before each `neighbors` call and crop the
   output back to the original spatial size.

2. **Stride-2 alignment.** PyTorch `conv2d(stride=2)` samples at
   `(0, 0), (0, 2), (0, 4), …`; `gdal raster reproject -r nearest
   --size W/2,H/2` maps each dest-pixel center to the nearest
   source-pixel center, which lands at `(2i+1, 2j+1)` — off by one.
   Over 5 stride-2 stages that's a 32-pixel global shift at the
   bottleneck. Fix: replace the reproject-based downsample with
   `data[:, ::2, ::2]` via numpy.

Same two fixes apply to `MaxPool2d(3, 2, 1)` (zero-pad + `::2`) and
to `Upsample(scale_factor=2, mode='nearest')` (`np.repeat(2)` twice).

All four geometric primitives now live in
[`predict_gdal.py`](predict_gdal.py) as small numpy helpers that read
the input tif with `rasterio`, transform spatially in memory, and write
out a fresh tif before handing back to GDAL CLI for the actual math.

## Why

Funny. Almost certainly impractical for real deployment compared to
PyTorch / ONNX. But it works.
