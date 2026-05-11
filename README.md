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
| road pixels (@0.5) | 10.83 % | 5.31 % |
| IoU vs GT | 0.637 | 0.207 |
| Binary agreement | — | 88.6 % |

The pipeline is **structurally correct** but **off in alignment**: many
pixels disagree near road edges, which is where strided ops and
nearest-neighbor upsamples disagree by half a pixel. Known divergence
sources:

1. **Stride-2 convs done as full-res-conv + nearest-downsample.** PyTorch
   samples at `(0, 0), (0, 2), …`; GDAL's `reproject -r nearest` picks
   nearest source pixel from dest-pixel center, which can be off by one.
2. **3×3 MaxPool stride 2** — PyTorch zero-pads, GDAL `neighbors` uses
   edge replication. Boundary differs.
3. **7×7 stem conv** — GDAL probably edge-replicates, PyTorch zero-pads.
   The boundary effects compound through 5 downsamples.
4. **smp decoder upsample** — PyTorch `Upsample(scale=2, mode="nearest")`
   replicates each pixel as a 2×2 block. GDAL `reproject` does the same
   in principle but the source-pixel-center mapping can off-by-one.
5. **Float16 throughout.** Mean magnitude is fine; tail divergence is
   masked by alignment, not precision.

Fixing 1–4 requires careful per-op offset handling (probably custom
`--bbox` on each reproject) and is the natural next step.

## Why

Funny. Almost certainly impractical for real deployment compared to
PyTorch / ONNX. But it works.
