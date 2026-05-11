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

## Why

Funny. Almost certainly impractical for real deployment compared to
PyTorch / ONNX. But it works.
