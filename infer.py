"""Run chesapeakersc UNet-ResNet18 road segmentation on a NAIP GeoTIFF.

Usage: python infer.py <input_naip.tif> [<output_probs.tif>]

Input:   4-band uint8 NAIP RGBN (any size).
Output:  2-band Float32 raster (per-class softmax probabilities) +
         a 1-band uint8 classified mask written next to it (..._mask.tif).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.v2 as T

CKPT = Path(__file__).parent / "unet-resnet18.pt"


def load_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=4,
        classes=2,
    )
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.inference_mode()
def predict(model, naip_arr: np.ndarray) -> np.ndarray:
    """naip_arr: (4, H, W) uint8.  Returns (2, H, W) float32 probabilities."""
    transforms = torch.nn.Sequential(T.Normalize(mean=[0.0], std=[255.0]))
    x = torch.from_numpy(naip_arr).float().unsqueeze(0)
    x = transforms(x)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="path to 4-band NAIP RGBN GeoTIFF")
    p.add_argument("output", nargs="?", default=None,
                   help="output 2-band Float32 probability tif "
                        "(default: <input>_probs.tif)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else \
        in_path.with_name(in_path.stem + "_probs.tif")
    mask_path = out_path.with_name(out_path.stem + "_mask.tif")

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        naip = src.read()
        print(f"[input] {in_path.name}  {src.count}x{src.height}x{src.width}  "
              f"dtype={naip.dtype}  CRS={src.crs}  DN=[{naip.min()},{naip.max()}]")

    if naip.shape[0] < 4:
        sys.exit(f"need >=4 bands (RGBN); got {naip.shape[0]}")
    naip = naip[:4]  # drop any alpha
    # pad H,W to multiples of 32 (ResNet18 downsamples 5x)
    H, W = naip.shape[1:]
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        naip = np.pad(naip, ((0, 0), (0, pad_h), (0, pad_w)),
                      mode="reflect")
        print(f"[pad]   reflect-padded to {naip.shape[1:]} (multiple of 32)")

    print(f"[model] loading {CKPT.name}")
    model = load_model()
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} params")

    t0 = time.time()
    probs = predict(model, naip)
    print(f"[infer] done in {time.time()-t0:.2f}s")

    # crop back to original dims
    probs = probs[:, :H, :W]
    print(f"[infer] probs: shape={probs.shape}  road prob range "
          f"[{probs[1].min():.4f}, {probs[1].max():.4f}]  "
          f"mean={probs[1].mean():.4f}")

    # Write 2-band probability raster
    # drop block-size hints from input profile that may not be multiples of 16
    for k in ("blockxsize", "blockysize"):
        profile.pop(k, None)
    profile.update(count=2, dtype="float32", nodata=None,
                   compress="DEFLATE", tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(probs.astype(np.float32))
        dst.set_band_description(1, "p(background)")
        dst.set_band_description(2, "p(road)")
    print(f"[write] {out_path.name}")

    # Write a 1-band uint8 mask (road = 1)
    mask = (probs.argmax(0) == 1).astype(np.uint8)
    mp = profile.copy(); mp.update(count=1, dtype="uint8")
    with rasterio.open(mask_path, "w", **mp) as dst:
        dst.write(mask, 1)
    print(f"[write] {mask_path.name}  road pixels: "
          f"{mask.sum()}/{mask.size} ({100*mask.mean():.2f}%)")


if __name__ == "__main__":
    main()
