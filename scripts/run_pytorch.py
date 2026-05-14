"""PyTorch reference inference: smp.Unet(resnet18) + chesapeakersc weights.

Writes:
    pytorch_probs.tif    Float16, 2 bands (P(bg), P(road)), georeferenced
    pytorch_class.tif    uint8, 1 band, argmax of softmax (0=bg, 1=road)

Run on CPU -- the crop is only 4096x4096 single-tile and resnet18 is light.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from huggingface_hub import hf_hub_download


HF_REPO = "isaaccorley/chesapeakersc"
HF_FILE = "unet-resnet18.pt"


def load_model() -> torch.nn.Module:
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    print(f"[pt] checkpoint {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=4,
        classes=2,
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("naip_md_4096.tif"))
    ap.add_argument("--probs", type=Path, default=Path("pytorch_probs.tif"))
    ap.add_argument("--cls", type=Path, default=Path("pytorch_class.tif"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[pt] input {args.input} missing", file=sys.stderr)
        return 2

    with rasterio.open(args.input) as src:
        x = src.read().astype(np.float32) / 255.0
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform

    print(f"[pt] input shape={x.shape} dtype={x.dtype}", flush=True)

    if x.shape[0] < 4:
        raise RuntimeError(f"need 4 bands (RGBN), got {x.shape[0]}")
    x = x[:4]

    model = load_model()
    inp = torch.from_numpy(x).unsqueeze(0)

    t0 = time.time()
    with torch.inference_mode():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
    dt = time.time() - t0
    print(f"[pt] forward {dt:.2f}s  out={tuple(probs.shape)}", flush=True)

    probs_np = probs.squeeze(0).numpy().astype(np.float16)
    cls_np = probs.squeeze(0).argmax(dim=0).numpy().astype(np.uint8)

    prof_p = profile.copy()
    prof_p.update(driver="GTiff", count=2, dtype="float16",
                  compress="deflate", tiled=True,
                  blockxsize=512, blockysize=512,
                  crs=crs, transform=transform)
    with rasterio.open(args.probs, "w", **prof_p) as dst:
        dst.write(probs_np)
        dst.set_band_description(1, "P(background)")
        dst.set_band_description(2, "P(road)")

    prof_c = profile.copy()
    prof_c.update(driver="GTiff", count=1, dtype="uint8",
                  compress="deflate", tiled=True,
                  blockxsize=512, blockysize=512,
                  crs=crs, transform=transform, nodata=None)
    with rasterio.open(args.cls, "w", **prof_c) as dst:
        dst.write(cls_np[None])
        dst.set_band_description(1, "argmax(0=bg,1=road)")

    print(f"[pt] wrote {args.probs} and {args.cls}", flush=True)
    print(f"[pt] road_pixels = {int((cls_np == 1).sum())} "
          f"({(cls_np == 1).mean()*100:.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
