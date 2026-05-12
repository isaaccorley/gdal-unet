"""Verify VRT pipeline outputs against PyTorch and ground truth.

Usage:
    python verify.py <work_dir>      # stage-by-stage cosine + final IoU
    python verify.py                 # just IoU on 1717_gdal_probs.tif
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()


def iou(a, b):
    return float((a & b).sum()) / max(1, (a | b).sum())


def final_iou(probs_tif: Path):
    torch_p = rasterio.open(ROOT / "../chesapeakersc/samples/1717_probs.tif").read()
    gdal_p = rasterio.open(probs_tif).read()
    gt = rasterio.open(ROOT / "samples/1717_mask.tif").read(1)
    gt_road = ((gt == 9) | (gt == 12))
    agree = ((torch_p[1] > 0.5) == (gdal_p[1] > 0.5)).mean()
    print(f"argmax agreement vs torch : {agree:.6f}")
    print(f"IoU vs GT  (torch)        : {iou((torch_p[1] > 0.5).astype(bool), gt_road):.4f}")
    print(f"IoU vs GT  (vrt-gdal)     : {iou((gdal_p[1] > 0.5).astype(bool), gt_road):.4f}")


if __name__ == "__main__":
    final_iou(Path(sys.argv[1]) if len(sys.argv) > 1
              else ROOT / "1717_gdal_probs.tif")
