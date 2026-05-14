"""Compare PyTorch and gdal-unet softmax outputs on the same NAIP crop."""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def read_probs(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch", type=Path, default=Path("pytorch_probs.tif"))
    ap.add_argument("--gdal", type=Path,
                    default=Path("intermediate_output/probs.tif"))
    # Float16 round-trips through .bin + .tif accumulate ~1e-2 worst-case
    # error per pixel (entirely rounding); whole-image argmax is bit-close.
    ap.add_argument("--max-diff-thresh", type=float, default=1.5e-2)
    ap.add_argument("--argmax-agree-thresh", type=float, default=0.999)
    args = ap.parse_args()

    pt = read_probs(args.pytorch)
    gd = read_probs(args.gdal)
    if pt.shape != gd.shape:
        print(f"[cmp] shape mismatch: pt={pt.shape} gd={gd.shape}",
              file=sys.stderr)
        return 2

    pt_road = pt[1]
    gd_road = gd[1]
    diff = np.abs(pt_road - gd_road)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    pt_cls = pt.argmax(axis=0)
    gd_cls = gd.argmax(axis=0)
    agree = float((pt_cls == gd_cls).mean())

    pt_road_px = int((pt_cls == 1).sum())
    gd_road_px = int((gd_cls == 1).sum())
    inter = int(((pt_cls == 1) & (gd_cls == 1)).sum())
    union = int(((pt_cls == 1) | (gd_cls == 1)).sum())
    iou = inter / union if union else float("nan")

    print(f"[cmp] P(road) max abs diff = {max_diff:.4g}")
    print(f"[cmp] P(road) mean abs diff = {mean_diff:.4g}")
    print(f"[cmp] argmax agreement     = {agree*100:.4f}%")
    print(f"[cmp] road px pt={pt_road_px} gd={gd_road_px} IoU={iou:.4f}")

    ok = (max_diff <= args.max_diff_thresh) and (agree >= args.argmax_agree_thresh)
    if not ok:
        print(f"[cmp] FAIL: thresholds max_diff<={args.max_diff_thresh}, "
              f"agree>={args.argmax_agree_thresh}", file=sys.stderr)
        return 1
    print("[cmp] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
