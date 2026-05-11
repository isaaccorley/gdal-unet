"""Compare predicted road masks against ground truth for each sample."""

from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).parent / "samples"

def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else float("nan")

def f1(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r

print(f"{'sample':>8s} {'gt_road%':>8s} {'pred_road%':>10s}  {'IoU':>6s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}")
print("-" * 60)
ious = []
for img in sorted(ROOT.glob("*_image.tif")):
    name = img.stem.replace("_image", "")
    with rasterio.open(ROOT / f"{name}_mask.tif") as src:
        gt = src.read(1)
    with rasterio.open(ROOT / f"{name}_probs_mask.tif") as src:
        pr = src.read(1)
    # Map to boolean road / not-road.  GT mask in ChesapeakeRSC uses class
    # indices; road is typically class 4 (impervious surface / road).
    # Inspect uniques first to confirm.
    # ChesapeakeRSC binary task: positive = Road (class 9) +
    # Tree Canopy over Road (class 12).
    gt_road = (gt == 9) | (gt == 12)
    pr_road = pr.astype(bool)
    f, p, r = f1(pr_road, gt_road)
    i = iou(pr_road, gt_road)
    ious.append(i)
    print(f"{name:>8s} {100*gt_road.mean():7.2f}% {100*pr_road.mean():9.2f}%  "
          f"{i:6.3f}  {f:6.3f}  {p:6.3f}  {r:6.3f}")
print("-" * 60)
print(f"{'mean IoU':>30s}  {np.mean(ious):.3f}")

# debug: peek at first mask unique values
with rasterio.open(ROOT / "1717_mask.tif") as src:
    gt = src.read(1)
print(f"\nGT mask 1717 unique values: {sorted(np.unique(gt).tolist())}")
