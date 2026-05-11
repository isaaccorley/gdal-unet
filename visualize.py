"""Render RGB / GT / prediction side-by-side panels for each sample."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).parent / "samples"

names = sorted(p.stem.replace("_image", "") for p in ROOT.glob("*_image.tif"))
fig, axes = plt.subplots(len(names), 4, figsize=(14, 3.2 * len(names)))

for row, name in enumerate(names):
    with rasterio.open(ROOT / f"{name}_image.tif") as src:
        img = src.read()  # (4, H, W) uint8
    with rasterio.open(ROOT / f"{name}_mask.tif") as src:
        gt = src.read(1)
    with rasterio.open(ROOT / f"{name}_probs.tif") as src:
        probs = src.read()  # (2, H, W)

    rgb = np.transpose(img[:3], (1, 2, 0))  # (H, W, 3)
    gt_road = ((gt == 9) | (gt == 12)).astype(np.uint8)
    road_prob = probs[1]
    pred = (road_prob > 0.5).astype(np.uint8)

    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(f"NAIP RGB ({name})")
    axes[row, 1].imshow(gt_road, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title(f"GT road  ({100*gt_road.mean():.1f}%)")
    axes[row, 2].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
    axes[row, 2].set_title("P(road)")
    axes[row, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[row, 3].set_title(f"Pred road  ({100*pred.mean():.1f}%)")
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
out = Path(__file__).parent / "results.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"wrote {out.name}")
