"""Argmax a 2-band P(bg)/P(road) GeoTIFF into a 1-band uint8 class raster."""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with rasterio.open(args.probs) as src:
        probs = src.read().astype(np.float32)
        prof = src.profile.copy()

    cls = probs.argmax(axis=0).astype(np.uint8)
    prof.update(driver="GTiff", count=1, dtype="uint8",
                compress="deflate", tiled=True,
                blockxsize=512, blockysize=512,
                nodata=None)
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(cls[None])
        dst.set_band_description(1, "argmax(0=bg,1=road)")
    print(f"[argmax] wrote {args.out}  road_px={int((cls==1).sum())} "
          f"({(cls==1).mean()*100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
