"""Argmax a 2-band P(bg)/P(road) GeoTIFF into a 1-band uint8 class raster.

Optionally re-stamps the CRS + geotransform from a reference raster -- the
gdal-unet pipeline's per-stage geotransform tracks the padded receptive-field
of each conv layer, which is intentional but doesn't pixel-align with the
original input.  For visualization we want output pixel (i,j) to map to the
same physical location as input NAIP pixel (i,j), so we re-stamp from the
original NAIP.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None,
                    help="Reference raster: copy CRS+transform from here.")
    args = ap.parse_args()

    with rasterio.open(args.probs) as src:
        probs = src.read().astype(np.float32)
        prof = src.profile.copy()
        crs = src.crs
        transform = src.transform

    if args.ref is not None:
        with rasterio.open(args.ref) as ref:
            crs = ref.crs
            transform = ref.transform

    cls = probs.argmax(axis=0).astype(np.uint8)
    prof.update(driver="GTiff", count=1, dtype="uint8",
                compress="deflate", tiled=True,
                blockxsize=512, blockysize=512,
                crs=crs, transform=transform, nodata=None)
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(cls[None])
        dst.set_band_description(1, "argmax(0=bg,1=road)")
    print(f"[argmax] wrote {args.out}  road_px={int((cls==1).sum())} "
          f"({(cls==1).mean()*100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
