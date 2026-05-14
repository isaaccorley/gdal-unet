"""Check whether a prediction raster is spatially aligned with a reference
(ground truth) raster — reports the geotransform-implied offset, the
data-level sub-pixel offset (via phase cross correlation), and their sum.

The "effective shift" is what the user actually sees on a map: if it's
non-zero, the prediction is mis-placed. A non-zero geo offset with a
matching opposite data offset means the gt is compensating for a
strided-op shift correctly — net zero on the map.

Usage:
    python tools/check_alignment.py PRED.tif GT.tif [--band 1] [--upsample 100]

Requires: rasterio, numpy, scikit-image.
"""
import argparse
import sys

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from skimage.registration import phase_cross_correlation


def read_band(path, band):
    with rasterio.open(path) as ds:
        return ds.read(band).astype(np.float32), ds.transform, ds.crs, ds.shape


def resample_to(src, src_tr, src_crs, dst_shape, dst_tr, dst_crs):
    out = np.zeros(dst_shape, dtype=np.float32)
    reproject(src, out,
              src_transform=src_tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs=dst_crs,
              resampling=Resampling.bilinear)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="prediction raster (e.g., UNet output)")
    ap.add_argument("gt",   help="ground-truth / reference raster")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--upsample", type=int, default=100,
                    help="sub-pixel precision for phase_cross_correlation")
    ap.add_argument("--mask-zeros", action="store_true",
                    help="ignore zero pixels (treat as nodata) in correlation")
    args = ap.parse_args()

    pred, pt, pc, ps = read_band(args.pred, args.band)
    gt,   gt_tr, gc, gs = read_band(args.gt, args.band)

    print(f"pred: shape={ps}  transform={pt}  crs={pc}")
    print(f"gt  : shape={gs}  transform={gt_tr}  crs={gc}")

    # Resample pred onto gt's grid if they differ.
    if ps != gs or pt != gt_tr or pc != gc:
        print("\nresampling pred -> gt grid (bilinear) for fair index-space comparison")
        pred_r = resample_to(pred, pt, pc, gs, gt_tr, gc)
    else:
        pred_r = pred

    # Mask non-finite + (optionally) zeros — phase_cross_correlation is sensitive.
    ref = np.where(np.isfinite(gt), gt, 0.0)
    mov = np.where(np.isfinite(pred_r), pred_r, 0.0)
    if args.mask_zeros:
        m = (ref != 0) & (mov != 0)
        ref = ref * m
        mov = mov * m

    # Geo offset implied purely by the gts (pred's origin vs gt's origin),
    # expressed in gt pixel units. Skips if we already resampled to gt grid.
    a, _, c, _, e, f = gt_tr.a, gt_tr.b, gt_tr.c, gt_tr.d, gt_tr.e, gt_tr.f
    pa, _, pc_, _, pe, pf = pt.a, pt.b, pt.c, pt.d, pt.e, pt.f
    geo_dx_px = (pc_ - c) / a              # +ve = pred origin east of gt origin
    geo_dy_px = (pf - f) / e               # +ve = pred origin south of gt (gt_tr.e<0)

    # Sub-pixel data shift, in gt pixel index space:
    # shift[0] = rows (down +), shift[1] = cols (right +). Sign convention:
    # phase_cross_correlation(reference, moving) returns the shift needed to
    # move `moving` onto `reference`, so positive shift = moving must move
    # down/right to align with reference.
    shift, err, _ = phase_cross_correlation(ref, mov,
                                            upsample_factor=args.upsample,
                                            normalization=None)
    data_dy_px, data_dx_px = float(shift[0]), float(shift[1])

    # Effective on-map shift (what the user sees): if pred's gt already
    # encodes an offset, the data needs to physically shift by the negative
    # of that to *land at the right map location*. Effective = data shift
    # the user must apply, given current gts, to align the predictions.
    eff_dx = data_dx_px - geo_dx_px
    eff_dy = data_dy_px - geo_dy_px

    print()
    print(f"geo offset      (pred_origin - gt_origin, in gt-pixel units):  "
          f"dx={geo_dx_px:+.4f}  dy={geo_dy_px:+.4f}")
    print(f"data offset     (phase cross correlation, gt-pixel units):    "
          f"dx={data_dx_px:+.4f}  dy={data_dy_px:+.4f}   err={err:.4g}")
    print(f"effective shift (what shows on the map, gt-pixel units):      "
          f"dx={eff_dx:+.4f}  dy={eff_dy:+.4f}")
    print()
    px = abs(gt_tr.a)
    print(f"in ground units (pixel size {px:g}): "
          f"dx={eff_dx*px:+.4g}  dy={eff_dy*px:+.4g}")

    # Heuristic verdict
    mag = max(abs(eff_dx), abs(eff_dy))
    if mag < 0.1:
        print("\nVERDICT: aligned (sub 0.1 px residual).")
    elif mag < 0.6:
        print("\nVERDICT: sub-pixel offset (<0.6 px). Likely inherent to "
              "strided/nearest ops; not a geotransform bug.")
    else:
        print("\nVERDICT: misaligned by >0.6 px. Likely a real bug — "
              "check (a) that the prediction was saved with the conv output "
              "geotransform, not the input's; (b) that no resampling step "
              "between gdal-conv2d and the final tif lost the transform.")


if __name__ == "__main__":
    main()
