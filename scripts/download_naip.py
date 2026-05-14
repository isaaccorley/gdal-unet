"""Download a 4096x4096 NAIP crop over a Maryland AOI via Planetary Computer.

The hardcoded AOI sits over the Roland Park / Lake Roland / Mt. Washington
area in north Baltimore -- this is exactly where the chesapeakersc training
samples (e.g. samples/1717_image.tif) come from, so the model is fully in
distribution.

NAIP pixel resolution matters: chesapeakersc was trained on 1 m NAIP
(samples are 1 m/pixel). Recent MD NAIP on Planetary Computer is 30 cm,
which puts the model far out of distribution and yields ~0% road detections.
We filter STAC for 1 m items (NAIP gsd in {1.0, 0.6, 0.3}; the resolution is
encoded in the asset filename token "_1_", "_060_", or "_030_") and pick the
most recent.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window


# Hardcoded Maryland AOI: small lon/lat bbox over Roland Park / Lake Roland /
# Mt. Washington -- residential Baltimore suburbs containing the area used
# to draw chesapeakersc training samples (e.g. samples/1717_image.tif sits
# inside this bbox at lon -76.615, lat 39.358).
AOI_LON_MIN, AOI_LON_MAX = -76.62, -76.59
AOI_LAT_MIN, AOI_LAT_MAX = 39.34, 39.37

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"
CROP_SIZE = 4096
TARGET_GSD_M = 1.0  # match chesapeakersc training resolution


def _gsd_from_href(href: str) -> float | None:
    """NAIP filename encodes resolution as e.g. '_1_' (1m), '_060_' (0.6m),
    '_030_' (0.3m).  Returns gsd in metres or None if not parseable."""
    fn = href.split("/")[-1]
    for tok, gsd in (("_1_", 1.0), ("_060_", 0.6), ("_030_", 0.3)):
        if tok in fn:
            return gsd
    return None


def search_naip_item(target_gsd: float):
    centroid = ((AOI_LON_MIN + AOI_LON_MAX) / 2, (AOI_LAT_MIN + AOI_LAT_MAX) / 2)
    bbox = [AOI_LON_MIN, AOI_LAT_MIN, AOI_LON_MAX, AOI_LAT_MAX]
    print(f"[naip] STAC search bbox={bbox} target_gsd={target_gsd}m", flush=True)
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(catalog.search(collections=[COLLECTION], bbox=bbox).items())

    matches = [it for it in items
               if _gsd_from_href(it.assets["image"].href) == target_gsd]
    if not matches:
        avail = sorted({_gsd_from_href(it.assets["image"].href) for it in items})
        raise RuntimeError(
            f"No NAIP items at {target_gsd} m; available gsds: {avail}")
    matches.sort(key=lambda it: it.properties.get("datetime", ""), reverse=True)
    item = matches[0]
    print(f"[naip] picked {item.id} datetime={item.properties.get('datetime')}",
          flush=True)
    return item, centroid


def crop_naip(item, centroid, out_path: Path, size: int):
    asset = item.assets["image"]
    href = asset.href
    print(f"[naip] streaming from {href[:120]}...", flush=True)

    with rasterio.open(href) as src:
        gsd = abs(src.transform.a)
        print(f"[naip] src crs={src.crs} size={src.width}x{src.height} "
              f"bands={src.count} dtype={src.dtypes[0]} gsd={gsd:.2f}m",
              flush=True)
        xs, ys = warp_transform("EPSG:4326", src.crs, [centroid[0]], [centroid[1]])
        cx, cy = xs[0], ys[0]
        row, col = rowcol(src.transform, cx, cy)
        row, col = int(row), int(col)
        col0 = max(0, min(src.width - size, col - size // 2))
        row0 = max(0, min(src.height - size, row - size // 2))
        window = Window(col0, row0, size, size)
        print(f"[naip] window col0={col0} row0={row0} size={size}", flush=True)
        data = src.read(window=window)
        win_transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=size,
            width=size,
            count=src.count,
            dtype=data.dtype,
            transform=win_transform,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

    if data.shape[0] < 4:
        raise RuntimeError(f"Expected >=4 bands (RGBN), got {data.shape[0]}")
    if data.shape[0] > 4:
        data = data[:4]
        profile["count"] = 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(("red", "green", "blue", "nir"), start=1):
            dst.set_band_description(i, name)
    print(f"[naip] wrote {out_path}  shape={data.shape}  "
          f"min={data.min()} max={data.max()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("naip_md_4096.tif"))
    ap.add_argument("--size", type=int, default=CROP_SIZE)
    ap.add_argument("--gsd", type=float, default=TARGET_GSD_M,
                    help="Target ground sample distance in metres "
                         "(must match what the model was trained on).")
    args = ap.parse_args()

    if args.output.exists():
        print(f"[naip] {args.output} already exists; skipping download. "
              f"Delete it to re-fetch.", flush=True)
        return 0

    item, centroid = search_naip_item(args.gsd)
    crop_naip(item, centroid, args.output, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
