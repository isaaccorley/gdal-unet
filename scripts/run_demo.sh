#!/usr/bin/env bash
# End-to-end driver: downloads NAIP, runs both PyTorch and gdal-unet inference,
# compares them, builds the colormap + XYZ tiles, and is fully idempotent
# (skips steps whose outputs already exist).
#
# Run from the repo root.  Requires the `gdal-unet` conda env active, or
# pass GDAL_CONV2D / paths via env.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
GDAL_CONV2D="${GDAL_CONV2D:-$ROOT/cpp/build/gdal-conv2d}"
export GDAL_CONV2D

NAIP="naip_md_4096.tif"
WEIGHTS_DIR="weights"
INTER="intermediate_output"
PROBS="$INTER/probs.tif"
GDAL_CLASS="gdal_class.tif"
PYTORCH_PROBS="pytorch_probs.tif"
PYTORCH_CLASS="pytorch_class.tif"
RGBA="classification_rgba.tif"
NAIP_RGB_VRT="naip_rgb.vrt"

mkdir -p "$INTER" "$WEIGHTS_DIR" web/tiles

echo "==[1/9] download NAIP crop=="
$PY scripts/download_naip.py -o "$NAIP"

echo "==[2/9] PyTorch reference inference=="
if [[ -f "$PYTORCH_PROBS" && -f "$PYTORCH_CLASS" ]]; then
  echo "[skip] $PYTORCH_PROBS exists"
else
  $PY scripts/run_pytorch.py --input "$NAIP" --probs "$PYTORCH_PROBS" --cls "$PYTORCH_CLASS"
fi

echo "==[3/9] export weights via gdal-unet=="
if [[ -f "$WEIGHTS_DIR/predict_resnet18.sh" && -f "$WEIGHTS_DIR/shapes.txt" ]]; then
  echo "[skip] $WEIGHTS_DIR/predict_resnet18.sh exists"
else
  CKPT="$($PY -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("isaaccorley/chesapeakersc","unet-resnet18.pt"))')"
  gdal-unet export "$CKPT" --arch resnet18 -o "$WEIGHTS_DIR"
fi

echo "==[4/9] gdal-conv2d inference=="
if [[ -f "$PROBS" ]]; then
  echo "[skip] $PROBS exists"
else
  WORK="$ROOT/$INTER" bash "$WEIGHTS_DIR/predict_resnet18.sh" "$WEIGHTS_DIR" "$NAIP" "$PROBS"
fi

echo "==[5/9] argmax to class raster=="
if [[ -f "$GDAL_CLASS" ]]; then
  echo "[skip] $GDAL_CLASS exists"
else
  $PY scripts/argmax_class.py --probs "$PROBS" --out "$GDAL_CLASS" --ref "$NAIP"
fi

echo "==[6/9] compare PyTorch vs gdal-unet=="
$PY scripts/compare.py --pytorch "$PYTORCH_PROBS" --gdal "$PROBS"

echo "==[7/9] colormap classification=="
if [[ -f "$RGBA" ]]; then
  echo "[skip] $RGBA exists"
else
  gdaldem color-relief -alpha "$GDAL_CLASS" color.txt "$RGBA"
fi

echo "==[8/9] tile NAIP RGB=="
if [[ -d web/tiles/naip ]]; then
  echo "[skip] web/tiles/naip exists"
else
  rm -f "$NAIP_RGB_VRT"
  gdal_translate -q -b 1 -b 2 -b 3 -of VRT "$NAIP" "$NAIP_RGB_VRT"
  mkdir -p web/tiles
  gdal raster tile --convention xyz --webviewer none --min-zoom 14 --max-zoom 19 -r bilinear "$NAIP_RGB_VRT" web/tiles/naip
fi

echo "==[9/9] tile classification RGBA=="
if [[ -d web/tiles/classification ]]; then
  echo "[skip] web/tiles/classification exists"
else
  mkdir -p web/tiles
  gdal raster tile --convention xyz --webviewer none --min-zoom 14 --max-zoom 19 -r nearest "$RGBA" web/tiles/classification
fi

echo "==[bonus] write web/bounds.json=="
$PY - <<EOF
import json, rasterio
from rasterio.warp import transform_bounds
with rasterio.open("$NAIP") as src:
    minx, miny, maxx, maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
import pathlib
pathlib.Path("web/bounds.json").write_text(json.dumps(
    {"minLon": minx, "minLat": miny, "maxLon": maxx, "maxLat": maxy}, indent=2))
print("wrote web/bounds.json")
EOF

echo
echo "[done] open web/index.html (e.g. python -m http.server 8000)"

# Optional: build the intermediate-layer viewer (tiles + manifest).
if [[ "${WITH_INTERMEDIATES:-0}" == "1" || "${1:-}" == "--with-intermediates" ]]; then
  echo
  echo "==[bonus] build intermediate-layer viewer=="
  bash scripts/build_intermediate_viewer.sh
fi
