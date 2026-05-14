#!/usr/bin/env bash
# Tile every intermediate_visuals/*.tif into web/tiles/intermediate/<stage>_<view>/.
# Parallelizes 4-way; idempotent (skips dirs that already exist).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VISUALS="${VISUALS:-intermediate_visuals}"
OUT_BASE="${OUT_BASE:-web/tiles/intermediate}"
PAR="${PAR:-4}"
ZMIN="${ZMIN:-14}"
ZMAX="${ZMAX:-17}"

mkdir -p "$OUT_BASE"

if ! command -v gdal >/dev/null 2>&1; then
  echo "gdal command not found on PATH; activate gdal-unet env first" >&2
  exit 2
fi

mapfile -t TIFS < <(ls -1 "$VISUALS"/*.tif 2>/dev/null || true)
if [[ ${#TIFS[@]} -eq 0 ]]; then
  echo "no tifs in $VISUALS/; run scripts/render_intermediate.py first" >&2
  exit 2
fi

tile_one() {
  local tif="$1"
  local stem
  stem="$(basename "$tif" .tif)"
  local out="$OUT_BASE/$stem"
  if [[ -d "$out" ]]; then
    echo "[skip] $stem"
    return 0
  fi
  echo "[tile] $stem"
  gdal raster tile \
    --convention xyz --webviewer none \
    --min-zoom "$ZMIN" --max-zoom "$ZMAX" \
    -r bilinear \
    "$tif" "$out" >/dev/null 2>&1
}
export -f tile_one
export OUT_BASE ZMIN ZMAX

printf '%s\n' "${TIFS[@]}" | xargs -I{} -n1 -P"$PAR" bash -c 'tile_one "$@"' _ {}

echo "[done] $(find "$OUT_BASE" -mindepth 1 -maxdepth 1 -type d | wc -l) tile sets"
