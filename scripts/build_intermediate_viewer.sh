#!/usr/bin/env bash
# End-to-end orchestrator for the intermediate-layer viewer:
#   1. render_intermediate.py  (intermediate_output -> intermediate_visuals)
#   2. tile_intermediate.sh    (intermediate_visuals -> web/tiles/intermediate)
#   3. build_layers_manifest.py (-> web/layers.json)
# Idempotent (each step skips work whose outputs already exist).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"

echo "==[1/3] render intermediate visualizations=="
$PY scripts/render_intermediate.py

echo "==[2/3] tile intermediate visualizations=="
bash scripts/tile_intermediate.sh

echo "==[3/3] build web/layers.json manifest=="
$PY scripts/build_layers_manifest.py

echo
echo "[done] open web/index.html (e.g. python -m http.server 8000 from repo root)"
