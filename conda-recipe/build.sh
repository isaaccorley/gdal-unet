#!/usr/bin/env bash
set -euo pipefail

mkdir -p cpp/build
cd cpp/build
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build . -j "${CPU_COUNT}"
cmake --install .

# Install the Python orchestrator alongside the binary so users have a one-step
# "conda install gdal-unet && python -m predict_cpp …" path.
install -m 0644 ../predict/predict_cpp.py "$PREFIX/bin/predict_cpp.py"
