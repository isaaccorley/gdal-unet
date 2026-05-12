#!/usr/bin/env bash
set -euo pipefail

# Build + install the C++ binary as $PREFIX/bin/gdal-unet-conv
mkdir -p cpp/build
cd cpp/build
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build . -j "${CPU_COUNT}"
cmake --install .
cd "$SRC_DIR"

# Install the Python package; provides the `gdal-unet` console script via
# the entry-point declared in pyproject.toml.
"$PYTHON" -m pip install . --no-deps --no-build-isolation -vv
