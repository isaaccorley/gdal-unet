#!/bin/bash
# Source this to get a working environment for gdal-unet on tgi-rails.
#  - GDAL CLIs (gdal raster ...) from ftw-tile env
#  - PyTorch + smp + rasterio from torchgeo env (compatible runtime)
#  - gcc 12.3 for compiling cpp/gdalnn_conv

export GDAL_ENV=/projects/bgtj/isaaccorley/envs/ftw-tile
export TORCH_ENV=/projects/bgtj/isaaccorley/envs/torchgeo
export GCC_ROOT=/sw/spack/v1/apps/gcc/12.3.0-gcc-8.5.0-mzq2s4z

export PATH=$GCC_ROOT/bin:$GDAL_ENV/bin:$PATH
export PROJ_DATA=$GDAL_ENV/share/proj
export PROJ_LIB=$PROJ_DATA
# Need torchgeo's libstdc++ for numpy 2.4; ftw-tile's libstdc++ for gdal.
# Both envs ship CXXABI_1.3.15 / GLIBCXX_3.4.29+, so either works at runtime.
export LD_LIBRARY_PATH=$TORCH_ENV/lib:$GDAL_ENV/lib:$LD_LIBRARY_PATH

# Use the torchgeo Python for orchestration.
export PYTHON=$TORCH_ENV/bin/python
