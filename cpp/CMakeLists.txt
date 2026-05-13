cmake_minimum_required(VERSION 3.15)
project(gdal_conv2d CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Locate GDAL via standard FindGDAL (works for conda, Homebrew, apt).
find_package(GDAL 3.11 REQUIRED)

# OpenMP — on macOS this needs libomp installed (brew install libomp, or
# conda install libomp).  CMake's FindOpenMP handles both apple-clang+libomp
# and the gcc/Linux case transparently as long as the headers are findable.
find_package(OpenMP REQUIRED)

# oneDNN (DNNL) for fast conv kernels (im2col+GEMM, Winograd, fused post-ops).
# conda-forge ships the package as `onednn`; the CMake config exports the
# `DNNL::dnnl` imported target.
find_package(dnnl REQUIRED CONFIG)

add_executable(gdal-conv2d
  src/gdal_conv2d.cpp
  src/conv_dnnl.cpp
  src/ops_stream.cpp
)
target_include_directories(gdal-conv2d PRIVATE src)
target_link_libraries(gdal-conv2d PRIVATE
  GDAL::GDAL
  OpenMP::OpenMP_CXX
  DNNL::dnnl
)
target_compile_options(gdal-conv2d PRIVATE -O3 -Wall)

# Portable RPATH so the binary finds libgdal next to it after install.
# Avoid -march=native: makes the binary CPU-specific, defeats redistribution.
if(APPLE)
  set_target_properties(gdal-conv2d PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/../lib")
else()
  set_target_properties(gdal-conv2d PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

install(TARGETS gdal-conv2d RUNTIME DESTINATION bin)
