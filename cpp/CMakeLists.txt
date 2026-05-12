cmake_minimum_required(VERSION 3.15)
project(gdalnn_conv CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ENV_PREFIX "/projects/bgtj/isaaccorley/envs/ftw-tile" CACHE PATH "GDAL env prefix")

find_package(OpenMP REQUIRED)

# GDAL via the env prefix (no FindGDAL needed)
add_executable(gdalnn_conv src/gdalnn_conv.cpp)
target_include_directories(gdalnn_conv PRIVATE ${ENV_PREFIX}/include)
target_link_directories(gdalnn_conv PRIVATE ${ENV_PREFIX}/lib)
target_link_libraries(gdalnn_conv PRIVATE gdal OpenMP::OpenMP_CXX)
target_compile_options(gdalnn_conv PRIVATE -O3 -march=native -Wall)
# Force the env's libstdc++/libgcc_s to be found first by using DT_RPATH
# (searched before LD_LIBRARY_PATH and before compiler-injected rpaths).
# Put env's lib FIRST in DT_RPATH so its libstdc++/libgcc_s win over the
# compiler-toolchain ones (env ships libstdc++.so.6.0.34 with the GLIBCXX
# symbols GDAL needs).
target_link_options(gdalnn_conv PRIVATE
  "-Wl,--disable-new-dtags"
  "-Wl,-rpath,${ENV_PREFIX}/lib")
set_target_properties(gdalnn_conv PROPERTIES
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${ENV_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE)

# Post-build: rewrite RPATH so env's libstdc++/libgcc_s win, and force DT_RPATH
# (searched before LD_LIBRARY_PATH, and propagated transitively to libgdal's
# deps) rather than DT_RUNPATH (which is not transitive).
find_program(PATCHELF patchelf)
if(PATCHELF)
  add_custom_command(TARGET gdalnn_conv POST_BUILD
    COMMAND ${PATCHELF} --force-rpath --set-rpath
      "${ENV_PREFIX}/lib:$<TARGET_PROPERTY:gdalnn_conv,INSTALL_RPATH>"
      "$<TARGET_FILE:gdalnn_conv>"
    COMMENT "Forcing DT_RPATH with env libs first")
endif()
