# Reference / historical implementations

These three files are **not part of what `gdal-unet` ships**. They're kept here for context — they document the progression that led to the current
`gdal-conv2d` C++ binary + shell-script approach.

| File | What it is | Why kept |
|---|---|---|
| [`predict_gdal.py`](predict_gdal.py) | Original pure-CLI U-Net forward pass: ~2185 `gdal raster …` subprocesses, ~520 s wall | Demonstrates that the whole network can be expressed in stock GDAL CLI alone, with no custom binary. Useful as a sanity-check oracle for the shell-only path. |
| [`build_vrt.py`](build_vrt.py) | One VRT XML per layer (`KernelFilteredSource` + `expression` pixel functions), materialized via `gdal_translate`. ~31 subprocesses, ~131 s wall | Shows that VRT can express a whole conv layer in one node. Limited by VRT's center-pixel resampling for strided ops, so geometric ops still need numpy. |
| [`profile_layers.py`](profile_layers.py) | Dumps per-stage activations from PyTorch using hooks and compares to each intermediate raster from the GDAL path | The debugger we used to find the boundary-padding + stride-alignment bugs that pushed cosine from 0.29 → 1.0000. Still useful when porting a new backbone. |

All three are bit-equivalent to PyTorch on the chesapeakersc resnet18 model
(IoU 0.6372 vs PyTorch 0.6374). The shipped path (`gdal-conv2d` + `gdal-unet-export`-generated shell) produces the same output in ~5.5 s on a 16-CPU node.

These three files **require** Python at runtime (torch, numpy, rasterio).
The shipped `gdal-conv2d` binary does not.
