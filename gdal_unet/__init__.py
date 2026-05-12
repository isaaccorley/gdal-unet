"""gdal-unet: run a U-Net forward pass through GDAL CLI primitives.

User-facing entry point is the `gdal-unet` console script (see `cli.py`).
Internally the heavy per-layer conv-BN-ReLU is delegated to the
`gdal-conv2d` C++ binary.
"""

__version__ = "0.1.0"
