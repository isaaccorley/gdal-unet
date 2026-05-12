"""MobileNetV3 encoder (stub).

Implementation deferred: requires per-block hswish vs relu, SE blocks at
specific indices, and the torchvision feature.{i}.block.{j} key convention.
The SE primitive in ops.se_block is already in place.
"""


def forward(*args, **kwargs):
    raise NotImplementedError(
        "MobileNetV3 backbone not yet implemented. "
        "All primitives (depthwise, hswish, SE) exist in gdal_unet.ops -- "
        "see gdal_unet/backbones/mobilenetv2.py for the analogous wiring."
    )
