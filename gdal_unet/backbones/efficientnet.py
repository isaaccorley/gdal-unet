"""EfficientNet-B0..B7 encoder (stub).

Implementation deferred: structurally identical to MobileNetV2 but with
swish/silu in place of relu6 plus a per-block SE bottleneck.  The SE op
(``ops.se_block``) and swish activation are already wired -- this stub
exists so the package imports cleanly for the resnet/mbv2 tests.
"""


def forward(*args, **kwargs):
    raise NotImplementedError(
        "EfficientNet backbone not yet implemented. "
        "Primitives exist in gdal_unet.ops -- pattern follows mobilenetv2.py."
    )
