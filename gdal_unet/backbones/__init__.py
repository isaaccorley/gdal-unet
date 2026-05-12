"""Per-backbone encoder modules.

Each module exposes a single function::

    forward(input_tif, state_dict, workdir, weights) -> list[Path]

returning encoder skip features in increasing-stride order (input, stem,
stage1, stage2, stage3, stage4 for resnet-like; analogous positions for
mobilenet / efficientnet).
"""

from . import resnet, mobilenetv2, mobilenetv3, efficientnet  # noqa: F401


def get_backbone(name: str):
    name = name.lower()
    if name.startswith("resnet"):
        return resnet
    if name == "mobilenet_v2":
        return mobilenetv2
    if name.startswith("mobilenet_v3"):
        return mobilenetv3
    if name.startswith("efficientnet"):
        return efficientnet
    raise ValueError(f"unknown / unsupported backbone: {name!r}")
