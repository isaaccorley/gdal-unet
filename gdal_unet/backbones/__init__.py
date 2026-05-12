"""Backbones for gdal-unet. Each module exports a `forward(...)` function."""

from . import resnet18

REGISTRY = {
    "unet-resnet18": resnet18.forward,
}


def get(name: str):
    if name not in REGISTRY:
        raise ValueError(
            f"unknown backbone {name!r}; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
