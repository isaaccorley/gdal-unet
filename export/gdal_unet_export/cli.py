"""CLI entry point for gdal-unet-export."""
import argparse
import sys
from pathlib import Path

import torch

from .archs import resnet as resnet_arch
from . import sh_emit


SUPPORTED = ("resnet18", "resnet50")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gdal-unet-export",
        description="Export a PyTorch U-Net .pt to Float16 .bin weights + emit predict_<arch>.sh",
    )
    p.add_argument("ckpt", type=Path, help="Path to .pt state_dict")
    p.add_argument("--arch", required=True, choices=SUPPORTED,
                   help="Backbone architecture")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Output weights directory")
    args = p.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    sd_raw = torch.load(str(args.ckpt), map_location="cpu", weights_only=True)
    sd = sd_raw["state_dict"] if "state_dict" in sd_raw else sd_raw

    (args.output / "arch.txt").write_text(args.arch + "\n")

    if args.arch in ("resnet18", "resnet50"):
        resnet_arch.export(sd, args.output, arch=args.arch)
        sh_emit.emit_resnet(args.output, arch=args.arch, sd=sd)
    else:
        print(f"unsupported arch: {args.arch}", file=sys.stderr)
        return 2

    print(f"[done] exported to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
