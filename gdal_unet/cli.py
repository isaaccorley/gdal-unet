"""`gdal-unet` console entry point.

Dispatches to `gdal_unet.predict.run` for the actual forward pass.  Weights
load directly from a PyTorch `.pt` state-dict; architecture is auto-detected
from the state-dict keys or overridden via `--arch`.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .predict import run as predict_run


def _cmd_predict(args: argparse.Namespace) -> int:
    in_tif = Path(args.input).resolve()
    out_tif = Path(args.output) if args.output else \
        in_tif.with_suffix(".cpp_probs.tif")
    predict_run(
        in_tif,
        Path(args.ckpt).resolve(),
        out_tif,
        arch=args.arch,
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        keep_work=args.keep_work,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gdal-unet",
        description="Run U-Net inference through GDAL CLI primitives.",
    )
    p.add_argument("--version", action="version",
                   version=f"gdal-unet {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("predict",
                        help="run forward pass on a single GeoTIFF")
    sp.add_argument("input", help="input GeoTIFF (4-band uint8, NAIP-like)")
    sp.add_argument("output", nargs="?", default=None,
                    help="output 2-band Float16 probability GeoTIFF "
                         "(default: <input>.cpp_probs.tif)")
    sp.add_argument("--ckpt", required=True,
                    help="path to PyTorch .pt state-dict (smp.Unet checkpoint)")
    sp.add_argument("--arch", default=None,
                    help="backbone name (e.g. resnet18, resnet50, "
                         "mobilenet_v2); auto-detected from state_dict keys "
                         "if omitted")
    sp.add_argument("--in-channels", type=int, default=4,
                    help="number of input bands (default: 4 for RGBN)")
    sp.add_argument("--n-classes", type=int, default=2,
                    help="number of output classes (default: 2)")
    sp.add_argument("--keep-work", action="store_true",
                    help="keep intermediate per-layer rasters in workdir")
    sp.set_defaults(func=_cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
