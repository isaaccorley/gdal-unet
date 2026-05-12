"""`gdal-unet` console entry point."""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, backbones
from .conv_runner import ConvRunner
from .ops import make_workdir, workdir_base


def _cmd_predict(args: argparse.Namespace) -> int:
    in_tif = Path(args.input).resolve()
    out_tif = Path(args.output or in_tif.with_suffix(".cpp_probs.tif")).resolve()

    weights_npz = np.load(args.weights)
    fwd = backbones.get(args.arch)

    base = Path.cwd() if args.keep_work else workdir_base()
    workdir = make_workdir(base)

    runner = ConvRunner(threads=args.threads, binary=args.binary)

    print(f"[arch]   {args.arch}")
    print(f"[input]  {in_tif.name}")
    print(f"[output] {out_tif.name}")
    print(f"[work]   {workdir}")

    t0 = time.time()
    try:
        fwd(in_tif, out_tif,
            workdir=workdir, weights_npz=weights_npz,
            runner=runner, keep_work=args.keep_work)
    finally:
        if not args.keep_work:
            shutil.rmtree(workdir, ignore_errors=True)
    dt = time.time() - t0
    n = max(runner.ncalls, 1)
    print(f"\n[done] {dt:.1f}s  ({runner.ncalls} gdal-unet-conv subprocesses, "
          f"avg {1000 * runner.t_sub / n:.0f} ms each, "
          f"sub-total = {runner.t_sub:.1f}s)")
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
                    help="output probability GeoTIFF (default: <input>.cpp_probs.tif)")
    sp.add_argument("--arch", default="unet-resnet18",
                    choices=sorted(backbones.REGISTRY),
                    help="model architecture (default: unet-resnet18)")
    sp.add_argument("--weights", required=True,
                    help="path to model_weights.npz")
    sp.add_argument("--threads", type=int, default=None,
                    help="threads per gdal-unet-conv call (default: $GDAL_UNET_THREADS or 16)")
    sp.add_argument("--binary", default=None,
                    help="path to the gdal-unet-conv binary (default: $GDAL_UNET_CONV or PATH)")
    sp.add_argument("--keep-work", action="store_true",
                    help="keep intermediate per-layer rasters in CWD")
    sp.set_defaults(func=_cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
