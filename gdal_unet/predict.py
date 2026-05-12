"""End-to-end predict: load a PyTorch ``.pt`` state_dict, run encoder + decoder
+ head via gdal-conv2d subprocesses, write a 2-band Float16 probabilities tif.

Architecture is auto-detected from state_dict keys (or via ``--arch``):
  - ``encoder.conv1.weight`` shape (?, 4, 7, 7) -> resnet*
    (further: conv3 in layer1.0 -> resnet50/101/152, else resnet18/34)
  - ``encoder.features.0.0.weight`` -> mobilenet_v2 (or v3 / efficientnet)
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from . import ops
from .backbones import get_backbone, resnet, mobilenetv2
from .decoders import unet


def detect_arch(sd: dict) -> str:
    if "encoder.conv1.weight" in sd:
        if "encoder.layer1.0.conv3.weight" in sd:
            # could be resnet50/101/152; the number of layer3 blocks
            # disambiguates but smp ignores -- we just say "resnet50" for
            # logging purposes.
            return "resnet50"
        return "resnet18"
    if "encoder.features.0.0.weight" in sd:
        # could be mbv2 / mbv3 / efficientnet -- mbv2 has features.{i}.conv.* ;
        # mbv3 has features.{i}.block.* ; efficientnet uses _blocks (timm).
        if any(k.startswith("encoder.features.1.conv.") for k in sd):
            return "mobilenet_v2"
        if any(k.startswith("encoder.features.1.block.") for k in sd):
            return "mobilenet_v3"
    raise ValueError("Could not auto-detect backbone from state_dict keys; "
                     "pass --arch explicitly.")


def run(input_tif: Path, ckpt: Path, output_tif: Path, *,
        arch: str | None = None, workdir: Path | None = None,
        in_channels: int = 4, n_classes: int = 2, keep_work: bool = False):
    ops.reset_counters()

    sd_raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    # Accept both raw state_dict and {'state_dict': ...} wrappers.
    sd = sd_raw["state_dict"] if "state_dict" in sd_raw else sd_raw

    if arch is None:
        arch = detect_arch(sd)
    print(f"[arch] {arch}")

    if workdir is None:
        base = None
        for cand in (os.environ.get("SLURM_TMPDIR"), "/dev/shm", "/tmp"):
            if cand and Path(cand).is_dir() and os.access(cand, os.W_OK):
                base = Path(cand); break
        if base is None: base = Path.cwd()
        workdir = base / f"gdal_unet_{int(time.time())}_{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)
    weights = ops.WeightDir(workdir / "_weights")

    backbone = get_backbone(arch)
    t0 = time.time()
    feats = backbone.forward(input_tif, sd, workdir, weights,
                              in_channels=in_channels, keep_work=keep_work)
    t_enc = time.time() - t0

    unet.decoder_forward(feats, sd, output_tif,
                          workdir=workdir, weights=weights,
                          n_classes=n_classes, keep_work=keep_work)
    t_total = time.time() - t0
    print(f"[done] arch={arch}  enc={t_enc:.1f}s  total={t_total:.1f}s  "
          f"calls={ops.NCALLS}  sub-total={ops.T_SUB:.1f}s")

    if not keep_work:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--ckpt", required=True, help="PyTorch .pt state_dict")
    ap.add_argument("--arch", default=None,
                    help="override auto-detect (resnet18, resnet50, mobilenet_v2, ...)")
    ap.add_argument("--in-channels", type=int, default=4)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    run(Path(args.input).resolve(), Path(args.ckpt).resolve(),
        Path(args.output).resolve(),
        arch=args.arch, workdir=args.workdir,
        in_channels=args.in_channels, n_classes=args.classes,
        keep_work=args.keep_work)


if __name__ == "__main__":
    main()
