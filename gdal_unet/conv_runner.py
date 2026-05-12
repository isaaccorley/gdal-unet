"""Thin wrapper that shells out to the `gdal-unet-conv` C++ binary."""

import os
import shutil
import subprocess
import time
from pathlib import Path

from .ops import WeightDir

CO_FLAGS = ["--co", "BIGTIFF=IF_SAFER", "--co", "INTERLEAVE=BAND"]


def _resolve_binary(explicit: str | None = None) -> str:
    """Find the gdal-unet-conv binary.

    Order: explicit override -> $GDAL_UNET_CONV -> PATH -> a sibling
    `cpp/build/gdal-unet-conv` in the source checkout (developer convenience).
    """
    if explicit:
        return explicit
    env = os.environ.get("GDAL_UNET_CONV")
    if env:
        return env
    found = shutil.which("gdal-unet-conv")
    if found:
        return found
    # fallback: source-tree build dir
    here = Path(__file__).resolve().parent.parent
    cand = here / "cpp" / "build" / "gdal-unet-conv"
    if cand.exists():
        return str(cand)
    raise RuntimeError(
        "gdal-unet-conv binary not found. Build cpp/ or set $GDAL_UNET_CONV."
    )


class ConvRunner:
    """Accounts subprocess count + wall, runs one full conv layer."""

    def __init__(self, *, threads: int | None = None, binary: str | None = None):
        self.binary = _resolve_binary(binary)
        self.threads = threads or int(os.environ.get("GDAL_UNET_THREADS",
                                       os.environ.get("GDALNN_THREADS", "16")))
        self.ncalls = 0
        self.t_sub = 0.0

    def _sh(self, cmd):
        t0 = time.time()
        cmd = [str(c) for c in cmd]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("CMD:", " ".join(cmd[:30]))
            print(e.stderr.decode(errors="replace"))
            raise
        self.ncalls += 1
        self.t_sub += time.time() - t0

    def conv(self, in_tif: Path, out_tif: Path, *,
             kernel_key: str,
             bn_a_key: str | None = None,
             bn_b_key: str | None = None,
             bias_key: str | None = None,
             relu: bool = True,
             stride: int = 1,
             padding: int | None = None,
             weights: WeightDir):
        """Run one conv layer (+ optional BN, bias, ReLU, stride)."""
        Cout, Cin, kH, kW = weights.shape(kernel_key)
        if padding is None:
            padding = kH // 2
        args = [self.binary,
                "--in", str(in_tif),
                "--kernel", str(weights.get(kernel_key)),
                "--kernel-shape", f"{Cout},{Cin},{kH},{kW}",
                "--stride", str(stride),
                "--padding", str(padding),
                "--out", str(out_tif),
                "--threads", str(self.threads),
                *CO_FLAGS]
        if relu:
            args.append("--relu")
        if bn_a_key is not None:
            args += ["--bn-a", str(weights.get(bn_a_key)),
                     "--bn-b", str(weights.get(bn_b_key))]
        if bias_key is not None:
            args += ["--bias", str(weights.get(bias_key))]
        self._sh(args)
