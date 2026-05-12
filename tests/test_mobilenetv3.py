"""Encoder-only parity test for MobileNetV3 large/small.

smp 0.5.0 does NOT include mobilenet_v3 as a built-in encoder, so we cannot
go through ``smp.Unet(encoder_name="mobilenet_v3_*")``.  Instead, we

  1. instantiate ``torchvision.models.mobilenet_v3_{small,large}``,
  2. monkey-replace the stem ``features.0[0]`` conv to take 4 input channels,
  3. dump its state_dict with an ``encoder.`` prefix (matching the convention
     gdal_unet/backbones/mobilenetv3.py expects),
  4. run torchvision forward + capture the per-stride features,
  5. run gdal_unet.backbones.mobilenetv3.forward,
  6. compare per-stage cosine + max|d|.

Decoder parity is not checked here -- there's no smp-Unet decoder definition
for mbv3 in smp 0.5.0, so we only verify the encoder.  Once smp adds support,
the same state_dict layout will plug into ``test_backbones.py``.
"""
from __future__ import annotations  # noqa: F401

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdal_unet import ops as gops             # noqa: E402
from gdal_unet.backbones import mobilenetv3 as bb  # noqa: E402


def cos_sim(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_parity(variant: str, *, in_channels: int = 4, h: int = 128,
               cos_tol: float = 0.999, keep_work: bool = False):
    """variant in {'large', 'small'}."""
    print(f"\n========== mobilenet_v3_{variant} ==========")
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
    builder = mobilenet_v3_large if variant == "large" else mobilenet_v3_small
    model = builder(weights=None)
    # Re-wire the stem conv to ``in_channels`` inputs.  Keep its bn/activation.
    old = model.features[0][0]
    new_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.features[0][0] = new_conv
    model.eval()

    tmp = Path(tempfile.mkdtemp(prefix=f"mbv3_{variant}_"))
    try:
        # Save state_dict with encoder. prefix.
        raw = model.state_dict()
        sd = {f"encoder.{k}": v for k, v in raw.items()
              if k.startswith("features.")}
        ckpt = tmp / f"mbv3_{variant}.pt"
        torch.save(sd, ckpt)

        # Synthetic input
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (in_channels, h, h), dtype=np.uint8)
        inp = tmp / "input.tif"
        prof = {"driver": "GTiff", "count": in_channels, "height": h, "width": h,
                "dtype": "uint8", "crs": "EPSG:4326",
                "transform": rasterio.Affine(1e-5, 0, 0, 0, -1e-5, 0)}
        with rasterio.open(inp, "w", **prof) as d:
            d.write(arr)

        # Torch reference: forward through features and pluck the same
        # indices that gdal_unet emits.
        with torch.inference_mode():
            x = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
            cfg_n = len(model.features)
            cfg = bb._CONFIGS[cfg_n]
            torch_feats = {"pre": x[0].numpy()}
            cur = x
            captured = {}
            for i, f in enumerate(model.features):
                cur = f(cur)
                if i == 0:
                    torch_feats["stem"] = cur[0].numpy()
                elif i in cfg["capture"] or i == cfg_n - 1:
                    captured[i] = cur[0].numpy()
            # Expected gdal_unet emission order:
            torch_stages = [torch_feats["pre"], torch_feats["stem"]]
            for ci in cfg["capture"]:
                torch_stages.append(captured[ci])
            torch_stages.append(captured[cfg_n - 1])

        # gdal_unet forward
        sd_load = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        workdir = tmp / "work"; workdir.mkdir()
        weights = gops.WeightDir(workdir / "_w")
        gops.reset_counters()
        t0 = time.time()
        fps = bb.forward(inp, sd_load, workdir, weights,
                         in_channels=in_channels, keep_work=True)
        dt = time.time() - t0
        print(f"[gdal] {dt:.1f}s  calls={gops.NCALLS}  nfeats={len(fps)}")

        assert len(fps) == len(torch_stages), (len(fps), len(torch_stages))

        # Compare
        print(f"{'stage':>10s} {'cos':>9s} {'max|d|':>9s}   shape")
        print("-" * 60)
        min_cos = 1.0
        for i, (fp, t) in enumerate(zip(fps, torch_stages)):
            with rasterio.open(fp) as src:
                g = src.read().astype(np.float32)
            ch = min(g.shape[1], t.shape[1])
            cw = min(g.shape[2], t.shape[2])
            cc = min(g.shape[0], t.shape[0])
            td = t[:cc, :ch, :cw]; gd = g[:cc, :ch, :cw]
            c = cos_sim(td, gd); m = float(np.abs(td - gd).max())
            print(f"{f'enc_{i}':>10s} {c:>9.6f} {m:>9.3e}   {td.shape}")
            if m > 1e-4:
                min_cos = min(min_cos, c)
        ok = min_cos >= cos_tol
        print(f"\n  min-cos (nonzero stages) = {min_cos:.6f}  -> {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        if not keep_work:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[keep] workdir at {tmp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=["small", "large"])
    ap.add_argument("--in-channels", type=int, default=4)
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()
    fails = [v for v in args.variants
             if not run_parity(v, in_channels=args.in_channels,
                               keep_work=args.keep_work)]
    print(f"\n=== {len(args.variants) - len(fails)}/{len(args.variants)} pass ===")
    if fails:
        print("FAIL:", fails)
        sys.exit(1)


if __name__ == "__main__":
    main()
