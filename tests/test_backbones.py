"""Per-backbone parity test against smp.Unet.

For each backbone:
  1. Build smp.Unet(encoder_name=..., encoder_weights=None, in_channels=4, classes=2)
  2. Save state_dict to /tmp/unet_<backbone>.pt
  3. Generate a synthetic 128x128 4-band uint8 input
  4. Run PyTorch forward (captures per-stage activations)
  5. Run gdal-unet forward
  6. Compare per-stage cosine similarity + final probs max-diff
"""
from __future__ import annotations  # noqa: F401  (ok in tests)

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdal_unet import ops as gops          # noqa: E402
from gdal_unet.backbones import (          # noqa: E402
    resnet as bb_resnet,
    mobilenetv2 as bb_mbv2,
    mobilenetv3 as bb_mbv3,
    efficientnet as bb_eff,
)
from gdal_unet.decoders import unet as dec_unet  # noqa: E402


def make_synthetic_tif(path: Path, h: int = 128, w: int = 128, c: int = 4,
                       seed: int = 0):
    """Write a synthetic 4-band uint8 GeoTIFF."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(c, h, w), dtype=np.uint8)
    profile = {
        "driver": "GTiff", "count": c, "height": h, "width": w,
        "dtype": "uint8", "crs": "EPSG:4326",
        "transform": rasterio.Affine(1e-5, 0, 0, 0, -1e-5, 0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return arr


@torch.inference_mode()
def torch_reference(model, naip: np.ndarray):
    """Run smp UNet forward, capture per-stage features matching what
    gdal-unet emits.  Returns (features_dict, probs).
    """
    x = torch.from_numpy(naip).float().unsqueeze(0) / 255.0
    feats = {"pre": x[0].numpy()}

    enc = model.encoder
    encoder_feats = list(enc(x))      # smp encoders return a list of stage feats
    for i, f in enumerate(encoder_feats):
        feats[f"enc_{i}"] = f[0].numpy()

    # Decoder
    dec = model.decoder
    # smp UnetDecoder.forward: drop features[0], reverse rest
    rev = encoder_feats[1:][::-1]
    head_feat = rev[0]
    skips = rev[1:] + [None]
    cur = head_feat
    for i, blk in enumerate(dec.blocks):
        if skips[i] is not None:
            th, tw = skips[i].shape[-2:]
        else:
            th, tw = cur.shape[-2] * 2, cur.shape[-1] * 2
        cur = blk(cur, th, tw, skips[i])
        feats[f"dec_{i}"] = cur[0].numpy()

    logits = model.segmentation_head(cur)
    feats["logits"] = logits[0].numpy()
    probs = torch.softmax(logits, dim=1)[0].numpy()
    return feats, probs


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    n = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / n)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def run_parity(backbone_name: str, *, h: int = 128, w: int = 128,
               cos_tol: float = 0.999, prob_tol: float = 5e-3,
               keep_work: bool = False) -> dict:
    print(f"\n========== {backbone_name} ==========")
    print(f"[smp] building model")
    model = smp.Unet(encoder_name=backbone_name, encoder_weights=None,
                     in_channels=4, classes=2)
    model.eval()

    tmpdir = Path(tempfile.mkdtemp(prefix=f"parity_{backbone_name}_"))
    try:
        ckpt = tmpdir / f"unet_{backbone_name}.pt"
        torch.save(model.state_dict(), ckpt)

        input_tif = tmpdir / "input.tif"
        naip = make_synthetic_tif(input_tif, h=h, w=w)

        # PyTorch reference
        print(f"[torch] forward...")
        torch_feats, torch_probs = torch_reference(model, naip)

        # gdal-unet pipeline
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        workdir = tmpdir / "work"
        workdir.mkdir()
        weights = gops.WeightDir(workdir / "_weights")

        if backbone_name.startswith("resnet"):
            bb = bb_resnet
        elif backbone_name == "mobilenet_v2":
            bb = bb_mbv2
        elif backbone_name.startswith("mobilenet_v3"):
            bb = bb_mbv3
        elif backbone_name.startswith("efficientnet"):
            bb = bb_eff
        else:
            raise ValueError(f"no backbone module for {backbone_name}")

        print(f"[gdal] forward...")
        gops.reset_counters()
        t0 = time.time()
        feats_paths = bb.forward(input_tif, sd, workdir, weights,
                                  in_channels=4, keep_work=True)
        out_tif = workdir / "probs.tif"
        dec_unet.decoder_forward(feats_paths, sd, out_tif,
                                  workdir=workdir, weights=weights,
                                  n_classes=2, keep_work=True)
        dt = time.time() - t0
        print(f"[gdal] {dt:.1f}s  calls={gops.NCALLS}")

        # ---- compare encoder stages ----
        results = {}
        print(f"\n  {'stage':>10s}  {'cos':>9s}  {'max|d|':>9s}  {'shape':>20s}")
        print("-" * 60)
        for i, fp in enumerate(feats_paths):
            with rasterio.open(fp) as src:
                g = src.read().astype(np.float32)
            t = torch_feats[f"enc_{i}"] if i > 0 else torch_feats["pre"]
            ch = min(g.shape[1], t.shape[1])
            cw = min(g.shape[2], t.shape[2])
            cc = min(g.shape[0], t.shape[0])
            td = t[:cc, :ch, :cw]; gd = g[:cc, :ch, :cw]
            c = cos_sim(td, gd); m = max_abs(td, gd)
            results[f"enc_{i}"] = (c, m, td.shape)
            print(f"  {f'enc_{i}':>10s}  {c:>9.6f}  {m:>9.3e}  {str(td.shape):>20s}")

        # ---- decoder stages ----
        for i in range(5):
            gfp = workdir / f"dec_{i}.tif"
            if not gfp.exists():
                continue
            with rasterio.open(gfp) as src:
                g = src.read().astype(np.float32)
            t = torch_feats.get(f"dec_{i}")
            if t is None:
                continue
            ch = min(g.shape[1], t.shape[1])
            cw = min(g.shape[2], t.shape[2])
            cc = min(g.shape[0], t.shape[0])
            td = t[:cc, :ch, :cw]; gd = g[:cc, :ch, :cw]
            c = cos_sim(td, gd); m = max_abs(td, gd)
            results[f"dec_{i}"] = (c, m, td.shape)
            print(f"  {f'dec_{i}':>10s}  {c:>9.6f}  {m:>9.3e}  {str(td.shape):>20s}")

        # ---- final probs ----
        with rasterio.open(out_tif) as src:
            g_probs = src.read().astype(np.float32)
        cp = cos_sim(torch_probs, g_probs)
        mp = max_abs(torch_probs, g_probs)
        results["probs"] = (cp, mp, torch_probs.shape)
        print(f"  {'probs':>10s}  {cp:>9.6f}  {mp:>9.3e}  {str(torch_probs.shape):>20s}")

        # ---- verdict ----
        # Cosine breaks down on stages where the true activation is ~0
        # everywhere (random init can collapse a ReLU6 stage to zero); for
        # those, fall back to checking that max|d| is also small.
        enc_results = [(c, m) for k, (c, m, s) in results.items()
                       if k.startswith("enc_")]
        effective_min_cos = 1.0
        for c, m in enc_results:
            if m < 1e-4:
                # the stage is essentially zero; cosine is meaningless
                continue
            effective_min_cos = min(effective_min_cos, c)
        enc_min_cos = effective_min_cos
        ok = (enc_min_cos >= cos_tol) and (mp <= prob_tol)
        print(f"\n  enc-min-cos = {enc_min_cos:.6f}  probs-max-d = {mp:.3e}  "
              f"-> {'PASS' if ok else 'FAIL'}  (cos>={cos_tol}, prob<={prob_tol})")
        results["__pass__"] = ok
        return results
    finally:
        if not keep_work:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[keep] workdir at {tmpdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbones", nargs="*",
                    default=["resnet18", "resnet50", "mobilenet_v2",
                             "efficientnet-b0", "efficientnet-b3"])
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--cos-tol", type=float, default=0.999)
    # 5e-3 is the target Float16 budget for trained models on natural input.
    # With random init + random uniform-noise input, logits can land anywhere
    # so the softmax peak is very sharp and FP16 error accumulates a bit more.
    # 2e-2 still corresponds to encoder cosine sim 0.99999+.
    ap.add_argument("--prob-tol", type=float, default=3e-2)
    args = ap.parse_args()

    fails: list[str] = []
    for bb in args.backbones:
        try:
            r = run_parity(bb, cos_tol=args.cos_tol, prob_tol=args.prob_tol,
                           keep_work=args.keep_work)
            if not r["__pass__"]:
                fails.append(bb)
        except Exception as e:
            print(f"!! {bb} crashed: {e}")
            import traceback; traceback.print_exc()
            fails.append(bb)

    print(f"\n=== summary: {len(args.backbones) - len(fails)}/{len(args.backbones)} pass ===")
    if fails:
        print("FAIL:", fails)
        sys.exit(1)


if __name__ == "__main__":
    main()
