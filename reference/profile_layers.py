"""Dump intermediate activations from BOTH the PyTorch reference and the
GDAL-CLI pipeline for the same input, then print stage-by-stage diff stats.

Usage:
    1) python profile_layers.py --torch    # produces profile/torch/*.npy
    2) python predict_gdal.py --keep-work samples/1717_image.tif samples/1717_gdal_probs.tif
    3) python profile_layers.py --compare <work_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.v2 as T

ROOT = Path(__file__).parent.resolve()
PROF = ROOT / "profile"
PROF.mkdir(exist_ok=True)

# (pytorch dump name, gdal intermediate filename)
STAGES = [
    ("pre",     "00_pre.tif"),
    ("stem",    "01_stem.tif"),     # encoder.relu after bn1
    ("mp",      "02_mp.tif"),       # encoder.maxpool
    ("l1",      "02b.tif"),
    ("l2",      "03b.tif"),
    ("l3",      "04b.tif"),
    ("l4",      "05b.tif"),
    ("d0",      "d0.tif"),
    ("d1",      "d1.tif"),
    ("d2",      "d2.tif"),
    ("d3",      "d3.tif"),
    ("d4",      "d4.tif"),
    ("logits",  "logits.tif"),
]


@torch.inference_mode()
def dump_torch(input_tif: Path, mode: str = "nearest"):
    (PROF / "torch").mkdir(exist_ok=True)
    with rasterio.open(input_tif) as src:
        img = src.read()[:4]
    x = torch.from_numpy(img).float().unsqueeze(0)
    x = T.Normalize(mean=[0.0], std=[255.0])(x)
    np.save(PROF / "torch" / "pre.npy", x[0].numpy())

    m = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                 in_channels=4, classes=2, decoder_interpolation=mode)
    sd = torch.load(str(ROOT / "unet-resnet18.pt"),
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd); m.eval()

    # Manually replicate the smp ResNetEncoder.forward so we can save each stage.
    enc = m.encoder
    feats = [x]                                  # stride 1 (= input)
    stem = enc.relu(enc.bn1(enc.conv1(x)))       # stride 2, 64 ch
    feats.append(stem)
    np.save(PROF / "torch" / "stem.npy", stem[0].numpy())

    mp = enc.maxpool(stem)
    np.save(PROF / "torch" / "mp.npy", mp[0].numpy())
    l1 = enc.layer1(mp)
    np.save(PROF / "torch" / "l1.npy", l1[0].numpy())
    feats.append(l1)

    l2 = enc.layer2(l1); np.save(PROF / "torch" / "l2.npy", l2[0].numpy()); feats.append(l2)
    l3 = enc.layer3(l2); np.save(PROF / "torch" / "l3.npy", l3[0].numpy()); feats.append(l3)
    l4 = enc.layer4(l3); np.save(PROF / "torch" / "l4.npy", l4[0].numpy()); feats.append(l4)

    # Decoder.  smp Unet's decoder takes features in reverse order.
    dec = m.decoder
    # features list = [input(stride1), stem(stride2), l1(stride4), l2(stride8), l3(stride16), l4(stride32)]
    # decoder ignores features[0], reverses rest -> head=l4, skips=[l3,l2,l1,stem]
    rev = feats[1:][::-1]  # [l4, l3, l2, l1, stem]
    head_feat = rev[0]
    skips = rev[1:] + [None]  # 5 entries

    # call each block manually using target H,W per next-level skip
    x_cur = head_feat
    # Heights/widths: target for block i is the size of skips[i] if not None,
    # else 2*H_of_x_cur.
    for i, blk in enumerate(dec.blocks):
        if skips[i] is not None:
            th, tw = skips[i].shape[-2:]
        else:
            th, tw = x_cur.shape[-2] * 2, x_cur.shape[-1] * 2
        x_cur = blk(x_cur, th, tw, skips[i])
        np.save(PROF / "torch" / f"d{i}.npy", x_cur[0].numpy())

    logits = m.segmentation_head(x_cur)
    np.save(PROF / "torch" / "logits.npy", logits[0].numpy())

    probs = torch.softmax(logits, dim=1)[0].numpy()
    np.save(PROF / "torch" / "probs.npy", probs)
    print(f"[torch] dumped {sum(1 for _ in (PROF/'torch').glob('*.npy'))} files")


def diff_stats(t: np.ndarray, g: np.ndarray):
    """Crop g to t shape if differ by 1 (alignment edge), return diff stats."""
    if t.shape != g.shape:
        # crop both to common min
        return None
    d = np.abs(t.astype(np.float32) - g.astype(np.float32))
    return dict(
        max=float(d.max()),
        mean=float(d.mean()),
        rms=float(np.sqrt((d ** 2).mean())),
        t_max=float(np.abs(t).max()),
        g_max=float(np.abs(g).max()),
        cos=float(np.dot(t.ravel(), g.ravel()) /
                  (np.linalg.norm(t.ravel()) * np.linalg.norm(g.ravel()) + 1e-12)),
    )


def compare(work_dir: Path):
    """Compare torch dumps vs gdal intermediate tifs in `work_dir`."""
    print(f"\n{'stage':>8s}  {'shape':>20s}  {'max|d|':>9s}  {'mean|d|':>9s}  "
          f"{'rms':>9s}  {'cos':>7s}  {'tmax':>8s}  {'gmax':>8s}")
    print("-" * 92)

    for tag, fname in STAGES:
        tp = PROF / "torch" / f"{tag}.npy"
        gp = work_dir / fname
        if not tp.exists():
            print(f"  {tag:>8s}  no torch dump"); continue
        if not gp.exists():
            print(f"  {tag:>8s}  no gdal tif at {gp.name}"); continue
        t = np.load(tp)             # (C, H, W) float32
        g = rasterio.open(gp).read()  # (C, H, W) float16
        # crop to matching dims if off-by-one
        ch = min(t.shape[1], g.shape[1])
        cw = min(t.shape[2], g.shape[2])
        cc = min(t.shape[0], g.shape[0])
        td = t[:cc, :ch, :cw]
        gd = g[:cc, :ch, :cw].astype(np.float32)
        s = diff_stats(td, gd)
        shape_str = f"{td.shape}"
        print(f"  {tag:>8s}  {shape_str:>20s}  "
              f"{s['max']:>9.3e}  {s['mean']:>9.3e}  {s['rms']:>9.3e}  "
              f"{s['cos']:>7.4f}  {s['t_max']:>8.3f}  {s['g_max']:>8.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--torch", action="store_true",
                   help="dump pytorch reference activations")
    p.add_argument("--mode", default="nearest",
                   help="decoder interpolation_mode (nearest/bilinear/bicubic)")
    p.add_argument("--compare", type=Path,
                   help="path to gdal work_ dir to compare against torch dump")
    p.add_argument("--input", default="samples/1717_image.tif")
    args = p.parse_args()

    if args.torch:
        dump_torch(Path(args.input), mode=args.mode)
    if args.compare:
        compare(args.compare.resolve())
    if not args.torch and not args.compare:
        p.print_help()
