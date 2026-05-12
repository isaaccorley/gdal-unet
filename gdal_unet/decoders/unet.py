"""smp.Unet decoder + segmentation_head, gdal-CLI style.

The decoder is identical for every smp encoder: same key conventions, same
op sequence.  Only the channel widths change.  We walk the state_dict at
runtime to learn those widths.
"""

from pathlib import Path

import numpy as np
import rasterio

from .. import ops


def _bn_tuple(sd: dict, prefix: str) -> tuple:
    return (
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
        1e-5,
    )


def _shape_hw(tif: Path) -> tuple[int, int]:
    with rasterio.open(tif) as src:
        return src.height, src.width


def decoder_forward(
    skip_feats: list[Path | None],
    sd: dict,
    out_tif: Path,
    *,
    workdir: Path,
    weights: ops.WeightDir,
    n_classes: int = 2,
    keep_work: bool = False,
):
    """Run smp Unet decoder + segmentation head.

    ``skip_feats`` is the list of encoder features in INCREASING stride order
    (input, stem, l1, l2, l3, l4 for resnet-like; identical positions for
    every other smp encoder family).  Following smp.UnetDecoder.forward we
    drop the first (full-resolution input) feature, reverse, take the deepest
    as the bottleneck, and use the remaining 4 as skips -- with a 5th, None,
    for the final block.
    """
    rev = list(reversed(skip_feats[1:]))   # [deepest, ..., stem]
    x = rev[0]
    skips = rev[1:] + [None]              # 5 skip slots (some may be None)

    # Discover number of decoder blocks by counting state_dict keys.  This
    # is always 5 for the default smp.Unet but we don't hardcode it.
    n_blocks = max(int(k.split(".")[2]) for k in sd
                   if k.startswith("decoder.blocks.")) + 1

    cur = x
    for i in range(n_blocks):
        # 1. upsample to target H,W
        if skips[i] is not None:
            th, tw = _shape_hw(skips[i])
        else:
            ch, cw = _shape_hw(cur)
            th, tw = ch * 2, cw * 2
        up = workdir / f"dec_{i}_up.tif"
        ops.upsample_to(cur, up, th, tw)

        # 2. concat with skip if present
        if skips[i] is not None:
            cat = workdir / f"dec_{i}_cat.tif"
            ops.concat([up, skips[i]], cat)
            if not keep_work:
                up.unlink(missing_ok=True)
            blk_in = cat
        else:
            blk_in = up

        # 3. conv1 + BN + ReLU
        w1 = sd[f"decoder.blocks.{i}.conv1.0.weight"].numpy()
        bn1 = _bn_tuple(sd, f"decoder.blocks.{i}.conv1.1")
        c1 = workdir / f"dec_{i}_c1.tif"
        ops.conv(blk_in, c1, weight=w1, bn=bn1, activation="relu",
                 weights=weights, key=f"dec{i}_c1")
        if not keep_work:
            blk_in.unlink(missing_ok=True)

        # 4. conv2 + BN + ReLU
        w2 = sd[f"decoder.blocks.{i}.conv2.0.weight"].numpy()
        bn2 = _bn_tuple(sd, f"decoder.blocks.{i}.conv2.1")
        c2 = workdir / f"dec_{i}.tif"
        ops.conv(c1, c2, weight=w2, bn=bn2, activation="relu",
                 weights=weights, key=f"dec{i}_c2")
        if not keep_work:
            c1.unlink(missing_ok=True)
            if cur != x and cur != skip_feats[-1]:
                Path(cur).unlink(missing_ok=True)
        cur = c2

    # segmentation head
    head_w = sd["segmentation_head.0.weight"].numpy()
    head_b = sd["segmentation_head.0.bias"].numpy()
    logits = workdir / "logits.tif"
    Cout, Cin, kH, kW = head_w.shape
    ops.conv(cur, logits, weight=head_w, bias=head_b,
             activation="none", padding=kH // 2,
             weights=weights, key="head")
    if not keep_work:
        Path(cur).unlink(missing_ok=True)

    # softmax
    if n_classes == 2:
        ops.softmax_2cls(logits, out_tif)
    else:
        ops.softmax_kcls(logits, out_tif)
    if not keep_work:
        logits.unlink(missing_ok=True)
