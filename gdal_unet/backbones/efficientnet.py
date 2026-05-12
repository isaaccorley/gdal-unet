"""EfficientNet-B0..B7 encoder (smp / lukemelas EfficientNet-PyTorch port).

smp 0.5.0 wraps `_efficientnet.py` (lukemelas) which uses TF-style
``Conv2dStaticSamePadding``.  Key differences vs torchvision-style:

  - Stem: 3x3 stride 2 conv (no bias) + BN(eps=1e-3) + swish
  - Each block is an MBConvBlock:
      [optional] expand 1x1 conv + BN + swish     (skipped when expand_ratio==1)
      depthwise 3x3 or 5x5 conv (stride 1 or 2) + BN + swish
      SE (reduce 1x1 + swish + expand 1x1 + sigmoid -> per-channel gate)
      project 1x1 conv + BN (no activation)
      residual add if same in/out channels AND stride==1 (id_skip)
  - No final head conv -- the encoder emits the last MBConv output.

State-dict keys (smp):

    encoder._conv_stem.weight                                  (Cout, Cin, 3, 3)
    encoder._bn0.{weight,bias,running_mean,running_var}
    encoder._blocks.{i}._expand_conv.weight                    (only if expand)
    encoder._blocks.{i}._bn0.*                                 (only if expand)
    encoder._blocks.{i}._depthwise_conv.weight                 (C, 1, k, k)
    encoder._blocks.{i}._bn1.*
    encoder._blocks.{i}._se_reduce.{weight,bias}               (Cmid, C, 1, 1)
    encoder._blocks.{i}._se_expand.{weight,bias}               (C, Cmid, 1, 1)
    encoder._blocks.{i}._project_conv.weight                   (Cout, C, 1, 1)
    encoder._blocks.{i}._bn2.*

Encoder features emitted (smp): [input, post-stem, blocks[oi[0]], blocks[oi[1]],
blocks[oi[2]], blocks[oi[3]]] -- at strides 1, 2, 4, 8, 16, 32.

The TF SAME-padding is asymmetric on stride-2 layers when input H/W is even
(pad goes (0,1,0,1) instead of symmetric (1,1,1,1) for k=3).  Our
gdal-conv2d only supports symmetric padding, so for stride-2 convs we
pre-pad in numpy via ``ops.pad_same_asym``.
"""
from pathlib import Path

import numpy as np
import rasterio

from .. import ops


# BN epsilon used by efficientnet (lukemelas / smp) -- NOT 1e-5.
BN_EPS = 1e-3


def _bn(sd: dict, prefix: str) -> tuple:
    return (
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
        BN_EPS,
    )


# Map nblocks (encoder._blocks length) -> (variant_name, out_indexes, strides).
# strides[i] is the stride of block i's depthwise conv.  Values copied
# directly from smp 0.5.0's efficientnet encoder registry (the strides come
# from `efficient_net_encoders` config + `BlockArgs` per variant).
_VARIANTS = {
    # nblocks: (variant_name, out_indexes, strides)
    16: ("efficientnet-b0", [2, 4, 8, 15],
         [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]),
    23: ("efficientnet-b1/b2", [4, 7, 15, 22],
         [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]),
    26: ("efficientnet-b3", [4, 7, 17, 25],
         [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]),
    32: ("efficientnet-b4", [5, 9, 21, 31],
         [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    39: ("efficientnet-b5", [7, 12, 26, 38],
         [1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    45: ("efficientnet-b6", [8, 14, 30, 44],
         [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    55: ("efficientnet-b7", [10, 17, 37, 54],
         [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
}


def _count_blocks(sd: dict) -> int:
    n = 0
    while any(k.startswith(f"encoder._blocks.{n}.") for k in sd):
        n += 1
    return n


def _has_expand(sd: dict, idx: int) -> bool:
    return f"encoder._blocks.{idx}._expand_conv.weight" in sd


def _has_se(sd: dict, idx: int) -> bool:
    return f"encoder._blocks.{idx}._se_reduce.weight" in sd


def _same_conv(in_tif: Path, out_tif: Path, *, weight: np.ndarray, bn,
               activation: str, stride: int, k: int,
               weights: ops.WeightDir, workdir: Path, key: str,
               depthwise: bool = False, keep_work: bool = False):
    """Conv with TF-SAME padding.  Handles asymmetric pad via pre-pad."""
    # For stride 1 SAME pad is symmetric k//2; just call conv directly.
    if stride == 1:
        ops.conv(in_tif, out_tif, weight=weight, bn=bn, activation=activation,
                 stride=1, padding=k // 2, depthwise=depthwise,
                 weights=weights, key=key)
        return
    # stride 2: may be asymmetric.  Pre-pad to a temporary, then conv with
    # padding=0.
    pre = workdir / f"{key}_prepad.tif"
    ops.pad_same_asym(in_tif, pre, k=k, stride=stride)
    ops.conv(pre, out_tif, weight=weight, bn=bn, activation=activation,
             stride=stride, padding=0, depthwise=depthwise,
             weights=weights, key=key)
    if not keep_work:
        pre.unlink(missing_ok=True)


def _se(in_tif: Path, out_tif: Path, sd: dict, *, idx: int,
        workdir: Path, weights: ops.WeightDir, keep_work: bool = False):
    p = f"encoder._blocks.{idx}"
    w1 = sd[f"{p}._se_reduce.weight"].numpy()
    b1 = sd[f"{p}._se_reduce.bias"].numpy()
    w2 = sd[f"{p}._se_expand.weight"].numpy()
    b2 = sd[f"{p}._se_expand.bias"].numpy()
    # efficientnet SE uses swish on the bottleneck, sigmoid on the gate.
    ops.se_block(in_tif, out_tif, w1=w1, b1=b1, w2=w2, b2=b2,
                 act1="swish", workdir=workdir, weights=weights,
                 key=f"eff_b{idx}_se")


def _block(in_tif: Path, out_tif: Path, *, idx: int, stride: int, sd: dict,
           workdir: Path, weights: ops.WeightDir, keep_work: bool = False):
    p = f"encoder._blocks.{idx}"
    pk = f"eff_b{idx}"
    has_exp = _has_expand(sd, idx)
    has_se = _has_se(sd, idx)

    # ---- expand 1x1 ----
    if has_exp:
        w = sd[f"{p}._expand_conv.weight"].numpy()
        bn = _bn(sd, f"{p}._bn0")
        t = workdir / f"{pk}_exp.tif"
        ops.conv(in_tif, t, weight=w, bn=bn, activation="swish",
                 stride=1, padding=0, weights=weights, key=f"{pk}_exp")
        cur = t
    else:
        cur = in_tif

    # ---- depthwise k x k (stride 1 or 2) ----
    w = sd[f"{p}._depthwise_conv.weight"].numpy()
    k = w.shape[-1]
    bn = _bn(sd, f"{p}._bn1")
    t_dw = workdir / f"{pk}_dw.tif"
    _same_conv(cur, t_dw, weight=w, bn=bn, activation="swish",
               stride=stride, k=k, weights=weights, workdir=workdir,
               key=f"{pk}_dw", depthwise=True, keep_work=keep_work)
    if has_exp and not keep_work:
        cur.unlink(missing_ok=True)

    # ---- SE ----
    if has_se:
        t_se = workdir / f"{pk}_se.tif"
        _se(t_dw, t_se, sd, idx=idx, workdir=workdir, weights=weights)
        if not keep_work:
            t_dw.unlink(missing_ok=True)
        cur2 = t_se
    else:
        cur2 = t_dw

    # ---- project 1x1 (no activation) ----
    w = sd[f"{p}._project_conv.weight"].numpy()
    bn = _bn(sd, f"{p}._bn2")
    t_pj = workdir / f"{pk}_pj.tif"
    ops.conv(cur2, t_pj, weight=w, bn=bn, activation="none",
             stride=1, padding=0, weights=weights, key=f"{pk}_pj")
    if not keep_work:
        cur2.unlink(missing_ok=True)

    # ---- residual ----
    Cin = rasterio.open(in_tif).count
    Cout = rasterio.open(t_pj).count
    if stride == 1 and Cin == Cout:
        ops.add(t_pj, in_tif, out_tif, activation="none")
        if not keep_work:
            t_pj.unlink(missing_ok=True)
    else:
        t_pj.rename(out_tif)


def forward(input_tif: Path, sd: dict, workdir: Path, weights: ops.WeightDir,
            *, in_channels: int = 4, keep_work: bool = False) -> list[Path]:
    nb = _count_blocks(sd)
    if nb not in _VARIANTS:
        raise ValueError(f"efficientnet: unknown block count {nb}; "
                         f"supported: {sorted(_VARIANTS)}")
    name, out_indexes, strides = _VARIANTS[nb]

    feats: list[Path] = []

    # /255 -> Float16
    pre = workdir / "00_pre.tif"
    ops.preprocess(input_tif, pre, in_channels=in_channels)
    feats.append(pre)

    # Stem: 3x3 stride 2 + BN(1e-3) + swish, TF SAME pad
    w = sd["encoder._conv_stem.weight"].numpy()
    bn = _bn(sd, "encoder._bn0")
    stem = workdir / "eff_stem.tif"
    _same_conv(pre, stem, weight=w, bn=bn, activation="swish",
               stride=2, k=w.shape[-1], weights=weights, workdir=workdir,
               key="eff_stem", depthwise=False, keep_work=keep_work)
    feats.append(stem)

    cur = stem
    captured: dict[int, Path] = {}
    last_blk_out = None
    for i in range(nb):
        s = strides[i]
        out = workdir / f"eff_blk{i}.tif"
        _block(cur, out, idx=i, stride=s, sd=sd, workdir=workdir,
               weights=weights, keep_work=keep_work)
        if i in out_indexes:
            captured[i] = out
        # cleanup intermediate block outputs that aren't kept
        if not keep_work and cur != stem and cur not in captured.values() and cur != out:
            Path(cur).unlink(missing_ok=True)
        cur = out
        last_blk_out = out

    for oi in out_indexes:
        feats.append(captured[oi])
    return feats
