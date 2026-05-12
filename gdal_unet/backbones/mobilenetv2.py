"""MobileNetV2 encoder (torchvision-style, as wrapped by smp).

State dict layout (smp wraps torchvision mobilenet_v2):

    encoder.features.0.0.weight                 # stem 3x3 stride 2 conv
    encoder.features.0.1.{weight,bias,running_mean,running_var}  # BN
    encoder.features.{1..17}.conv.0.0.weight    # expand 1x1 (skipped on InvertedResidual that has expand_ratio=1)
    encoder.features.{1..17}.conv.0.1.*         # expand BN
    encoder.features.{1..17}.conv.1.0.weight    # 3x3 depthwise (or .0 if no expand)
    encoder.features.{1..17}.conv.1.1.*         # depthwise BN
    encoder.features.{1..17}.conv.2.weight      # 1x1 project (or .1 if no expand)
    encoder.features.{1..17}.conv.2.*           # project BN
    encoder.features.18.0.weight                # final 1x1 to 1280
    (smp uses out_channels=[3, 16, 24, 32, 96, 320]; only blocks producing these
     stages are surfaced as skips)

Strides per feature index (torchvision):
    0: stride 2 (stem) -> 32ch  -> stride 2
    1: ir t=1  -> 16ch  -> stride 2
    2,3: ir s=2,1 -> 24ch -> stride 4
    4,5,6: s=2,1,1 -> 32 -> stride 8
    7..10: s=2,1,1,1 -> 64 -> stride 16
    11,12,13: 96 -> stride 16
    14,15,16: s=2,1,1 -> 160 -> stride 32
    17: 320 -> stride 32

smp encoder.out_channels = (in_chan, 16, 24, 32, 96, 320).
"""
from pathlib import Path

from .. import ops


def _bn(sd: dict, prefix: str):
    return (
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
        1e-5,
    )


def _ir_block(in_tif, out_tif, *, idx, sd, workdir, weights, stride,
              keep_work=False):
    """One InvertedResidual block at encoder.features.{idx}.conv.*"""
    p = f"encoder.features.{idx}.conv"
    # Detect whether this block has an expand 1x1: present iff
    # `{p}.0.0.weight` is a 1x1 kernel (== `[Cmid, Cin, 1, 1]`).
    has_expand = (f"{p}.0.0.weight" in sd
                  and sd[f"{p}.0.0.weight"].shape[-1] == 1)
    pk = f"f{idx}"

    if has_expand:
        # 1x1 expand (conv + bn + relu6)
        w_e = sd[f"{p}.0.0.weight"].numpy()
        bn_e = _bn(sd, f"{p}.0.1")
        t_e = workdir / f"{pk}_e.tif"
        ops.conv(in_tif, t_e, weight=w_e, bn=bn_e, activation="relu6",
                 stride=1, padding=0, weights=weights, key=f"{pk}_e")
        dw_idx = 1
        pj_idx = 2
        cur = t_e
    else:
        dw_idx = 0
        pj_idx = 1
        cur = in_tif

    # 3x3 depthwise (groups=Cmid)
    w_dw = sd[f"{p}.{dw_idx}.0.weight"].numpy()
    bn_dw = _bn(sd, f"{p}.{dw_idx}.1")
    t_dw = workdir / f"{pk}_dw.tif"
    ops.conv(cur, t_dw, weight=w_dw, bn=bn_dw, activation="relu6",
             stride=stride, padding=1, depthwise=True,
             weights=weights, key=f"{pk}_dw")
    if has_expand and not keep_work:
        cur.unlink(missing_ok=True)

    # 1x1 project (conv + bn, NO activation)
    w_pj = sd[f"{p}.{pj_idx}.weight"].numpy()
    bn_pj = _bn(sd, f"{p}.{pj_idx + 1}")
    t_pj = workdir / f"{pk}_pj.tif"
    ops.conv(t_dw, t_pj, weight=w_pj, bn=bn_pj, activation="none",
             stride=1, padding=0, weights=weights, key=f"{pk}_pj")
    if not keep_work:
        t_dw.unlink(missing_ok=True)

    # Residual: only if stride==1 and Cin==Cout
    import rasterio
    Cin = rasterio.open(in_tif).count
    Cout = rasterio.open(t_pj).count
    if stride == 1 and Cin == Cout:
        ops.add(t_pj, in_tif, out_tif, activation="none")
        if not keep_work:
            t_pj.unlink(missing_ok=True)
    else:
        # rename
        t_pj.rename(out_tif)


# Stages: smp emits feature AFTER these indices.
# encoder.out_channels[-5:] = [16, 24, 32, 96, 1280]
#   features.1   ->  16  stride 2
#   features.3   ->  24  stride 4
#   features.6   ->  32  stride 8
#   features.13  ->  96  stride 16
#   features.18  -> 1280 stride 32   (this is the post-stage 1x1 expand)
STAGE_END = [1, 3, 6, 13, 18]

# stride for each feature index (1..17); stem is index 0 (stride-2 conv)
IR_STRIDE = {
    1: 1,   # but H halved by stem already -> overall stride 2
    2: 2, 3: 1,
    4: 2, 5: 1, 6: 1,
    7: 2, 8: 1, 9: 1, 10: 1,
    11: 1, 12: 1, 13: 1,
    14: 2, 15: 1, 16: 1,
    17: 1,
}


def forward(input_tif: Path, sd: dict, workdir: Path, weights: ops.WeightDir,
            *, in_channels: int = 4, keep_work: bool = False) -> list[Path]:
    feats: list[Path] = []

    pre = workdir / "00_pre.tif"
    ops.preprocess(input_tif, pre, in_channels=in_channels)
    feats.append(pre)

    # Stem (features.0): 3x3 stride 2 + BN + relu6
    w0 = sd["encoder.features.0.0.weight"].numpy()
    bn0 = _bn(sd, "encoder.features.0.1")
    stem = workdir / "stem.tif"
    ops.conv(pre, stem, weight=w0, bn=bn0, activation="relu6",
             stride=2, padding=1, weights=weights, key="mbv2_stem")

    cur = stem
    stage_outs: dict[int, Path] = {}
    for idx in range(1, 18):
        stride = IR_STRIDE[idx]
        out = workdir / f"mbv2_f{idx}.tif"
        _ir_block(cur, out, idx=idx, sd=sd, workdir=workdir,
                  weights=weights, stride=stride, keep_work=keep_work)
        if idx in STAGE_END:
            stage_outs[idx] = out
        if not keep_work and cur != stem and cur not in stage_outs.values() and cur != out:
            Path(cur).unlink(missing_ok=True)
        cur = out

    # features.18: 1x1 conv 320 -> 1280, BN, ReLU6.
    w18 = sd["encoder.features.18.0.weight"].numpy()
    bn18 = _bn(sd, "encoder.features.18.1")
    f18 = workdir / "mbv2_f18.tif"
    ops.conv(cur, f18, weight=w18, bn=bn18, activation="relu6",
             stride=1, padding=0, weights=weights, key="mbv2_f18")
    if not keep_work and cur not in stage_outs.values():
        Path(cur).unlink(missing_ok=True)
    stage_outs[18] = f18

    # smp Unet decoder expects [input, s2, s4, s8, s16, s32].
    feats += [stage_outs[1], stage_outs[3], stage_outs[6],
              stage_outs[13], stage_outs[18]]
    return feats
