"""MobileNetV3-{Small,Large} encoder, torchvision state-dict layout.

NOTE: smp 0.5.0 does NOT ship a ``mobilenet_v3_*`` encoder.  This module
targets the **torchvision** ``mobilenet_v3_{small,large}`` state-dict layout
directly so that:

  * any external code base that wraps torchvision-mbv3 as the encoder of an
    smp Unet (prefixing keys with ``encoder.``) will load here as expected,
  * we can parity-test against torchvision's classifier model directly
    (see tests/test_mobilenetv3.py).

Architecture (torchvision):

  features.0 = Conv2dNormActivation (3x3 stride 2, BN, Hardswish) -- stem
  features.{1..N-2} = InvertedResidual blocks; each has a ``.block`` Sequential:
      block.0 = expand 1x1 + BN + activation   (skipped if expand==input width)
      block.1 = dw 3x3 or 5x5 + BN + activation
      block.2 = (optional) SqueezeExcitation
      block.{2 or 3} = project 1x1 + BN (no activation)
  features.{N-1} = Conv2dNormActivation (1x1, BN, Hardswish) -- tail

Each InvertedResidual has:
  - use_res_connect (bool): residual add when stride==1 AND in==out
  - hardswish if "use_hs" else relu     (per-block activation)
  - SqueezeExcitation when present has:
      .fc1 (1x1 conv to squeezed),  ReLU,
      .fc2 (1x1 conv back),         Hardsigmoid -> per-channel gate.

State-dict key naming (relative to encoder prefix):

  features.0.0.weight                                  stem conv
  features.0.1.{weight,bias,running_mean,running_var}  stem BN
  features.{i}.block.{j}.0.weight                      Conv2dNormActivation conv
  features.{i}.block.{j}.1.*                           Conv2dNormActivation BN
  features.{i}.block.{j}.fc1.{weight,bias}             SE fc1
  features.{i}.block.{j}.fc2.{weight,bias}             SE fc2
  features.{i}.block.{LAST}.0.weight                   project conv (no act module)
  features.{i}.block.{LAST}.1.*                        project BN

For smp-Unet compat, every key in this module is read with the prefix
``encoder.`` (set ``ENC_PREFIX = ""`` to read raw torchvision keys).

Feature emission (smp-style, [stride 1, 2, 4, 8, 16, 32]):

  - input
  - features.0 output (stride 2)
  - encoder feature at stride 4   (block before the stride-2 to stride-8 transition)
  - feature at stride 8
  - feature at stride 16
  - features.{N-1} output (stride 32; the 1x1 tail conv)
"""
from pathlib import Path

import numpy as np
import rasterio

from .. import ops


def _bn(sd: dict, prefix: str) -> tuple:
    return (
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
        1e-5,
    )


def _key_exists(sd: dict, key: str) -> bool:
    return key in sd


def _count_features(sd: dict, enc: str) -> int:
    n = 0
    while (f"{enc}features.{n}.0.weight" in sd
           or f"{enc}features.{n}.block.0.0.weight" in sd):
        n += 1
    return n


def _infer_enc_prefix(sd: dict) -> str:
    """Return ``"encoder."`` if keys are prefixed (smp-style), else ``""``."""
    if "encoder.features.0.0.weight" in sd:
        return "encoder."
    if "features.0.0.weight" in sd:
        return ""
    raise ValueError("mobilenetv3: no recognizable features.0.0.weight key")


# Torchvision mobilenet_v3 per-block configs:
# (total features incl. stem+tail) -> per-block (dw_stride, dw_k, has_se, use_hs)
# Index in tuple corresponds to features[i] for i in 1..N-2 (the IR blocks).
_CONFIGS = {
    # large: 17 features total (1 stem + 15 IR + 1 tail)
    17: dict(
        name="mobilenet_v3_large",
        # per IR block: (stride, kernel, has_se, use_hs)
        ir=[
            (1, 3, False, False),  # 1
            (2, 3, False, False),  # 2
            (1, 3, False, False),  # 3
            (2, 5, True,  False),  # 4
            (1, 5, True,  False),  # 5
            (1, 5, True,  False),  # 6
            (2, 3, False, True),   # 7
            (1, 3, False, True),   # 8
            (1, 3, False, True),   # 9
            (1, 3, False, True),   # 10
            (1, 3, True,  True),   # 11
            (1, 3, True,  True),   # 12
            (2, 5, True,  True),   # 13
            (1, 5, True,  True),   # 14
            (1, 5, True,  True),   # 15
        ],
        # IR block indices at which to capture features for the smp Unet:
        # cumulative stride after each block (stem starts at stride 2):
        #   blocks 1..3 -> stride 4 (block 2 is stride-2 transition; block 3 last)
        #   blocks 4..6 -> stride 8
        #   blocks 7..12 -> stride 16
        #   blocks 13..15 -> stride 32
        # smp emits features just BEFORE each downsample (the last block at
        # each stride).  For Unet:
        #   stride 4  -> block 3
        #   stride 8  -> block 6
        #   stride 16 -> block 12
        #   stride 32 -> tail (features.16)
        capture=[3, 6, 12],   # plus tail (features.16) emitted separately
    ),
    # small: 13 features (1 stem + 11 IR + 1 tail)
    13: dict(
        name="mobilenet_v3_small",
        ir=[
            (2, 3, True,  False),  # 1  -> stride 4
            (2, 3, False, False),  # 2
            (1, 3, False, False),  # 3
            (2, 5, True,  True),   # 4
            (1, 5, True,  True),   # 5
            (1, 5, True,  True),   # 6
            (1, 5, True,  True),   # 7
            (1, 5, True,  True),   # 8
            (2, 5, True,  True),   # 9
            (1, 5, True,  True),   # 10
            (1, 5, True,  True),   # 11
        ],
        # Strides: stem -> 2, block 1 s=2 -> 4, block 2 s=2 -> 8, block 4 s=2 ->16,
        # block 9 s=2 -> 32.
        #   stride 4: block 1 (only one at this scale)
        #   stride 8: block 3 (last at scale 8: blocks 2-3)
        #   stride 16: block 8 (last at scale 16: blocks 4-8)
        #   stride 32: tail (features.12)
        capture=[1, 3, 8],
    ),
}


def _project_idx(sd: dict, enc: str, i: int) -> int:
    """Return the block.{j} index that is the project conv (largest j with
    a ``block.{j}.0.weight`` key).  Walks up to j=5 -- enough since
    torchvision mbv3 blocks have at most 4 entries.
    """
    last = -1
    for j in range(6):
        if f"{enc}features.{i}.block.{j}.0.weight" in sd:
            last = j
    if last < 0:
        raise KeyError(f"no project conv found in features.{i}.block")
    return last


def _has_expand(sd: dict, enc: str, i: int) -> bool:
    """If block.0's conv has 1x1 kernel and is followed by another conv, it's
    the expand.  In torchvision, mbv3 always has an expand 1x1 EXCEPT for
    block 1 of mbv3_large (which is c=16 -> c=16 with no expansion).
    Heuristic: expand exists iff block.0.0.weight kernel size is 1.
    """
    w = sd[f"{enc}features.{i}.block.0.0.weight"]
    return w.shape[-1] == 1


def _has_se(sd: dict, enc: str, i: int) -> bool:
    return any(k.startswith(f"{enc}features.{i}.block.") and ".fc1." in k
               for k in sd)


def _se_block_j(sd: dict, enc: str, i: int) -> int:
    """Return the block.{j} index that holds the SE."""
    for k in sd:
        pref = f"{enc}features.{i}.block."
        if k.startswith(pref) and ".fc1.weight" in k:
            return int(k[len(pref):].split(".")[0])
    raise KeyError(f"no SE in features.{i}")


def _ir_block(in_tif: Path, out_tif: Path, *, sd: dict, enc: str, i: int,
              stride: int, k: int, has_se: bool, use_hs: bool,
              workdir: Path, weights: ops.WeightDir, keep_work: bool = False):
    act = "hswish" if use_hs else "relu"
    pref = f"{enc}features.{i}.block"
    pk = f"mbv3_f{i}"
    has_exp = _has_expand(sd, enc, i)

    # ---- expand 1x1 + BN + act (optional) ----
    if has_exp:
        w = sd[f"{pref}.0.0.weight"].numpy()
        bn = _bn(sd, f"{pref}.0.1")
        t = workdir / f"{pk}_e.tif"
        ops.conv(in_tif, t, weight=w, bn=bn, activation=act,
                 stride=1, padding=0, weights=weights, key=f"{pk}_e")
        dw_j = 1
        cur = t
    else:
        dw_j = 0
        cur = in_tif

    # ---- depthwise ----
    w = sd[f"{pref}.{dw_j}.0.weight"].numpy()
    bn = _bn(sd, f"{pref}.{dw_j}.1")
    # mbv3 uses SYMMETRIC padding = k // 2 (torchvision style), unlike efficientnet.
    t_dw = workdir / f"{pk}_dw.tif"
    ops.conv(cur, t_dw, weight=w, bn=bn, activation=act,
             stride=stride, padding=k // 2, depthwise=True,
             weights=weights, key=f"{pk}_dw")
    if has_exp and not keep_work:
        cur.unlink(missing_ok=True)

    # ---- SE (optional; hard-sigmoid gate) ----
    if has_se:
        se_j = _se_block_j(sd, enc, i)
        w1 = sd[f"{pref}.{se_j}.fc1.weight"].numpy()
        b1 = sd[f"{pref}.{se_j}.fc1.bias"].numpy()
        w2 = sd[f"{pref}.{se_j}.fc2.weight"].numpy()
        b2 = sd[f"{pref}.{se_j}.fc2.bias"].numpy()
        t_se = workdir / f"{pk}_se.tif"
        ops.se_block(t_dw, t_se, w1=w1, b1=b1, w2=w2, b2=b2,
                     act1="relu", scale_activation="hsigmoid",
                     workdir=workdir, weights=weights, key=f"{pk}_se")
        if not keep_work:
            t_dw.unlink(missing_ok=True)
        cur2 = t_se
    else:
        cur2 = t_dw

    # ---- project 1x1 + BN (no activation) ----
    pj = _project_idx(sd, enc, i)
    w = sd[f"{pref}.{pj}.0.weight"].numpy()
    bn = _bn(sd, f"{pref}.{pj}.1")
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
    enc = _infer_enc_prefix(sd)
    N = _count_features(sd, enc)
    if N not in _CONFIGS:
        raise ValueError(f"mobilenetv3: unsupported features count {N}; "
                         f"supported: {sorted(_CONFIGS)}")
    cfg = _CONFIGS[N]

    feats: list[Path] = []
    pre = workdir / "00_pre.tif"
    ops.preprocess(input_tif, pre, in_channels=in_channels)
    feats.append(pre)

    # Stem: features.0 (3x3 stride 2 + BN + hswish)
    w = sd[f"{enc}features.0.0.weight"].numpy()
    bn = _bn(sd, f"{enc}features.0.1")
    stem = workdir / "mbv3_stem.tif"
    ops.conv(pre, stem, weight=w, bn=bn, activation="hswish",
             stride=2, padding=1, weights=weights, key="mbv3_stem")
    feats.append(stem)

    cur = stem
    captured: dict[int, Path] = {}
    nIR = N - 2  # IR block count
    for i in range(1, nIR + 1):
        stride, k, has_se, use_hs = cfg["ir"][i - 1]
        out = workdir / f"mbv3_f{i}.tif"
        _ir_block(cur, out, sd=sd, enc=enc, i=i, stride=stride, k=k,
                  has_se=has_se, use_hs=use_hs, workdir=workdir,
                  weights=weights, keep_work=keep_work)
        if i in cfg["capture"]:
            captured[i] = out
        if not keep_work and cur != stem and cur not in captured.values() and cur != out:
            Path(cur).unlink(missing_ok=True)
        cur = out

    # Tail: features.{N-1} (1x1 + BN + hswish)
    w = sd[f"{enc}features.{N-1}.0.weight"].numpy()
    bn = _bn(sd, f"{enc}features.{N-1}.1")
    tail = workdir / f"mbv3_f{N-1}.tif"
    ops.conv(cur, tail, weight=w, bn=bn, activation="hswish",
             stride=1, padding=0, weights=weights, key=f"mbv3_f{N-1}")
    if not keep_work and cur not in captured.values():
        Path(cur).unlink(missing_ok=True)

    for ci in cfg["capture"]:
        feats.append(captured[ci])
    feats.append(tail)
    return feats
