"""ResNet encoder forward pass (BasicBlock for r18/r34, Bottleneck for r50+).

Detects block type from state_dict (presence of ``layer1.0.conv3.weight``).
Detects number of blocks per stage by scanning the state_dict.
"""

from pathlib import Path
import re

from .. import ops


def _bn(sd: dict, prefix: str) -> tuple:
    return (
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
        1e-5,
    )


def _is_bottleneck(sd: dict) -> bool:
    return any(k == "encoder.layer1.0.conv3.weight" for k in sd)


def _num_blocks(sd: dict) -> dict[int, int]:
    pat = re.compile(r"^encoder\.layer(\d+)\.(\d+)\.")
    nb: dict[int, int] = {}
    for k in sd:
        m = pat.match(k)
        if m:
            L = int(m.group(1)); B = int(m.group(2))
            nb[L] = max(nb.get(L, 0), B + 1)
    return nb


def _basic_block(in_tif, out_tif, *, layer, block, stride, sd, workdir, weights,
                 keep_work=False):
    p = f"encoder.layer{layer}.{block}"
    pk = f"l{layer}b{block}"

    w1 = sd[f"{p}.conv1.weight"].numpy()
    bn1 = _bn(sd, f"{p}.bn1")
    c1 = workdir / f"{pk}_c1.tif"
    ops.conv(in_tif, c1, weight=w1, bn=bn1, activation="relu",
             stride=stride, weights=weights, key=f"{pk}_c1")

    w2 = sd[f"{p}.conv2.weight"].numpy()
    bn2 = _bn(sd, f"{p}.bn2")
    c2 = workdir / f"{pk}_c2.tif"
    ops.conv(c1, c2, weight=w2, bn=bn2, activation="none",
             weights=weights, key=f"{pk}_c2")
    if not keep_work:
        c1.unlink(missing_ok=True)

    # residual branch
    has_ds = f"{p}.downsample.0.weight" in sd
    if has_ds:
        wd = sd[f"{p}.downsample.0.weight"].numpy()
        bnd = _bn(sd, f"{p}.downsample.1")
        skip = workdir / f"{pk}_ds.tif"
        ops.conv(in_tif, skip, weight=wd, bn=bnd, activation="none",
                 stride=stride, weights=weights, key=f"{pk}_ds")
    else:
        skip = in_tif

    ops.add(c2, skip, out_tif, activation="relu")
    if not keep_work:
        c2.unlink(missing_ok=True)
        if has_ds:
            Path(skip).unlink(missing_ok=True)


def _bottleneck(in_tif, out_tif, *, layer, block, stride, sd, workdir, weights,
                keep_work=False):
    """ResNet Bottleneck (resnet50/101/152).

    Three convs: 1x1 reduce -> 3x3 (stride here) -> 1x1 expand.  Residual via
    optional 1x1 downsample with the same overall stride.

    NOTE: PyTorch's torchvision uses the v1.5 ("stride at 3x3") variant, which
    is what smp inherits.
    """
    p = f"encoder.layer{layer}.{block}"
    pk = f"l{layer}b{block}"

    w1 = sd[f"{p}.conv1.weight"].numpy()
    bn1 = _bn(sd, f"{p}.bn1")
    c1 = workdir / f"{pk}_c1.tif"
    ops.conv(in_tif, c1, weight=w1, bn=bn1, activation="relu",
             stride=1, padding=0, weights=weights, key=f"{pk}_c1")

    w2 = sd[f"{p}.conv2.weight"].numpy()
    bn2 = _bn(sd, f"{p}.bn2")
    c2 = workdir / f"{pk}_c2.tif"
    ops.conv(c1, c2, weight=w2, bn=bn2, activation="relu",
             stride=stride, weights=weights, key=f"{pk}_c2")
    if not keep_work:
        c1.unlink(missing_ok=True)

    w3 = sd[f"{p}.conv3.weight"].numpy()
    bn3 = _bn(sd, f"{p}.bn3")
    c3 = workdir / f"{pk}_c3.tif"
    ops.conv(c2, c3, weight=w3, bn=bn3, activation="none",
             stride=1, padding=0, weights=weights, key=f"{pk}_c3")
    if not keep_work:
        c2.unlink(missing_ok=True)

    has_ds = f"{p}.downsample.0.weight" in sd
    if has_ds:
        wd = sd[f"{p}.downsample.0.weight"].numpy()
        bnd = _bn(sd, f"{p}.downsample.1")
        skip = workdir / f"{pk}_ds.tif"
        ops.conv(in_tif, skip, weight=wd, bn=bnd, activation="none",
                 stride=stride, padding=0, weights=weights, key=f"{pk}_ds")
    else:
        skip = in_tif

    ops.add(c3, skip, out_tif, activation="relu")
    if not keep_work:
        c3.unlink(missing_ok=True)
        if has_ds:
            Path(skip).unlink(missing_ok=True)


def forward(
    input_tif: Path,
    sd: dict,
    workdir: Path,
    weights: ops.WeightDir,
    *,
    in_channels: int = 4,
    keep_work: bool = False,
) -> list[Path]:
    """Run the ResNet encoder; return skip features at strides 1, 2, 4, 8, 16, 32."""
    feats: list[Path] = []

    # Preprocess: /255 -> Float16
    pre = workdir / "00_pre.tif"
    ops.preprocess(input_tif, pre, in_channels=in_channels)
    feats.append(pre)

    # Stem: conv1 (7x7 stride 2 pad 3) + bn1 + relu
    w = sd["encoder.conv1.weight"].numpy()
    bn = _bn(sd, "encoder.bn1")
    stem = workdir / "01_stem.tif"
    ops.conv(pre, stem, weight=w, bn=bn, activation="relu",
             stride=2, padding=3, weights=weights, key="stem")
    feats.append(stem)

    # MaxPool 3x3 stride 2 pad 1
    mp = workdir / "02_mp.tif"
    ops.maxpool3x3_s2(stem, mp)

    bottleneck = _is_bottleneck(sd)
    nb = _num_blocks(sd)
    blk_fn = _bottleneck if bottleneck else _basic_block

    # Layer 1: stride 1 throughout
    cur = mp
    for b in range(nb[1]):
        out = workdir / f"l1b{b}.tif"
        blk_fn(cur, out, layer=1, block=b, stride=1,
               sd=sd, workdir=workdir, weights=weights, keep_work=keep_work)
        if not keep_work and cur != mp:
            Path(cur).unlink(missing_ok=True)
        cur = out
    if not keep_work:
        mp.unlink(missing_ok=True)
    feats.append(cur)  # stride 4

    # Layers 2..4: first block strides 2, rest stride 1
    for L in (2, 3, 4):
        for b in range(nb[L]):
            stride = 2 if b == 0 else 1
            out = workdir / f"l{L}b{b}.tif"
            blk_fn(cur, out, layer=L, block=b, stride=stride,
                   sd=sd, workdir=workdir, weights=weights, keep_work=keep_work)
            if not keep_work and cur not in feats:
                Path(cur).unlink(missing_ok=True)
            cur = out
        feats.append(cur)

    return feats
