"""smp.Unet(resnet18) forward pass driven by `gdal-unet-conv` subprocesses.

Ported verbatim from the original `predict_cpp.py`; identical math
(per-stage cosine 1.0000 vs the PyTorch reference, IoU 0.6372 on
sample 1717, vs 0.6374 from PyTorch).
"""

from pathlib import Path

from ..conv_runner import ConvRunner
from ..ops import (
    WeightDir,
    add_relu,
    concat,
    maxpool3x3_s2,
    preprocess,
    softmax_2cls,
    upsample2x,
)


def _basic_block(runner: ConvRunner, in_tif: Path, out_tif: Path, *,
                 layer: int, block: int, stride: int,
                 workdir: Path, weights: WeightDir, keep_work: bool):
    p = f"l{layer}.b{block}"
    c1 = workdir / f"{p}_c1.tif"
    runner.conv(in_tif, c1,
                kernel_key=f"{p}.conv1.kernel",
                bn_a_key=f"{p}.conv1.bn_a", bn_b_key=f"{p}.conv1.bn_b",
                relu=True, stride=stride, weights=weights)
    c2 = workdir / f"{p}_c2.tif"
    runner.conv(c1, c2,
                kernel_key=f"{p}.conv2.kernel",
                bn_a_key=f"{p}.conv2.bn_a", bn_b_key=f"{p}.conv2.bn_b",
                relu=False, weights=weights)
    if not keep_work:
        c1.unlink(missing_ok=True)

    if block == 0 and layer > 1:
        skip = workdir / f"{p}_ds.tif"
        runner.conv(in_tif, skip,
                    kernel_key=f"{p}.ds.kernel",
                    bn_a_key=f"{p}.ds.bn_a", bn_b_key=f"{p}.ds.bn_b",
                    relu=False, stride=stride, weights=weights)
    else:
        skip = in_tif
    add_relu(c2, skip, out_tif, relu=True)
    if not keep_work:
        c2.unlink(missing_ok=True)
        if skip != in_tif:
            skip.unlink(missing_ok=True)


def _decoder_block(runner: ConvRunner, in_tif: Path, skip_tif: Path | None,
                   out_tif: Path, *, idx: int, workdir: Path,
                   weights: WeightDir, keep_work: bool):
    up = workdir / f"d{idx}_up.tif"
    upsample2x(in_tif, up)
    if skip_tif is not None:
        cat = workdir / f"d{idx}_cat.tif"
        concat([up, skip_tif], cat)
        if not keep_work:
            up.unlink(missing_ok=True)
        layer_in = cat
    else:
        layer_in = up
    c1 = workdir / f"d{idx}_c1.tif"
    runner.conv(layer_in, c1,
                kernel_key=f"d{idx}.c1.kernel",
                bn_a_key=f"d{idx}.c1.bn_a", bn_b_key=f"d{idx}.c1.bn_b",
                relu=True, weights=weights)
    if not keep_work:
        layer_in.unlink(missing_ok=True)
    runner.conv(c1, out_tif,
                kernel_key=f"d{idx}.c2.kernel",
                bn_a_key=f"d{idx}.c2.bn_a", bn_b_key=f"d{idx}.c2.bn_b",
                relu=True, weights=weights)
    if not keep_work:
        c1.unlink(missing_ok=True)


def forward(in_tif: Path, out_tif: Path, *, workdir: Path,
            weights_npz, runner: ConvRunner, keep_work: bool = False):
    """Full forward pass: input -> 2-band Float16 probability raster."""
    weights = WeightDir(workdir / "weights", weights_npz)

    pre = workdir / "00_pre.tif"
    preprocess(in_tif, pre)

    s1 = workdir / "01_stem.tif"
    runner.conv(pre, s1,
                kernel_key="stem.kernel",
                bn_a_key="stem.bn_a", bn_b_key="stem.bn_b",
                relu=True, stride=2, padding=3, weights=weights)
    if not keep_work:
        pre.unlink(missing_ok=True)

    mp = workdir / "02_mp.tif"
    maxpool3x3_s2(s1, mp)

    l1a = workdir / "02a.tif"
    _basic_block(runner, mp, l1a, layer=1, block=0, stride=1,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    if not keep_work:
        mp.unlink(missing_ok=True)
    l1b = workdir / "02b.tif"
    _basic_block(runner, l1a, l1b, layer=1, block=1, stride=1,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    if not keep_work:
        l1a.unlink(missing_ok=True)

    l2a = workdir / "03a.tif"
    _basic_block(runner, l1b, l2a, layer=2, block=0, stride=2,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    l2b = workdir / "03b.tif"
    _basic_block(runner, l2a, l2b, layer=2, block=1, stride=1,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    if not keep_work:
        l2a.unlink(missing_ok=True)

    l3a = workdir / "04a.tif"
    _basic_block(runner, l2b, l3a, layer=3, block=0, stride=2,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    l3b = workdir / "04b.tif"
    _basic_block(runner, l3a, l3b, layer=3, block=1, stride=1,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    if not keep_work:
        l3a.unlink(missing_ok=True)

    l4a = workdir / "05a.tif"
    _basic_block(runner, l3b, l4a, layer=4, block=0, stride=2,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    l4b = workdir / "05b.tif"
    _basic_block(runner, l4a, l4b, layer=4, block=1, stride=1,
                 workdir=workdir, weights=weights, keep_work=keep_work)
    if not keep_work:
        l4a.unlink(missing_ok=True)

    d0 = workdir / "d0.tif"
    _decoder_block(runner, l4b, l3b, d0, idx=0, workdir=workdir,
                   weights=weights, keep_work=keep_work)
    if not keep_work:
        l4b.unlink(missing_ok=True)
        l3b.unlink(missing_ok=True)
    d1 = workdir / "d1.tif"
    _decoder_block(runner, d0, l2b, d1, idx=1, workdir=workdir,
                   weights=weights, keep_work=keep_work)
    if not keep_work:
        d0.unlink(missing_ok=True)
        l2b.unlink(missing_ok=True)
    d2 = workdir / "d2.tif"
    _decoder_block(runner, d1, l1b, d2, idx=2, workdir=workdir,
                   weights=weights, keep_work=keep_work)
    if not keep_work:
        d1.unlink(missing_ok=True)
        l1b.unlink(missing_ok=True)
    d3 = workdir / "d3.tif"
    _decoder_block(runner, d2, s1, d3, idx=3, workdir=workdir,
                   weights=weights, keep_work=keep_work)
    if not keep_work:
        d2.unlink(missing_ok=True)
        s1.unlink(missing_ok=True)
    d4 = workdir / "d4.tif"
    _decoder_block(runner, d3, None, d4, idx=4, workdir=workdir,
                   weights=weights, keep_work=keep_work)
    if not keep_work:
        d3.unlink(missing_ok=True)

    logits = workdir / "logits.tif"
    runner.conv(d4, logits,
                kernel_key="head.kernel", bias_key="head.bias",
                relu=False, padding=1, weights=weights)
    if not keep_work:
        d4.unlink(missing_ok=True)

    softmax_2cls(logits, out_tif)
    if not keep_work:
        logits.unlink(missing_ok=True)
