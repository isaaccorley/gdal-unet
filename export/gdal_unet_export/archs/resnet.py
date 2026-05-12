"""smp.Unet(resnet*) state_dict -> Float16 .bin files + shapes.txt manifest.

BN is folded at export time (a = w/sqrt(v+eps); b = b - m*a) so the shell
predict script just needs --bn-a / --bn-b paths.  shapes.txt holds the
--kernel-shape strings keyed by layer name.
"""
import re
from pathlib import Path

import numpy as np


def _bn_fold(weight, bn_w, bn_b, bn_m, bn_v, eps=1e-5):
    a = bn_w / np.sqrt(bn_v + eps)
    b = bn_b - bn_m * a
    return a.astype(np.float16), b.astype(np.float16)


def _save(arr: np.ndarray, path: Path):
    np.ascontiguousarray(arr.astype(np.float16)).tofile(path)


def _bn_tuple(sd: dict, prefix: str):
    return (
        sd[f"{prefix}.weight"].numpy(),
        sd[f"{prefix}.bias"].numpy(),
        sd[f"{prefix}.running_mean"].numpy(),
        sd[f"{prefix}.running_var"].numpy(),
        1e-5,
    )


def _dump_conv_bn(sd: dict, out: Path, key: str, conv_key: str, bn_prefix: str | None,
                  shapes: list[str]):
    w = sd[conv_key].numpy()
    _save(w, out / f"{key}.kernel.bin")
    shapes.append(f"{key} {w.shape[0]},{w.shape[1]},{w.shape[2]},{w.shape[3]}")
    if bn_prefix is not None:
        bn = _bn_tuple(sd, bn_prefix)
        a, b = _bn_fold(w, *bn)
        _save(a, out / f"{key}.bn_a.bin")
        _save(b, out / f"{key}.bn_b.bin")


def _num_blocks(sd: dict) -> dict[int, int]:
    pat = re.compile(r"^encoder\.layer(\d+)\.(\d+)\.")
    nb: dict[int, int] = {}
    for k in sd:
        m = pat.match(k)
        if m:
            L = int(m.group(1)); B = int(m.group(2))
            nb[L] = max(nb.get(L, 0), B + 1)
    return nb


def _is_bottleneck(sd: dict) -> bool:
    return any(k == "encoder.layer1.0.conv3.weight" for k in sd)


def export(sd: dict, out: Path, *, arch: str):
    shapes: list[str] = []
    _dump_conv_bn(sd, out, "stem", "encoder.conv1.weight", "encoder.bn1", shapes)

    bottleneck = _is_bottleneck(sd)
    nb = _num_blocks(sd)
    for L in sorted(nb):
        for b in range(nb[L]):
            p = f"encoder.layer{L}.{b}"
            pk = f"l{L}b{b}"
            _dump_conv_bn(sd, out, f"{pk}_c1", f"{p}.conv1.weight", f"{p}.bn1", shapes)
            _dump_conv_bn(sd, out, f"{pk}_c2", f"{p}.conv2.weight", f"{p}.bn2", shapes)
            if bottleneck:
                _dump_conv_bn(sd, out, f"{pk}_c3", f"{p}.conv3.weight", f"{p}.bn3", shapes)
            if f"{p}.downsample.0.weight" in sd:
                _dump_conv_bn(sd, out, f"{pk}_ds",
                              f"{p}.downsample.0.weight",
                              f"{p}.downsample.1", shapes)

    n_dec = max(int(k.split(".")[2]) for k in sd if k.startswith("decoder.blocks.")) + 1
    for i in range(n_dec):
        _dump_conv_bn(sd, out, f"dec{i}_c1",
                      f"decoder.blocks.{i}.conv1.0.weight",
                      f"decoder.blocks.{i}.conv1.1", shapes)
        _dump_conv_bn(sd, out, f"dec{i}_c2",
                      f"decoder.blocks.{i}.conv2.0.weight",
                      f"decoder.blocks.{i}.conv2.1", shapes)

    hw = sd["segmentation_head.0.weight"].numpy()
    hb = sd["segmentation_head.0.bias"].numpy()
    _save(hw, out / "head.kernel.bin")
    _save(hb, out / "head.bias.bin")
    shapes.append(f"head {hw.shape[0]},{hw.shape[1]},{hw.shape[2]},{hw.shape[3]}")

    (out / "shapes.txt").write_text("\n".join(shapes) + "\n")
    (out / "n_decoder_blocks.txt").write_text(f"{n_dec}\n")
    (out / "num_blocks.txt").write_text(
        "\n".join(f"{L} {nb[L]}" for L in sorted(nb)) + "\n"
    )
