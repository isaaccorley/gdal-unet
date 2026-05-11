"""Extract chesapeakersc U-Net-ResNet18 weights from the .pt checkpoint
into a flat .npz that the GDAL-CLI orchestrator can load.

All BatchNorm layers are pre-fused into per-channel affine (a, b) so that
inference becomes:  max(0, a*conv_out + b).
"""

from pathlib import Path
import numpy as np
import torch

CKPT = Path(__file__).parent / "unet-resnet18.pt"
OUT = Path(__file__).parent / "model_weights.npz"

sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)


def bn_affine(prefix: str, eps: float = 1e-5):
    # BN folding done in fp32 for numerical stability; cast result to fp16.
    g = sd[f"{prefix}.weight"].numpy().astype(np.float16)
    b = sd[f"{prefix}.bias"].numpy().astype(np.float16)
    m = sd[f"{prefix}.running_mean"].numpy().astype(np.float16)
    v = sd[f"{prefix}.running_var"].numpy().astype(np.float16)
    a = g / np.sqrt(v + eps)
    return a.astype(np.float16), (b - m * a).astype(np.float16)


out: dict[str, np.ndarray] = {}


def add_conv_bn(out_prefix: str, conv_key: str, bn_prefix: str):
    out[f"{out_prefix}.kernel"] = sd[conv_key].numpy().astype(np.float16)
    a, b = bn_affine(bn_prefix)
    out[f"{out_prefix}.bn_a"] = a
    out[f"{out_prefix}.bn_b"] = b


# stem
add_conv_bn("stem", "encoder.conv1.weight", "encoder.bn1")

# layers 1..4, 2 blocks each
for L in range(1, 5):
    for B in range(2):
        pin = f"encoder.layer{L}.{B}"
        pout = f"l{L}.b{B}"
        add_conv_bn(f"{pout}.conv1", f"{pin}.conv1.weight", f"{pin}.bn1")
        add_conv_bn(f"{pout}.conv2", f"{pin}.conv2.weight", f"{pin}.bn2")
        if B == 0 and L > 1:
            add_conv_bn(f"{pout}.ds", f"{pin}.downsample.0.weight", f"{pin}.downsample.1")

# decoder blocks 0..4
for i in range(5):
    for cn in (1, 2):
        pin = f"decoder.blocks.{i}.conv{cn}"
        pout = f"d{i}.c{cn}"
        add_conv_bn(pout, f"{pin}.0.weight", f"{pin}.1")

# segmentation head (Conv + bias, no BN, no ReLU)
out["head.kernel"] = sd["segmentation_head.0.weight"].numpy().astype(np.float16)
out["head.bias"] = sd["segmentation_head.0.bias"].numpy().astype(np.float16)

np.savez(OUT, **out)
print(f"saved {len(out)} arrays  ({OUT.stat().st_size if OUT.exists() else 0:,} bytes)")
# print sanity check
for k in list(out)[:6]:
    print(f"  {k:25s} {out[k].shape}  {out[k].dtype}")
print("  ...")
print(f"weight tensor count: {sum(1 for k in out if k.endswith('.kernel'))}")
print(f"total numel: {sum(v.size for v in out.values()):,}")
