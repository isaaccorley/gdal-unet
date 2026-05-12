"""Generate per-layer VRTs for the chesapeakersc U-Net and execute them
with one `gdal_translate` per layer.

Goal: shrink the ~2185-subprocess pipeline in predict_gdal.py to ~30 by
expressing each conv layer (multi-input-channel conv + BN + ReLU fused)
as a single VRT-derived dataset, then materializing it once.

Strides and 2x upsamples and concats still happen via numpy here, because
VRT downsample/upsample resamples at pixel *centers*, not pixel corners,
so they don't match PyTorch's conv2d(stride=2) / Upsample(nearest) sampling
exactly.

Layers/conv counts:
  stem (4->64, 7x7, stride 2)             : 1 conv  +  pad/stride helpers
  maxpool 3x3 stride 2                    : 0 conv  (numpy)
  l1.b0  conv1 (64->64), conv2 (64->64)   : 2 convs
  l1.b1  conv1, conv2                     : 2 convs
  l2.b0  conv1 (64->128 stride 2), conv2  : 2 convs (+ ds 1x1 stride 2)
  l2.b1  conv1, conv2                     : 2 convs
  l3.b0  conv1 (128->256 stride 2), conv2 : 2 convs (+ ds)
  l3.b1  conv1, conv2                     : 2 convs
  l4.b0  conv1 (256->512 stride 2), conv2 : 2 convs (+ ds)
  l4.b1  conv1, conv2                     : 2 convs
  d0.c1, d0.c2                            : 2 convs   (in 768, out 256)
  d1.c1, d1.c2                            : 2 convs
  d2.c1, d2.c2                            : 2 convs
  d3.c1, d3.c2                            : 2 convs
  d4.c1, d4.c2                            : 2 convs (no skip)
  head (16->2)                            : 1 conv
Total conv layers: 1 + 16 + 10 + 1 = ~28 + 4 downsample 1x1 + ~5 add/relu
                   = ~30 layer-VRT materializations + a handful of numpy steps.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.windows import Window

ENV = "/projects/bgtj/isaaccorley/envs/ftw-tile"
os.environ["PATH"] = f"{ENV}/bin:" + os.environ.get("PATH", "")
os.environ["PROJ_DATA"] = f"{ENV}/share/proj"
os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"]

ROOT = Path(__file__).parent.resolve()
W = np.load(ROOT / "model_weights.npz")

NCALLS = 0
T_GDAL = 0.0


def sh(*cmd):
    global NCALLS, T_GDAL
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("CMD:", " ".join(cmd))
        print(e.stderr.decode(errors="replace"))
        raise
    NCALLS += 1
    T_GDAL += time.time() - t0


COFLAGS = ["-co", "BIGTIFF=IF_SAFER", "-co", "INTERLEAVE=BAND"]


# ----------------------------------------------------------------------------
# Raster IO helpers (numpy) used for pad / stride / upsample / concat where
# VRT semantics differ from PyTorch.
# ----------------------------------------------------------------------------
def read_tif(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        return src.read()


def write_tif(p: Path, arr: np.ndarray, dtype="float16"):
    if arr.ndim == 2:
        arr = arr[None]
    C, H, Wd = arr.shape
    prof = {"driver": "GTiff", "height": H, "width": Wd, "count": C,
            "dtype": dtype, "tiled": False, "interleave": "band",
            "bigtiff": "IF_SAFER"}
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(arr.astype(dtype))


def pad_zero(in_p: Path, out_p: Path, pad: int):
    """Zero-pad a TIFF by `pad` rows/cols on each side."""
    a = read_tif(in_p)
    out = np.pad(a, ((0, 0), (pad, pad), (pad, pad)))
    write_tif(out_p, out, dtype=a.dtype)


def stride2_subsample(in_p: Path, out_p: Path):
    a = read_tif(in_p)
    write_tif(out_p, a[:, ::2, ::2], dtype=a.dtype)


def upsample2x_nearest(in_p: Path, out_p: Path):
    a = read_tif(in_p)
    out = a.repeat(2, axis=1).repeat(2, axis=2)
    write_tif(out_p, out, dtype=a.dtype)


def concat_tifs(tifs, out_p: Path):
    arrs = [read_tif(t) for t in tifs]
    out = np.concatenate(arrs, axis=0)
    write_tif(out_p, out, dtype=arrs[0].dtype)


def maxpool3x3_s2(in_p: Path, out_p: Path):
    """3x3 maxpool stride 2 with padding 1, matching nn.MaxPool2d(3,2,1)."""
    a = read_tif(in_p).astype(np.float32)
    C, H, Wd = a.shape
    # zero-pad (post-relu activations are >=0 so zero pad equals -inf for max)
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    # full-res 3x3 max via sliding
    out_full = np.maximum.reduce([p[:, i:i+H, j:j+Wd]
                                  for i in range(3) for j in range(3)])
    out = out_full[:, ::2, ::2]
    write_tif(out_p, out, dtype="float16")


def add_relu(a_p: Path, b_p: Path, out_p: Path, relu: bool = True):
    a = read_tif(a_p).astype(np.float32)
    b = read_tif(b_p).astype(np.float32)
    out = a + b
    if relu:
        out = np.maximum(out, 0)
    write_tif(out_p, out, dtype="float16")


# ----------------------------------------------------------------------------
# VRT generation: a conv layer as a single VRT with Cout derived bands,
# each band being an `expression` pixel function over Cin KernelFilteredSources.
# ----------------------------------------------------------------------------
def kernel_node(coefs: np.ndarray, kH: int) -> str:
    # PyTorch conv2d == cross-correlation. GDAL's KernelFilteredSource is also
    # cross-correlation in image-space (kernel applied without flipping when
    # entered in row-major).
    # Coefs are entered row-major as the kernel is read in (kH, kW).
    flat = " ".join(f"{float(c):.6g}" for c in coefs.ravel())
    return (f'      <Kernel normalized="0">\n'
            f'        <Size>{kH}</Size>\n'
            f'        <Coefs>{flat}</Coefs>\n'
            f'      </Kernel>\n')


def conv_layer_vrt(src_tif: Path, src_W: int, src_H: int, src_C: int,
                   kernel: np.ndarray, *,
                   bn_a: np.ndarray | None = None,
                   bn_b: np.ndarray | None = None,
                   bias: np.ndarray | None = None,
                   relu: bool = True) -> str:
    """Build VRT XML representing one full-resolution conv layer.

    Layout:
      <VRTDataset W H>
        for each output channel oc:
          <VRTRasterBand subClass="VRTDerivedRasterBand" Float32>
            <PixelFunctionType>expression</PixelFunctionType>
            <PixelFunctionArguments expression="bn_a*(sum of B_i) + bn_b" />  (with relu max)
            <SourceTransferType>Float32</SourceTransferType>
            for each input channel ic:
              <KernelFilteredSource>
                source=src_tif band ic+1, kernel = kernel[oc, ic]
    """
    Cout, Cin, kH, kW = kernel.shape
    assert kH == kW
    assert Cin == src_C, (Cin, src_C)
    lines = [f'<VRTDataset rasterXSize="{src_W}" rasterYSize="{src_H}">']
    for oc in range(Cout):
        # build the expression
        sum_expr = " + ".join(f"B{ic+1}" for ic in range(Cin))
        expr = f"({sum_expr})"
        if bias is not None:
            expr = f"({expr} + ({float(bias[oc])}))"
        if bn_a is not None:
            expr = f"(({float(bn_a[oc])}) * {expr} + ({float(bn_b[oc])}))"
        if relu:
            expr = f"max(0, {expr})"
        expr_esc = escape(expr, {'"': '&quot;'})

        lines.append(
            f'  <VRTRasterBand dataType="Float32" band="{oc+1}" '
            f'subClass="VRTDerivedRasterBand">')
        lines.append(f'    <PixelFunctionType>expression</PixelFunctionType>')
        lines.append(
            f'    <PixelFunctionArguments expression="{expr_esc}"/>')
        lines.append(f'    <SourceTransferType>Float32</SourceTransferType>')
        for ic in range(Cin):
            lines.append(f'    <KernelFilteredSource>')
            lines.append(
                f'      <SourceFilename relativeToVRT="0">{src_tif}'
                f'</SourceFilename>')
            lines.append(f'      <SourceBand>{ic+1}</SourceBand>')
            lines.append(kernel_node(kernel[oc, ic], kH).rstrip())
            lines.append(f'    </KernelFilteredSource>')
        lines.append(f'  </VRTRasterBand>')
    lines.append('</VRTDataset>')
    return "\n".join(lines) + "\n"


def run_conv_layer(in_tif: Path, out_tif: Path, *, name: str,
                   kernel: np.ndarray,
                   bn_a=None, bn_b=None, bias=None, relu=True,
                   stride: int = 1, workdir: Path):
    """conv -> [BN] -> [+bias] -> [ReLU] -> [stride 2 subsample].

    Implementation: zero-pad the input (numpy), build a one-VRT conv that
    computes the full-resolution output, run one gdal_translate to
    materialize it as Float16 TIFF, then optionally subsample with numpy.
    """
    Cout, Cin, kH, kW = kernel.shape
    pad = kH // 2

    # 1. zero-pad input (numpy, single read+write — same I/O as predict_gdal)
    if pad > 0:
        padded = workdir / f"{name}_padded.tif"
        pad_zero(in_tif, padded, pad)
        conv_in = padded
    else:
        conv_in = in_tif

    with rasterio.open(conv_in) as s:
        W_, H_ = s.width, s.height
        nb = s.count
    assert nb == Cin, (name, nb, Cin)

    # 2. build the conv VRT (full-res, padded input)
    vrt_p = workdir / f"{name}.vrt"
    vrt_p.write_text(conv_layer_vrt(conv_in, W_, H_, Cin,
                                     kernel, bn_a=bn_a, bn_b=bn_b,
                                     bias=bias, relu=relu))

    # 3. materialize via single gdal_translate
    conv_out = workdir / f"{name}_full.tif"
    sh("gdal_translate", "-q", "-of", "GTiff",
       "-ot", "Float32",  # compute in Float32 for accuracy; cast on save
       *COFLAGS,
       str(vrt_p), str(conv_out))

    # 4. crop padded boundary (numpy), cast to Float16
    if pad > 0:
        a = read_tif(conv_out)
        # the padded-conv full-res output has shape (Cout, H_, W_); the
        # "valid" output (matching PyTorch's same-pad conv on H,W input) is
        # the inner H, W block at offset (pad, pad).
        a = a[:, pad:H_-pad, pad:W_-pad]
        cropped = workdir / f"{name}_cropped.tif"
        write_tif(cropped, a, dtype="float16")
        conv_out_final = cropped
    else:
        # cast Float32 -> Float16
        a = read_tif(conv_out)
        cast = workdir / f"{name}_f16.tif"
        write_tif(cast, a, dtype="float16")
        conv_out_final = cast

    # 5. stride-2 subsample if requested
    if stride == 2:
        stride2_subsample(conv_out_final, out_tif)
    else:
        shutil.move(str(conv_out_final), str(out_tif))


# ----------------------------------------------------------------------------
# Higher-level building blocks
# ----------------------------------------------------------------------------
def preprocess(in_tif: Path, out_tif: Path):
    """Normalize uint8 NAIP -> float16 / 255."""
    a = read_tif(in_tif).astype(np.float32) / 255.0
    write_tif(out_tif, a, dtype="float16")


def softmax_2cls(in_tif: Path, out_tif: Path):
    a = read_tif(in_tif).astype(np.float32)
    m = a.max(axis=0, keepdims=True)
    e = np.exp(a - m)
    p = e / e.sum(axis=0, keepdims=True)
    with rasterio.open(in_tif) as s:
        H, Wd = s.height, s.width
    prof = {"driver": "GTiff", "height": H, "width": Wd, "count": 2,
            "dtype": "float16", "tiled": True, "compress": "deflate",
            "blockxsize": 256, "blockysize": 256, "bigtiff": "IF_SAFER"}
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(p.astype("float16"))


def basic_block(in_tif: Path, out_tif: Path, *, layer, block, stride, workdir):
    p = f"l{layer}.b{block}"
    c1 = workdir / f"{p}_c1.tif"
    run_conv_layer(in_tif, c1, name=f"{p}_c1",
                   kernel=W[f"{p}.conv1.kernel"],
                   bn_a=W[f"{p}.conv1.bn_a"], bn_b=W[f"{p}.conv1.bn_b"],
                   relu=True, stride=stride, workdir=workdir)
    c2 = workdir / f"{p}_c2.tif"
    run_conv_layer(c1, c2, name=f"{p}_c2",
                   kernel=W[f"{p}.conv2.kernel"],
                   bn_a=W[f"{p}.conv2.bn_a"], bn_b=W[f"{p}.conv2.bn_b"],
                   relu=False, workdir=workdir)

    if block == 0 and layer > 1:
        skip = workdir / f"{p}_ds.tif"
        run_conv_layer(in_tif, skip, name=f"{p}_ds",
                       kernel=W[f"{p}.ds.kernel"],
                       bn_a=W[f"{p}.ds.bn_a"], bn_b=W[f"{p}.ds.bn_b"],
                       relu=False, stride=stride, workdir=workdir)
    else:
        skip = in_tif

    add_relu(c2, skip, out_tif, relu=True)


def decoder_block(in_tif: Path, skip_tif, out_tif: Path, *, idx, workdir):
    up = workdir / f"d{idx}_up.tif"
    upsample2x_nearest(in_tif, up)
    if skip_tif is not None:
        cat = workdir / f"d{idx}_cat.tif"
        concat_tifs([up, skip_tif], cat)
        layer_in = cat
    else:
        layer_in = up
    c1 = workdir / f"d{idx}_c1.tif"
    run_conv_layer(layer_in, c1, name=f"d{idx}_c1",
                   kernel=W[f"d{idx}.c1.kernel"],
                   bn_a=W[f"d{idx}.c1.bn_a"], bn_b=W[f"d{idx}.c1.bn_b"],
                   relu=True, workdir=workdir)
    run_conv_layer(c1, out_tif, name=f"d{idx}_c2",
                   kernel=W[f"d{idx}.c2.kernel"],
                   bn_a=W[f"d{idx}.c2.bn_a"], bn_b=W[f"d{idx}.c2.bn_b"],
                   relu=True, workdir=workdir)


# ----------------------------------------------------------------------------
# Network forward
# ----------------------------------------------------------------------------
def forward(in_tif: Path, out_tif: Path, *, workdir: Path):
    pre = workdir / "00_pre.tif"
    preprocess(in_tif, pre)

    s1 = workdir / "01_stem.tif"
    run_conv_layer(pre, s1, name="stem",
                   kernel=W["stem.kernel"],
                   bn_a=W["stem.bn_a"], bn_b=W["stem.bn_b"],
                   relu=True, stride=2, workdir=workdir)

    mp = workdir / "02_mp.tif"
    maxpool3x3_s2(s1, mp)
    l1a = workdir / "02a.tif"
    basic_block(mp, l1a, layer=1, block=0, stride=1, workdir=workdir)
    l1b = workdir / "02b.tif"
    basic_block(l1a, l1b, layer=1, block=1, stride=1, workdir=workdir)

    l2a = workdir / "03a.tif"
    basic_block(l1b, l2a, layer=2, block=0, stride=2, workdir=workdir)
    l2b = workdir / "03b.tif"
    basic_block(l2a, l2b, layer=2, block=1, stride=1, workdir=workdir)

    l3a = workdir / "04a.tif"
    basic_block(l2b, l3a, layer=3, block=0, stride=2, workdir=workdir)
    l3b = workdir / "04b.tif"
    basic_block(l3a, l3b, layer=3, block=1, stride=1, workdir=workdir)

    l4a = workdir / "05a.tif"
    basic_block(l3b, l4a, layer=4, block=0, stride=2, workdir=workdir)
    l4b = workdir / "05b.tif"
    basic_block(l4a, l4b, layer=4, block=1, stride=1, workdir=workdir)

    d0 = workdir / "d0.tif"
    decoder_block(l4b, l3b, d0, idx=0, workdir=workdir)
    d1 = workdir / "d1.tif"
    decoder_block(d0, l2b, d1, idx=1, workdir=workdir)
    d2 = workdir / "d2.tif"
    decoder_block(d1, l1b, d2, idx=2, workdir=workdir)
    d3 = workdir / "d3.tif"
    decoder_block(d2, s1, d3, idx=3, workdir=workdir)
    d4 = workdir / "d4.tif"
    decoder_block(d3, None, d4, idx=4, workdir=workdir)

    logits = workdir / "logits.tif"
    run_conv_layer(d4, logits, name="head",
                   kernel=W["head.kernel"], bias=W["head.bias"],
                   relu=False, workdir=workdir)

    softmax_2cls(logits, out_tif)
    return out_tif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--stop-after", default=None,
                    help="stop after this layer name (for debugging)")
    args = ap.parse_args()

    in_tif = Path(args.input).resolve()
    out_tif = Path(args.output or in_tif.with_suffix(".vrt_probs.tif")).resolve()
    base = ROOT if args.keep_work else Path(
        os.environ.get("SLURM_TMPDIR") or "/dev/shm")
    workdir = base / f"work_{int(time.time())}_{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[input]  {in_tif.name}")
    print(f"[output] {out_tif.name}")
    print(f"[work]   {workdir}")
    t0 = time.time()
    try:
        forward(in_tif, out_tif, workdir=workdir)
    finally:
        if not args.keep_work:
            shutil.rmtree(workdir, ignore_errors=True)
    dt = time.time() - t0
    print(f"\n[done] {dt:.1f}s  ({NCALLS} gdal subprocesses, "
          f"gdal_time = {T_GDAL:.1f}s)")


if __name__ == "__main__":
    main()
