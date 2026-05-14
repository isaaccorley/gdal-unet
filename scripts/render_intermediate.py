"""Render intermediate U-Net layers as visualization GeoTIFFs.

For each tif in intermediate_output/, produce:
  - <stage>_rgb.tif       PCA top-3 components -> uint8 RGB (RGBA, full alpha)
  - <stage>_ch<NN>.tif    top-8 highest-variance channels -> turbo heatmap RGBA

Special cases:
  - 00_pre.tif      4-band RGBN: ch00..03 are R,G,B,NIR; rgb is natural color
  - logits.tif      2-band: ch00 = bg, ch01 = road; both turbo
  - probs.tif       2-band: viridis P(road) only

Each output is upsampled (nearest) to the NAIP grid (4096x4096, 1m, NAIP CRS)
so all overlays align pixel-perfect on the map.

A sidecar <name>.json captures normalization metadata for the legend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from matplotlib import colormaps
from rasterio.warp import Resampling, reproject

TURBO = colormaps["turbo"]
VIRIDIS = colormaps["viridis"]

TOP_K_CHANNELS = 8
PCA_N_COMPONENTS = 3
ALPHA_OPAQUE = 220


def _pct_clip(x: np.ndarray, lo: float = 1.0, hi: float = 99.0
              ) -> tuple[np.ndarray, float, float]:
    """Clip array to (lo, hi) percentile, return clipped + (vmin, vmax)."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x), 0.0, 0.0
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip(x, vmin, vmax), float(vmin), float(vmax)


def _to_uint8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(vmax - vmin, 1e-6)
    return np.clip((x - vmin) / span * 255.0, 0, 255).astype(np.uint8)


def _colormap_to_rgba(x_u8: np.ndarray, cmap, alpha_u8: np.ndarray) -> np.ndarray:
    """Apply matplotlib cmap to a uint8 1-band array, return uint8 RGBA HxWx4."""
    rgba = (cmap(x_u8) * 255).astype(np.uint8)  # H,W,4 with alpha=255
    rgba[..., 3] = alpha_u8
    return rgba


def _resample_to_grid(src_data: np.ndarray, src_transform, src_crs,
                      dst_transform, dst_crs, dst_h: int, dst_w: int,
                      resampling=Resampling.nearest) -> np.ndarray:
    """Resample (C, H, W) to (C, dst_h, dst_w) on the destination grid."""
    out = np.zeros((src_data.shape[0], dst_h, dst_w), dtype=src_data.dtype)
    for c in range(src_data.shape[0]):
        reproject(
            source=src_data[c],
            destination=out[c],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return out


def _write_rgba(path: Path, rgba: np.ndarray, transform, crs):
    """Write HxWx4 uint8 RGBA as a GeoTIFF."""
    h, w, _ = rgba.shape
    profile = dict(
        driver="GTiff",
        height=h, width=w, count=4, dtype="uint8",
        transform=transform, crs=crs,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
        photometric="RGB", alpha="unspecified",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(4):
            dst.write(rgba[..., i], i + 1)
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]


def _pca_rgb_lowres(stack_native: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """PCA across channels on the NATIVE (low-res) array.
    Returns (RGB at native res H,W,3 uint8, alpha at native res H,W uint8,
    vmins[3], vmaxs[3]).  Caller upsamples the result.

    Speed strategy:
      - subsample for the covariance fit when N > 200k pixels
      - eigh on C×C covariance instead of SVD on N×C
      - fit vectors are then applied to all native-res pixels (cheap projection)
    """
    C, H, W = stack_native.shape
    if C < 1:
        return (np.zeros((H, W, 3), np.uint8),
                np.full((H, W), ALPHA_OPAQUE, np.uint8),
                [0,0,0], [0,0,0])

    flat = stack_native.reshape(C, -1).T  # (N, C) float32 view
    finite_mask = np.isfinite(flat).all(axis=1)
    if finite_mask.sum() < 4:
        return (np.zeros((H, W, 3), np.uint8),
                np.full((H, W), ALPHA_OPAQUE, np.uint8),
                [0,0,0], [0,0,0])

    finite_idx = np.flatnonzero(finite_mask)
    sample_max = 200_000
    if finite_idx.size > sample_max:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(finite_idx, sample_max, replace=False)
    else:
        sample_idx = finite_idx
    sample = flat[sample_idx]                       # (Ns, C)

    mean = sample.mean(axis=0, keepdims=True)       # (1, C)
    Xs = sample - mean                              # (Ns, C)
    cov = (Xs.T @ Xs) / max(1, Xs.shape[0] - 1)     # (C, C)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)      # ascending
    except np.linalg.LinAlgError:
        return (np.zeros((H, W, 3), np.uint8),
                np.full((H, W), ALPHA_OPAQUE, np.uint8),
                [0,0,0], [0,0,0])
    order = np.argsort(eigvals)[::-1]
    n_comp = min(PCA_N_COMPONENTS, C)
    comps = eigvecs[:, order[:n_comp]].T            # (n_comp, C)

    # Project ALL native pixels (cheap: N * C * 3 ops, e.g. 16M*64*3 = 3 GFLOPs)
    centered = flat - mean                          # (N, C)
    proj = centered @ comps.T                       # (N, n_comp)

    bands = [proj[:, i].reshape(H, W) for i in range(n_comp)]
    while len(bands) < 3:
        bands.append(np.zeros((H, W), dtype=np.float32))

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vmins, vmaxs = [], []
    for i, b in enumerate(bands[:3]):
        clipped, vmin, vmax = _pct_clip(b)
        rgb[..., i] = _to_uint8(clipped, vmin, vmax)
        vmins.append(vmin); vmaxs.append(vmax)

    alpha = np.full((H, W), ALPHA_OPAQUE, dtype=np.uint8)
    zeros = (stack_native == 0).all(axis=0)
    alpha[zeros] = 0
    return rgb, alpha, vmins, vmaxs


def _upsample_rgba(rgba_native: np.ndarray,
                   src_transform, src_crs,
                   dst_transform, dst_crs, dst_h: int, dst_w: int
                   ) -> np.ndarray:
    """Upsample HxWx4 uint8 RGBA to dst grid (nearest)."""
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    for i in range(4):
        reproject(
            source=rgba_native[..., i],
            destination=out[..., i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return out


def _channel_indices_by_variance(stack: np.ndarray, k: int) -> list[int]:
    C = stack.shape[0]
    if C <= k:
        return list(range(C))
    var = stack.reshape(C, -1).var(axis=1)
    return list(np.argsort(-var)[:k].astype(int))


def _channel_label(stage: str, idx: int) -> str:
    if stage == "00_pre":
        return ["R", "G", "B", "NIR"][idx] if idx < 4 else f"ch{idx}"
    if stage in ("logits", "probs"):
        return ["bg", "road"][idx] if idx < 2 else f"ch{idx}"
    return f"ch{idx}"


def render_stage(in_tif: Path, out_dir: Path,
                 dst_transform, dst_crs, dst_h: int, dst_w: int) -> dict:
    stage = in_tif.stem
    print(f"[render] {stage:18s}", end=" ", flush=True)
    with rasterio.open(in_tif) as src:
        data = src.read().astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
    if src_crs is None:
        src_crs = dst_crs

    C = data.shape[0]
    H_n, W_n = data.shape[1], data.shape[2]
    print(f"C={C:3d}  shape_native={H_n}x{W_n}", flush=True)

    meta = {
        "stage": stage,
        "channels": int(C),
        "shape_native": [int(H_n), int(W_n)],
        "shape_rendered": [int(dst_h), int(dst_w)],
        "views": {},
    }

    # ---- RGB / PCA-RGB / natural color (work at NATIVE res, then upsample) ----
    if stage == "00_pre":
        # natural-color RGB from R,G,B at native res (already 4096x4096 here)
        rgb_u8 = np.zeros((H_n, W_n, 3), dtype=np.uint8)
        vmins, vmaxs = [], []
        for i in range(3):
            clipped, vmin, vmax = _pct_clip(data[i], 1.0, 99.0)
            rgb_u8[..., i] = _to_uint8(clipped, vmin, vmax)
            vmins.append(vmin); vmaxs.append(vmax)
        alpha = np.full((H_n, W_n), ALPHA_OPAQUE, dtype=np.uint8)
        rgba_native = np.dstack([rgb_u8, alpha])
        rgba = _upsample_rgba(rgba_native, src_transform, src_crs,
                              dst_transform, dst_crs, dst_h, dst_w)
        _write_rgba(out_dir / f"{stage}_rgb.tif", rgba, dst_transform, dst_crs)
        meta["views"]["rgb"] = {
            "kind": "rgb", "label": "Natural color (RGB)",
            "bands": ["R", "G", "B"],
            "vmins": vmins, "vmaxs": vmaxs,
        }
    elif stage in ("logits", "probs"):
        pass  # no PCA / RGB view for 2-band outputs
    else:
        rgb_u8, alpha_u8, vmins, vmaxs = _pca_rgb_lowres(data)
        rgba_native = np.dstack([rgb_u8, alpha_u8])
        rgba = _upsample_rgba(rgba_native, src_transform, src_crs,
                              dst_transform, dst_crs, dst_h, dst_w)
        _write_rgba(out_dir / f"{stage}_rgb.tif", rgba, dst_transform, dst_crs)
        meta["views"]["rgb"] = {
            "kind": "pca_rgb", "label": "PCA false color (RGB)",
            "bands": ["PC1", "PC2", "PC3"],
            "vmins": vmins, "vmaxs": vmaxs,
        }

    # ---- Per-channel heatmaps (also work at native res, upsample at end) ----
    if stage == "probs":
        idxs = [1]
        cmap = VIRIDIS
        cmap_name = "viridis"
    else:
        idxs = _channel_indices_by_variance(data, TOP_K_CHANNELS)
        cmap = TURBO
        cmap_name = "turbo"

    for rank, ch in enumerate(idxs):
        band = data[ch]
        clipped, vmin, vmax = _pct_clip(band, 1.0, 99.0)
        u8 = _to_uint8(clipped, vmin, vmax)
        alpha = np.where(band == 0, 0, ALPHA_OPAQUE).astype(np.uint8)
        rgba_native = _colormap_to_rgba(u8, cmap, alpha)
        rgba = _upsample_rgba(rgba_native, src_transform, src_crs,
                              dst_transform, dst_crs, dst_h, dst_w)
        view_id = f"ch{ch:02d}"
        out = out_dir / f"{stage}_{view_id}.tif"
        _write_rgba(out, rgba, dst_transform, dst_crs)
        meta["views"][view_id] = {
            "kind": "channel",
            "label": f"Ch {ch} ({_channel_label(stage, ch)})",
            "channel_index": int(ch),
            "rank_by_variance": int(rank),
            "colormap": cmap_name,
            "vmin": vmin, "vmax": vmax,
        }

    sidecar = out_dir / f"{stage}.json"
    sidecar.write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("intermediate_output"))
    ap.add_argument("--output-dir", type=Path, default=Path("intermediate_visuals"))
    ap.add_argument("--reference", type=Path, default=Path("naip_md_4096.tif"),
                    help="Reference raster providing CRS+transform for output grid.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing per-stage outputs.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional subset of stage names (e.g. 00_pre 01_stem probs).")
    args = ap.parse_args()

    if not args.reference.exists():
        print(f"missing reference {args.reference}", file=sys.stderr); return 2

    with rasterio.open(args.reference) as ref:
        dst_transform = ref.transform
        dst_crs = ref.crs
        dst_h, dst_w = ref.height, ref.width
    print(f"[render] reference grid {dst_h}x{dst_w}@{abs(dst_transform.a):.2f}m  crs={dst_crs}",
          flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(args.input_dir.glob("*.tif"))
    if args.only:
        tifs = [t for t in tifs if t.stem in set(args.only)]

    for t in tifs:
        sidecar = args.output_dir / f"{t.stem}.json"
        if sidecar.exists() and not args.force:
            print(f"[render] {t.stem:18s}  [skip] sidecar exists", flush=True)
            continue
        render_stage(t, args.output_dir, dst_transform, dst_crs, dst_h, dst_w)

    print(f"[render] done. {len(list(args.output_dir.glob('*.tif')))} tifs in {args.output_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
