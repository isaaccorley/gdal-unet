"""Build web/layers.json from intermediate_visuals/ sidecars + tile dirs.

Groups layers into Input / Encoder L1-L4 / Decoder 0-4 / Output sections.
Each layer has one or more `views` (rgb / chXX) with metadata for the legend.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


GROUPS = [
    {
        "id": "input",
        "name": "Input",
        "subtitle": "0–2 · stem & maxpool",
        "stages": ["00_pre", "01_stem", "02_mp"],
        "open": True,
    },
    {
        "id": "encoder1",
        "name": "Encoder L1",
        "subtitle": "1/4 res · 64 ch",
        "stages": ["l1b0_c1", "l1b0_c2", "l1b0", "l1b1_c1", "l1b1_c2", "l1b1"],
        "open": False,
    },
    {
        "id": "encoder2",
        "name": "Encoder L2",
        "subtitle": "1/8 res · 128 ch",
        "stages": ["l2b0_c1", "l2b0_c2", "l2b0_ds", "l2b0",
                   "l2b1_c1", "l2b1_c2", "l2b1"],
        "open": False,
    },
    {
        "id": "encoder3",
        "name": "Encoder L3",
        "subtitle": "1/16 res · 256 ch",
        "stages": ["l3b0_c1", "l3b0_c2", "l3b0_ds", "l3b0",
                   "l3b1_c1", "l3b1_c2", "l3b1"],
        "open": False,
    },
    {
        "id": "encoder4",
        "name": "Encoder L4",
        "subtitle": "1/32 res · 512 ch",
        "stages": ["l4b0_c1", "l4b0_c2", "l4b0_ds", "l4b0",
                   "l4b1_c1", "l4b1_c2", "l4b1"],
        "open": False,
    },
    {
        "id": "decoder0",
        "name": "Decoder 0",
        "subtitle": "1/16 res · 256 ch",
        "stages": ["dec_0_up", "dec_0_cat", "dec_0_c1", "dec_0"],
        "open": False,
    },
    {
        "id": "decoder1",
        "name": "Decoder 1",
        "subtitle": "1/8 res · 128 ch",
        "stages": ["dec_1_up", "dec_1_cat", "dec_1_c1", "dec_1"],
        "open": False,
    },
    {
        "id": "decoder2",
        "name": "Decoder 2",
        "subtitle": "1/4 res · 64 ch",
        "stages": ["dec_2_up", "dec_2_cat", "dec_2_c1", "dec_2"],
        "open": False,
    },
    {
        "id": "decoder3",
        "name": "Decoder 3",
        "subtitle": "1/2 res · 32 ch",
        "stages": ["dec_3_up", "dec_3_cat", "dec_3_c1", "dec_3"],
        "open": False,
    },
    {
        "id": "decoder4",
        "name": "Decoder 4",
        "subtitle": "full res · 16 ch",
        "stages": ["dec_4_up", "dec_4_c1", "dec_4"],
        "open": False,
    },
    {
        "id": "output",
        "name": "Output",
        "subtitle": "logits → softmax → roads",
        "stages": ["logits", "probs", "__roads__"],
        "open": True,
    },
]


# Pretty layer-name overrides; otherwise stage id is used.
NAME_OVERRIDES = {
    "00_pre":      "00 — Input (x/255)",
    "01_stem":     "01 — Stem conv (7×7, s=2)",
    "02_mp":       "02 — MaxPool (3×3, s=2)",
    "logits":      "logits (pre-softmax)",
    "probs":       "probs (softmax)",
    "__roads__":   "Roads (final classification)",
}


def _stage_label(stage: str) -> str:
    if stage in NAME_OVERRIDES:
        return NAME_OVERRIDES[stage]
    # l<L>b<B>[_<part>]
    m = re.match(r"^l(\d)b(\d)(_(c1|c2|ds))?$", stage)
    if m:
        L, B, _, part = m.groups()
        suffix = {"c1": "·conv1", "c2": "·conv2", "ds": "·downsample", None: ""}[part]
        return f"L{L}.b{B}{suffix}"
    m = re.match(r"^dec_(\d)(_(up|cat|c1))?$", stage)
    if m:
        i, _, part = m.groups()
        suffix = {"up": "·upsample", "cat": "·concat-skip", "c1": "·conv1", None: ""}[part]
        return f"Dec {i}{suffix}"
    return stage


def _shape_chip(meta: dict) -> str:
    h, w = meta["shape_native"]
    c = meta["channels"]
    return f"{c}×{h}×{w}"


def _layer_from_sidecar(stage: str, sidecar: dict) -> dict:
    layer = {
        "id": stage,
        "name": _stage_label(stage),
        "kind": "stage",
        "channels": sidecar["channels"],
        "shape_native": sidecar["shape_native"],
        "shape_chip": _shape_chip(sidecar),
        "views": [],
    }
    for view_id, view in sidecar["views"].items():
        v = {
            "id": view_id,
            "label": view["label"],
            "tiles": f"tiles/intermediate/{stage}_{view_id}/{{z}}/{{x}}/{{y}}.png",
        }
        if view["kind"] == "channel":
            v["legend"] = {
                "type": "colormap",
                "name": view["colormap"],
                "vmin": view["vmin"],
                "vmax": view["vmax"],
            }
        elif view["kind"] in ("pca_rgb", "rgb"):
            v["legend"] = {
                "type": "rgb",
                "bands": view["bands"],
                "vmins": view.get("vmins", [0,0,0]),
                "vmaxs": view.get("vmaxs", [1,1,1]),
                "subtype": view["kind"],
            }
        layer["views"].append(v)
    return layer


def _final_roads_layer() -> dict:
    return {
        "id": "roads",
        "name": NAME_OVERRIDES["__roads__"],
        "kind": "stage",
        "channels": 1,
        "shape_native": [4096, 4096],
        "shape_chip": "1×4096×4096",
        "views": [{
            "id": "default",
            "label": "Roads (blue)",
            "tiles": "tiles/classification/{z}/{x}/{y}.png",
            "legend": {
                "type": "categorical",
                "items": [
                    {"color": "#0066ff", "label": "road"},
                    {"color": "transparent", "label": "background"},
                ],
            },
        }],
    }


def _naip_layer() -> dict:
    return {
        "id": "naip",
        "name": "NAIP RGB (NIR dropped)",
        "kind": "base",
        "channels": 3,
        "shape_native": [4096, 4096],
        "shape_chip": "3×4096×4096",
        "views": [{
            "id": "default",
            "label": "Natural color",
            "tiles": "tiles/naip/{z}/{x}/{y}.png",
            "legend": {
                "type": "rgb",
                "bands": ["R", "G", "B"],
                "vmins": [0, 0, 0],
                "vmaxs": [255, 255, 255],
                "subtype": "rgb",
            },
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("naip_md_4096.tif"))
    ap.add_argument("--visuals-dir", type=Path, default=Path("intermediate_visuals"))
    ap.add_argument("--out", type=Path, default=Path("web/layers.json"))
    args = ap.parse_args()

    if not args.reference.exists():
        print(f"missing {args.reference}", file=sys.stderr); return 2

    import math
    with rasterio.open(args.reference) as src:
        if src.crs is None:
            print(f"[manifest] {args.reference} has no CRS", file=sys.stderr); return 2
        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        if not all(math.isfinite(v) for v in b):
            print(f"[manifest] non-finite bounds from {args.reference}: "
                  f"src.bounds={tuple(src.bounds)} src.crs={src.crs} → {b}",
                  file=sys.stderr); return 2
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    viewer = {
        "center": [cy, cx],
        "bounds": [[b[1], b[0]], [b[3], b[2]]],
        "minZoom": 14,
        "maxZoom": 19,
        "maxNativeZoom": 19,
        "intermediateMaxNativeZoom": 17,
    }

    sidecars = {p.stem: json.loads(p.read_text())
                for p in sorted(args.visuals_dir.glob("*.json"))}

    groups_out = []
    for grp in GROUPS:
        layers = []
        if grp["id"] == "input":
            layers.append(_naip_layer())
        for stage in grp["stages"]:
            if stage == "__roads__":
                layers.append(_final_roads_layer())
            elif stage in sidecars:
                layers.append(_layer_from_sidecar(stage, sidecars[stage]))
            else:
                print(f"[manifest] missing visuals for {stage}", file=sys.stderr)
        groups_out.append({
            "id": grp["id"],
            "name": grp["name"],
            "subtitle": grp.get("subtitle", ""),
            "open": grp.get("open", False),
            "layers": layers,
        })

    out = {"viewer": viewer, "groups": groups_out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False))
    n_layers = sum(len(g["layers"]) for g in groups_out)
    n_views = sum(len(l["views"]) for g in groups_out for l in g["layers"])
    print(f"[manifest] wrote {args.out}  ({n_layers} layers, {n_views} views)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
