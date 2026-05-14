# Web viewer

Single-page Leaflet viewer for the `gdal-unet` road-inference demo.

Dark glass-themed UI with a left-side **collapsible layer panel** that exposes
every U-Net stage (4 input + 27 encoder + 19 decoder + 3 output = 53 layers,
442 individual views). For each stage you get:

- **PCA false color (RGB)** — top 3 principal components of the channel stack.
- **Per-channel scrubber** — top 8 highest-variance channels rendered with
  the `turbo` colormap; slider switches between them.
- The final road layer is a 1-class blue overlay (background transparent).
- Probabilities (`probs`) use `viridis` for `P(road)`.

The legend in the bottom-right auto-updates with the active layer's name,
shape, colormap gradient, and per-band min/max.

## Layers

- **NAIP RGB** — base layer (NIR dropped).
- **Roads (final classification)** — colormapped overlay (background
  transparent, roads blue).
- **Intermediate stages** — populated by `scripts/build_intermediate_viewer.sh`.

## Serving

Tile paths in `app.js` are relative (`tiles/...`), so serve from `web/`
directly:

```bash
cd web && python -m http.server 8000
# open http://localhost:8000/
```

or from the repo root:

```bash
python -m http.server 8000
# open http://localhost:8000/web/
```

## (Re)building

```bash
# basic pipeline (NAIP + classification only):
bash scripts/run_demo.sh

# full viewer (above + all 51 intermediate stages, ~15-20 min):
bash scripts/run_demo.sh --with-intermediates

# or just the intermediate viewer (assumes scripts/run_demo.sh already ran):
bash scripts/build_intermediate_viewer.sh
```

The intermediate viewer pipeline is:

1. `scripts/render_intermediate.py` — for each `intermediate_output/*.tif`,
   produce a PCA-RGB false-color GeoTIFF and 8 per-channel turbo heatmap
   GeoTIFFs (probs gets a single viridis P(road) view; logits stays raw).
   PCA is computed at the layer's native resolution (covariance + `eigh`,
   subsampled fit when N>200k pixels) and only the 3 PCA bands or the 8
   top-variance channels are upsampled to the NAIP grid.
2. `scripts/tile_intermediate.sh` — `gdal raster tile --convention xyz` over
   each visualization, parallel `xargs -P 8`, z=14–17 (matches ≤1 m source res).
3. `scripts/build_layers_manifest.py` — emits `web/layers.json` driving the
   side panel.

## Adding a custom overlay

Add an entry to `web/layers.json`:

```jsonc
{
  "id": "my-layer",
  "name": "My layer",
  "kind": "stage",
  "shape_chip": "1×4096×4096",
  "views": [{
    "id": "default",
    "label": "Custom heatmap",
    "tiles": "tiles/my-layer/{z}/{x}/{y}.png",
    "legend": { "type": "colormap", "name": "turbo", "vmin": 0, "vmax": 1 }
  }]
}
```

…inside one of the `groups[].layers` arrays.
