// MapLibre setup + GPU-crossfade overlay swap.
// Coordinates in the manifest follow Leaflet conventions ([lat, lng]);
// MapLibre wants [lng, lat], so all conversions happen here.

const CROSSFADE_MS = 280;
const BASE_OPACITY = 1.0;
const ROADS_OPACITY = 0.85;
const OVERLAY_OPACITY = 0.92;

let map = null;
let viewer = null;
let activeLayerKey = null;       // string id used as MapLibre source/layer id
let pendingLayerKey = null;
let pendingNonce = 0;            // bumped on every setOverlay call
let currentOpacityScale = 1.0;   // user slider, 0..1

export function initMap(viewerDef, naipLayer) {
  viewer = viewerDef;

  // MapLibre default = 16. Bumping past the browser's per-origin connection
  // cap (6 over HTTP/1.1) is only an incremental win, but it keeps the queue
  // saturated as connections free up.
  if (typeof maplibregl.setMaxParallelImageRequests === "function") {
    maplibregl.setMaxParallelImageRequests(32);
  }

  const [lat, lng] = viewerDef.center;
  const [[s, w], [n, e]] = viewerDef.bounds;

  map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: {
        osm: {
          type: "raster",
          tiles: [
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
          ],
          tileSize: 256,
          minzoom: 0,
          maxzoom: 19,
          attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        },
      },
      layers: [
        {
          id: "bg",
          type: "background",
          paint: { "background-color": "#0e0b08" },
        },
        {
          id: "osm",
          type: "raster",
          source: "osm",
          paint: {
            // dim + desaturate + slight warm hue shift to match the espresso theme
            "raster-opacity": 0.55,
            "raster-saturation": -0.55,
            "raster-brightness-max": 0.55,
            "raster-contrast": -0.05,
            "raster-hue-rotate": 18,
            "raster-fade-duration": 0,
          },
        },
      ],
    },
    center: [lng, lat],
    zoom: viewerDef.zoom || 15,
    minZoom: 12,
    maxZoom: viewerDef.maxZoom,
    maxBounds: [[w - 0.1, s - 0.1], [e + 0.1, n + 0.1]],
    attributionControl: { compact: true },
    fadeDuration: 120,
    refreshExpiredTiles: false,
  });

  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-left");
  map.fitBounds([[w, s], [e, n]], { padding: 30, duration: 0 });

  return new Promise(resolve => {
    map.on("load", () => {
      // base NAIP layer (always on)
      if (naipLayer) {
        const naipTiles = naipLayer.views[0].tiles;
        map.addSource("naip", {
          type: "raster",
          tiles: [absolute(naipTiles)],
          tileSize: 256,
          minzoom: viewer.minZoom,
          maxzoom: viewer.maxNativeZoom,
          attribution:
            'NAIP / USDA via Microsoft Planetary Computer · ' +
            '<a href="https://github.com/isaaccorley/gdal-unet">gdal-unet</a>',
        });
        map.addLayer({
          id: "naip",
          type: "raster",
          source: "naip",
          paint: {
            "raster-opacity": BASE_OPACITY,
            "raster-fade-duration": 120,
          },
        });
      }
      resolve(map);
    });
  });
}

function absolute(url) {
  // tile paths in manifest are relative; leave them as-is — MapLibre resolves
  // them against the document base.
  return url;
}

function targetOpacityFor(layerDef) {
  return (layerDef.id === "roads" ? ROADS_OPACITY : OVERLAY_OPACITY) * currentOpacityScale;
}

export function setOverlay(layerDef, viewDef) {
  if (!map) return;

  const key = `ov:${layerDef.id}:${viewDef.id}`;

  // user asked for the layer that's already showing → cancel any in-flight swap.
  if (key === activeLayerKey) {
    pendingNonce++;
    if (pendingLayerKey && pendingLayerKey !== key) cleanup(pendingLayerKey);
    pendingLayerKey = null;
    return;
  }
  if (key === pendingLayerKey) return;

  // a different swap was in flight → drop its half-loaded source and start fresh.
  if (pendingLayerKey) cleanup(pendingLayerKey);

  const myNonce = ++pendingNonce;
  pendingLayerKey = key;

  const maxNative =
    layerDef.id === "roads" ? viewer.maxNativeZoom : viewer.intermediateMaxNativeZoom;

  if (!map.getSource(key)) {
    map.addSource(key, {
      type: "raster",
      tiles: [absolute(viewDef.tiles)],
      tileSize: 256,
      minzoom: viewer.minZoom,
      maxzoom: maxNative,
    });
  }
  if (!map.getLayer(key)) {
    map.addLayer({
      id: key,
      type: "raster",
      source: key,
      paint: {
        "raster-opacity": 0,
        "raster-opacity-transition": { duration: CROSSFADE_MS },
        "raster-fade-duration": 0,
        // Pin overlays to nearest-neighbor sampling. MapLibre's default
        // "linear" anchors its 2x2 interpolation kernel on source-pixel
        // CENTERS. When overlays have a coarser native resolution than the
        // NAIP base, the kernel's effective grid phase differs per layer
        // and the eye reads it as a sub-pixel drift between the two —
        // even though every layer's geotransform is correct. nearest
        // shows true source cells, which makes alignment visually
        // verifiable.
        "raster-resampling": "nearest",
      },
    });
  }

  const target = targetOpacityFor(layerDef);

  // dequeue from preload ring (the source is being promoted to active)
  const idx = preloadedKeys.indexOf(key);
  if (idx !== -1) preloadedKeys.splice(idx, 1);

  // if preloaded and already fully loaded, no waiting needed
  if (map.isSourceLoaded && map.isSourceLoaded(key)) {
    commit(key, target, myNonce);
    return;
  }

  const onData = (e) => {
    if (myNonce !== pendingNonce) { map.off("sourcedata", onData); return; }
    if (e.sourceId !== key || !e.isSourceLoaded) return;
    map.off("sourcedata", onData);
    commit(key, target, myNonce);
  };
  map.on("sourcedata", onData);

  // safety: blank tiles may never trigger isSourceLoaded — commit anyway.
  setTimeout(() => {
    if (myNonce !== pendingNonce) return;
    map.off("sourcedata", onData);
    commit(key, target, myNonce);
  }, 1400);
}

function commit(newKey, newOpacity, myNonce) {
  if (myNonce !== pendingNonce) return;
  const oldKey = activeLayerKey;
  map.setPaintProperty(newKey, "raster-opacity", newOpacity);
  if (oldKey && oldKey !== newKey) {
    map.setPaintProperty(oldKey, "raster-opacity", 0);
    setTimeout(() => {
      if (activeLayerKey !== oldKey) cleanup(oldKey);
    }, CROSSFADE_MS + 80);
  }
  activeLayerKey = newKey;
  pendingLayerKey = null;
}

function cleanup(key) {
  try {
    if (map.getLayer(key)) map.removeLayer(key);
    if (map.getSource(key)) map.removeSource(key);
  } catch {}
}

// Preload a layer/view without making it active. Adds the source + a hidden
// layer (opacity 0). Tiles fetch in the background; if the user later clicks
// the same layer, setOverlay() reuses the existing source for an instant swap.
const PRELOAD_CAP = 4;
const preloadedKeys = [];   // FIFO, excludes the active layer

export function preloadOverlay(layerDef, viewDef) {
  if (!map) return;
  const key = `ov:${layerDef.id}:${viewDef.id}`;
  if (key === activeLayerKey || key === pendingLayerKey) return;
  if (map.getSource(key)) return;     // already present

  const maxNative =
    layerDef.id === "roads" ? viewer.maxNativeZoom : viewer.intermediateMaxNativeZoom;

  map.addSource(key, {
    type: "raster",
    tiles: [absolute(viewDef.tiles)],
    tileSize: 256,
    minzoom: viewer.minZoom,
    maxzoom: maxNative,
  });
  map.addLayer({
    id: key,
    type: "raster",
    source: key,
    paint: {
      "raster-opacity": 0,
      "raster-opacity-transition": { duration: CROSSFADE_MS },
      "raster-fade-duration": 0,
      "raster-resampling": "nearest",
    },
  });

  preloadedKeys.push(key);
  while (preloadedKeys.length > PRELOAD_CAP) {
    const evict = preloadedKeys.shift();
    if (evict !== activeLayerKey && evict !== pendingLayerKey) cleanup(evict);
  }
}

export function setOpacityScale(scale01) {
  currentOpacityScale = Math.max(0, Math.min(1, scale01));
  if (activeLayerKey && map.getLayer(activeLayerKey)) {
    const baseTarget = activeLayerKey.startsWith("ov:roads") ? ROADS_OPACITY : OVERLAY_OPACITY;
    map.setPaintProperty(activeLayerKey, "raster-opacity", baseTarget * currentOpacityScale);
  }
}

export function getMap() { return map; }
