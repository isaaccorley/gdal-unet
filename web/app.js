// gdal-unet viewer · main wiring.
//
// State + sidebar + search + keyboard nav + legend. Map work lives in
// map.js; topology mini-map in unet-minimap.js.

import { initMap, setOverlay, setOpacityScale, preloadOverlay } from "./map.js";
import { renderMinimap, setActiveGroup } from "./unet-minimap.js";

const state = {
  viewer: null,
  groups: [],
  layers: new Map(),    // id -> { layer, group, channelViewIds, row, controls }
  order: [],            // flat array of layer ids in sidebar order
  activeLayerId: null,
  activeViewId: null,
  channelMode: false,   // is the active layer's view a channel scrubber view?
  channelIdx: 0,
};

const els = {
  search: null,
  sidebarBody: null,
  legend: null,
  legendName: null,
  legendChip: null,
  legendView: null,
  legendBody: null,
  opacity: null,
  opacityVal: null,
  minimap: null,
  toggleGroups: null,
};

// ---------- number formatting ----------
function fmtNum(x) {
  if (!isFinite(x)) return "—";
  if (Math.abs(x) >= 100) return x.toFixed(0);
  if (Math.abs(x) >= 10) return x.toFixed(2);
  if (Math.abs(x) >= 1) return x.toFixed(3);
  return x.toFixed(4);
}

// ---------- sidebar ----------
function buildSidebar(manifest) {
  const body = els.sidebarBody;
  body.innerHTML = "";

  manifest.groups.forEach(group => {
    const det = document.createElement("details");
    det.className = "group";
    det.dataset.groupId = group.id;
    if (group.open) det.setAttribute("open", "");
    det.innerHTML = `
      <summary>
        <span>${group.name}</span>
        <span class="group-sub">${group.subtitle || ""}</span>
      </summary>
      <div class="group-body"></div>`;
    const groupBody = det.querySelector(".group-body");

    group.layers.forEach(layer => {
      const channelViewIds = layer.views
        .filter(v => v.id.startsWith("ch"))
        .map(v => v.id);

      const row = document.createElement("div");
      row.className = "layer-row";
      row.dataset.layerId = layer.id;
      row.innerHTML = `
        <span class="layer-name">${layer.name}</span>
        <span class="shape-chip">${layer.shape_chip || ""}</span>`;
      row.addEventListener("click", () => activateLayer(layer.id));
      groupBody.appendChild(row);

      const ctrl = document.createElement("div");
      ctrl.className = "layer-controls";
      ctrl.dataset.layerId = layer.id;
      ctrl.innerHTML = controlsHtml(layer);
      groupBody.appendChild(ctrl);
      wireControls(ctrl, layer);

      state.layers.set(layer.id, { layer, group, channelViewIds, row, controls: ctrl });
      state.order.push(layer.id);
    });

    body.appendChild(det);
  });
}

function controlsHtml(layer) {
  const hasChannels = layer.views.some(v => v.id.startsWith("ch"));
  const tabs = [];
  layer.views.forEach(v => {
    if (v.id === "rgb" || v.id === "default") {
      const label = v.id === "rgb" ? "RGB" : "MAP";
      tabs.push(`<button class="view-tab" data-view-id="${v.id}">${label}</button>`);
    }
  });
  if (hasChannels) {
    tabs.push(`<button class="view-tab" data-view-mode="channel">CH</button>`);
  }

  let scrubber = "";
  if (hasChannels) {
    const channels = layer.views.filter(v => v.id.startsWith("ch"));
    const max = channels.length - 1;
    scrubber = `
      <div class="scrubber">
        <div class="scrubber-label">
          <span class="scrub-name">Channel</span>
          <span class="scrub-range">[—, —]</span>
        </div>
        <input type="range" min="0" max="${max}" value="0" step="1" />
      </div>`;
  }

  return `<div class="view-tabs">${tabs.join("")}</div>${scrubber}`;
}

function wireControls(ctrl, layer) {
  const tabs = ctrl.querySelectorAll(".view-tab");
  const scrubber = ctrl.querySelector(".scrubber");
  const range = ctrl.querySelector('input[type="range"]');
  const scrubName = ctrl.querySelector(".scrub-name");
  const scrubRange = ctrl.querySelector(".scrub-range");
  const channels = layer.views.filter(v => v.id.startsWith("ch"));

  function applyChannel(idx) {
    const v = channels[idx];
    if (!v) return;
    state.channelIdx = idx;
    state.channelMode = true;
    scrubName.textContent = v.label;
    const lg = v.legend || {};
    scrubRange.textContent =
      lg.vmin !== undefined && lg.vmax !== undefined
        ? `[${fmtNum(lg.vmin)}, ${fmtNum(lg.vmax)}]`
        : "";
    updateRangeFill(range);
    switchView(layer.id, v.id);
  }

  tabs.forEach(t => {
    t.addEventListener("click", () => {
      setTabActive(ctrl, t);
      if (t.dataset.viewId) {
        state.channelMode = false;
        scrubber.classList.remove("is-visible");
        switchView(layer.id, t.dataset.viewId);
      } else if (t.dataset.viewMode === "channel") {
        state.channelMode = true;
        scrubber.classList.add("is-visible");
        applyChannel(parseInt(range.value, 10));
      }
    });
  });

  if (range) {
    range.addEventListener("input", () => applyChannel(parseInt(range.value, 10)));
    updateRangeFill(range);
  }
}

function setTabActive(ctrl, tabEl) {
  ctrl.querySelectorAll(".view-tab").forEach(x => x.classList.remove("is-active"));
  tabEl.classList.add("is-active");
}

function updateRangeFill(el) {
  if (!el) return;
  const min = +el.min, max = +el.max, v = +el.value;
  const pct = max === min ? 0 : ((v - min) / (max - min)) * 100;
  el.style.setProperty("--v", `${pct}%`);
}

// ---------- layer / view activation ----------
function activateLayer(layerId, opts = {}) {
  const entry = state.layers.get(layerId);
  if (!entry) return;
  const { layer, group, controls, row } = entry;

  document.querySelectorAll(".layer-row.is-active").forEach(el => el.classList.remove("is-active"));
  row.classList.add("is-active");

  // Expand the group containing this row so the controls are visible.
  const det = row.closest("details.group");
  if (det && !det.hasAttribute("open")) det.setAttribute("open", "");

  state.activeLayerId = layerId;

  // pick default view
  const tabs = controls.querySelectorAll(".view-tab");
  tabs.forEach(t => t.classList.remove("is-active"));
  const scrubber = controls.querySelector(".scrubber");

  const rgbView = layer.views.find(v => v.id === "rgb" || v.id === "default");
  if (rgbView) {
    const tab = controls.querySelector(`.view-tab[data-view-id="${rgbView.id}"]`);
    if (tab) tab.classList.add("is-active");
    state.channelMode = false;
    if (scrubber) scrubber.classList.remove("is-visible");
    switchView(layerId, rgbView.id);
  } else if (entry.channelViewIds.length) {
    const tab = controls.querySelector('.view-tab[data-view-mode="channel"]');
    if (tab) tab.classList.add("is-active");
    state.channelMode = true;
    if (scrubber) scrubber.classList.add("is-visible");
    const range = controls.querySelector('input[type="range"]');
    const idx = parseInt((range && range.value) || "0", 10);
    state.channelIdx = idx;
    const v = layer.views.find(x => x.id === entry.channelViewIds[idx]);
    if (v) {
      controls.querySelector(".scrub-name").textContent = v.label;
      const lg = v.legend || {};
      controls.querySelector(".scrub-range").textContent =
        lg.vmin !== undefined && lg.vmax !== undefined
          ? `[${fmtNum(lg.vmin)}, ${fmtNum(lg.vmax)}]`
          : "";
      switchView(layerId, v.id);
    }
  } else if (layer.views[0]) {
    switchView(layerId, layer.views[0].id);
  }

  setActiveGroup(els.minimap, group.id);

  if (!opts.silentScroll) {
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  prefetchNeighbors(layerId);
}

// Start fetching tiles for ±1 and ±2 neighbors so keyboard nav feels instant.
// Uses each layer's preferred default view.
function prefetchNeighbors(layerId) {
  const order = visibleOrder();
  const i = order.indexOf(layerId);
  if (i < 0) return;
  const offsets = [+1, -1, +2, -2];
  for (const o of offsets) {
    const id = order[i + o];
    if (!id) continue;
    const entry = state.layers.get(id);
    if (!entry) continue;
    const view = preferredView(entry.layer);
    if (view) preloadOverlay(entry.layer, view);
  }
}

function preferredView(layer) {
  return (
    layer.views.find(v => v.id === "rgb" || v.id === "default") ||
    layer.views.find(v => v.id.startsWith("ch")) ||
    layer.views[0]
  );
}

function switchView(layerId, viewId) {
  const entry = state.layers.get(layerId);
  if (!entry) return;
  const view = entry.layer.views.find(v => v.id === viewId);
  if (!view) return;
  state.activeViewId = viewId;
  setOverlay(entry.layer, view);
  renderLegend(entry.layer, view);
}

// ---------- legend ----------
function renderLegend(layer, view) {
  els.legend.classList.remove("hidden");
  els.legendName.textContent = layer.name;
  els.legendChip.textContent = layer.shape_chip || "";
  els.legendView.textContent = view.label;

  const lg = view.legend || {};
  if (lg.type === "colormap") {
    const mid = (lg.vmin + lg.vmax) / 2;
    els.legendBody.innerHTML = `
      <div class="colormap-bar ${lg.name}"></div>
      <div class="range-row">
        <span>${fmtNum(lg.vmin)}</span>
        <span class="mid">${fmtNum(mid)}</span>
        <span>${fmtNum(lg.vmax)}</span>
      </div>
      <div class="legend-note">
        <code>${lg.name}</code> · 1–99 pct stretch
      </div>`;
  } else if (lg.type === "rgb") {
    const subtype = lg.subtype === "pca_rgb" ? "PCA components" : "bands";
    const pieces = lg.bands.map((b, i) => {
      const lo = lg.vmins ? fmtNum(lg.vmins[i]) : "0";
      const hi = lg.vmaxs ? fmtNum(lg.vmaxs[i]) : "255";
      return `<span title="${lo} → ${hi}">${b}</span>`;
    });
    els.legendBody.innerHTML = `
      <div class="rgb-bands">${pieces.join("")}</div>
      <div class="legend-note">${subtype} · 1–99 pct stretch</div>`;
  } else if (lg.type === "categorical") {
    const rows = lg.items.map(it => `
      <div class="swatch-row">
        <span class="swatch" style="background:${it.color}"></span>
        <span>${it.label}</span>
      </div>`).join("");
    els.legendBody.innerHTML = `<div class="swatch-list">${rows}</div>`;
  } else {
    els.legendBody.innerHTML = "—";
  }
}

// ---------- group toggle button ----------
function refreshToggleState() {
  if (!els.toggleGroups) return;
  const groups = document.querySelectorAll("details.group");
  const anyOpen = Array.from(groups).some(d => d.hasAttribute("open"));
  els.toggleGroups.dataset.state = anyOpen ? "open" : "closed";
  els.toggleGroups.title = anyOpen ? "Collapse all groups" : "Expand all groups";
}

// ---------- search ----------
function applySearch(q) {
  const norm = q.trim().toLowerCase();
  let firstVisible = null;
  state.layers.forEach(entry => {
    const hay = (entry.layer.name + " " + entry.layer.id).toLowerCase();
    const match = !norm || hay.includes(norm);
    entry.row.classList.toggle("is-hidden", !match);
    entry.controls.classList.toggle("is-hidden", !match);
    if (match && !firstVisible) firstVisible = entry.layer.id;
  });
  // open every group while a query is present, collapse others when cleared
  document.querySelectorAll("details.group").forEach(det => {
    if (norm) det.setAttribute("open", "");
  });
}

// ---------- keyboard ----------
function visibleOrder() {
  return state.order.filter(id => !state.layers.get(id).row.classList.contains("is-hidden"));
}

function stepLayer(delta) {
  const order = visibleOrder();
  if (!order.length) return;
  const cur = order.indexOf(state.activeLayerId);
  const next = order[(cur + delta + order.length) % order.length];
  activateLayer(next);
}

function stepChannel(delta) {
  if (!state.channelMode) return;
  const entry = state.layers.get(state.activeLayerId);
  if (!entry || !entry.channelViewIds.length) return;
  const range = entry.controls.querySelector('input[type="range"]');
  if (!range) return;
  const max = +range.max;
  const next = Math.max(0, Math.min(max, state.channelIdx + delta));
  if (next === state.channelIdx) return;
  range.value = String(next);
  range.dispatchEvent(new Event("input"));
}

function attachKeyboard() {
  document.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) || "";
    const isInput = tag === "INPUT" || tag === "TEXTAREA";

    if (e.key === "/" && !isInput) {
      e.preventDefault();
      els.search.focus();
      els.search.select();
      return;
    }
    if (e.key === "Escape" && isInput) {
      els.search.value = "";
      applySearch("");
      els.search.blur();
      return;
    }
    if (isInput) return;

    if (e.key === "ArrowLeft")  { e.preventDefault(); stepLayer(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); stepLayer(+1); }
    if (e.key === "j") stepChannel(+1);
    if (e.key === "k") stepChannel(-1);
  });
}

// ---------- boot ----------
async function boot() {
  els.search       = document.getElementById("search");
  els.sidebarBody  = document.getElementById("sidebar-body");
  els.legend       = document.getElementById("legend");
  els.legendName   = els.legend.querySelector(".legend-name");
  els.legendChip   = els.legend.querySelector(".legend-chip");
  els.legendView   = els.legend.querySelector(".legend-view");
  els.legendBody   = els.legend.querySelector(".legend-body");
  els.opacity      = document.getElementById("opacity");
  els.opacityVal   = document.getElementById("opacity-val");
  els.minimap      = document.getElementById("minimap-svg");
  els.toggleGroups = document.getElementById("toggle-groups");

  updateRangeFill(els.opacity);
  els.opacity.addEventListener("input", () => {
    updateRangeFill(els.opacity);
    const v = +els.opacity.value;
    els.opacityVal.textContent = String(v);
    setOpacityScale(v / 100);
  });

  els.search.addEventListener("input", () => applySearch(els.search.value));

  els.toggleGroups.addEventListener("click", () => {
    const groups = document.querySelectorAll("details.group");
    const anyOpen = Array.from(groups).some(d => d.hasAttribute("open"));
    if (anyOpen) {
      groups.forEach(d => d.removeAttribute("open"));
    } else {
      groups.forEach(d => d.setAttribute("open", ""));
    }
    refreshToggleState();
  });
  // also keep the icon in sync when individual groups toggle
  document.addEventListener("toggle", e => {
    if (e.target.matches("details.group")) refreshToggleState();
  }, true);

  let manifest;
  try {
    const r = await fetch("layers.json");
    if (!r.ok) throw new Error(`layers.json HTTP ${r.status}`);
    manifest = await r.json();
  } catch (err) {
    document.getElementById("loading").innerHTML = `
      <div style="color: var(--accent-strong); padding: 6px 0;">
        Failed to load layers.json: ${err}
      </div>
      <div style="font-size: 11px; color: var(--muted); margin-top: 6px;">
        Run <code>scripts/build_intermediate_viewer.sh</code> to generate it.
      </div>`;
    console.error(err);
    return;
  }

  state.viewer = manifest.viewer;
  state.groups = manifest.groups;

  const inputGroup = manifest.groups.find(g => g.id === "input");
  const naipLayer = inputGroup && inputGroup.layers.find(l => l.id === "naip");

  await initMap(manifest.viewer, naipLayer);

  buildSidebar(manifest);
  document.getElementById("loading")?.remove();
  refreshToggleState();

  renderMinimap(els.minimap, {
    onGroupClick: (groupId) => {
      const group = manifest.groups.find(g => g.id === groupId);
      if (!group || !group.layers.length) return;
      const det = document.querySelector(`details.group[data-group-id="${groupId}"]`);
      if (det) det.setAttribute("open", "");
      // pick a sensible default: layer named like 'fa' (final activation) or first.
      const pick =
        group.layers.find(l => l.id.endsWith("_fa")) ||
        group.layers.find(l => l.id === "probs") ||
        group.layers.find(l => l.id === "roads") ||
        group.layers[0];
      activateLayer(pick.id);
    },
  });

  // initial selection
  const initial =
    state.layers.has("probs") ? "probs" :
    state.layers.has("roads") ? "roads" :
    state.order[0];
  if (initial) activateLayer(initial, { silentScroll: true });

  attachKeyboard();
}

boot();
