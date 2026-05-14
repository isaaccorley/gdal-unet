// Inline SVG U-Net topology mini-map.
// Layout: encoder column on left, decoder column on right, bottleneck centered.
// Skip connections drawn as dashed arcs across the middle.

const NODES = [
  // row: 0 = top, 5 = bottom (bottleneck)
  { id: "input",    label: "IN",   col: "L", row: 0 },
  { id: "output",   label: "OUT",  col: "R", row: 0 },
  { id: "encoder1", label: "ENC1", col: "L", row: 1 },
  { id: "decoder4", label: "DEC4", col: "R", row: 1 },
  { id: "encoder2", label: "ENC2", col: "L", row: 2 },
  { id: "decoder3", label: "DEC3", col: "R", row: 2 },
  { id: "encoder3", label: "ENC3", col: "L", row: 3 },
  { id: "decoder2", label: "DEC2", col: "R", row: 3 },
  { id: "encoder4", label: "ENC4", col: "L", row: 4 },
  { id: "decoder1", label: "DEC1", col: "R", row: 4 },
  { id: "decoder0", label: "BOT",  col: "C", row: 5 },
];

const SKIPS = [
  ["encoder1", "decoder4"],
  ["encoder2", "decoder3"],
  ["encoder3", "decoder2"],
  ["encoder4", "decoder1"],
];

const VB_W = 260;
const ROW_GAP = 30;
const NODE_W = 56;
const NODE_H = 22;
const COL_L = 60;
const COL_R = 200;
const COL_C = 130;
const TOP_PAD = 8;

function nodeXY(node) {
  const x = node.col === "L" ? COL_L : node.col === "R" ? COL_R : COL_C;
  const y = TOP_PAD + node.row * ROW_GAP + NODE_H / 2;
  return { x, y };
}

export function renderMinimap(container, { onGroupClick }) {
  const totalH = TOP_PAD * 2 + 5 * ROW_GAP + NODE_H;
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${VB_W} ${totalH}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "U-Net topology");

  const byId = new Map(NODES.map(n => [n.id, n]));

  // edges: vertical along each column
  const colChain = (col, ids) => {
    for (let i = 0; i < ids.length - 1; i++) {
      const a = nodeXY(byId.get(ids[i]));
      const b = nodeXY(byId.get(ids[i + 1]));
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", a.x);
      line.setAttribute("y1", a.y + NODE_H / 2 - 1);
      line.setAttribute("x2", b.x);
      line.setAttribute("y2", b.y - NODE_H / 2 + 1);
      line.setAttribute("class", "mm-edge");
      svg.appendChild(line);
    }
  };
  colChain("L", ["input", "encoder1", "encoder2", "encoder3", "encoder4"]);
  colChain("R", ["output", "decoder4", "decoder3", "decoder2", "decoder1"]);

  // edges to bottleneck
  for (const id of ["encoder4", "decoder1"]) {
    const a = nodeXY(byId.get(id));
    const b = nodeXY(byId.get("decoder0"));
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y + NODE_H / 2 - 1);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y - NODE_H / 2 + 1);
    line.setAttribute("class", "mm-edge");
    svg.appendChild(line);
  }

  // skip connections (dashed arcs)
  for (const [aId, bId] of SKIPS) {
    const a = nodeXY(byId.get(aId));
    const b = nodeXY(byId.get(bId));
    const path = document.createElementNS(svgNS, "path");
    const midX = (a.x + b.x) / 2;
    const dx = Math.abs(b.x - a.x) * 0.35;
    path.setAttribute(
      "d",
      `M ${a.x + NODE_W / 2} ${a.y} C ${a.x + NODE_W / 2 + dx} ${a.y}, ${b.x - NODE_W / 2 - dx} ${b.y}, ${b.x - NODE_W / 2} ${b.y}`
    );
    path.setAttribute("class", "mm-skip");
    svg.appendChild(path);
  }

  // nodes
  for (const node of NODES) {
    const { x, y } = nodeXY(node);
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("class", "mm-node");
    g.setAttribute("data-group-id", node.id);
    g.setAttribute("transform", `translate(${x - NODE_W / 2}, ${y - NODE_H / 2})`);

    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("class", "mm-node-bg");
    rect.setAttribute("width", NODE_W);
    rect.setAttribute("height", NODE_H);
    rect.setAttribute("rx", 4);
    g.appendChild(rect);

    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("class", "mm-node-label");
    text.setAttribute("x", NODE_W / 2);
    text.setAttribute("y", NODE_H / 2 + 3);
    text.textContent = node.label;
    g.appendChild(text);

    g.addEventListener("click", () => onGroupClick && onGroupClick(node.id));
    svg.appendChild(g);
  }

  container.innerHTML = "";
  container.appendChild(svg);
  return { svg };
}

export function setActiveGroup(container, groupId) {
  const svg = container.querySelector("svg");
  if (!svg) return;
  svg.querySelectorAll(".mm-node").forEach(n => {
    n.classList.toggle("is-active", n.getAttribute("data-group-id") === groupId);
  });
}
