import Graph from "graphology";
import Sigma from "sigma";
import DATA from "./virus_network_data.json";

type NodeData = {
  id: string; label: string; family: string;
  x: number; y: number; size: number; color: string;
  h2h: number; zoo: number; labelled: boolean; famous: boolean;
};
type EdgeData = { source: string; target: string; weight: number; };

const nodes = DATA.nodes as NodeData[];
const edges = DATA.edges as EdgeData[];

document.getElementById("app")!.innerHTML = `
<div style="background:#060a12;height:100vh;width:100%;position:relative;overflow:hidden;font-family:'SF Mono',monospace;">
  <div id="sigma-container" style="width:100%;height:100%;"></div>

  <div style="position:absolute;top:10px;left:10px;width:14px;height:14px;border-top:1px solid rgba(100,200,255,0.35);border-left:1px solid rgba(100,200,255,0.35);pointer-events:none;"></div>
  <div style="position:absolute;top:10px;right:10px;width:14px;height:14px;border-top:1px solid rgba(100,200,255,0.35);border-right:1px solid rgba(100,200,255,0.35);pointer-events:none;"></div>
  <div style="position:absolute;bottom:10px;left:10px;width:14px;height:14px;border-bottom:1px solid rgba(100,200,255,0.35);border-left:1px solid rgba(100,200,255,0.35);pointer-events:none;"></div>
  <div style="position:absolute;bottom:10px;right:10px;width:14px;height:14px;border-bottom:1px solid rgba(100,200,255,0.35);border-right:1px solid rgba(100,200,255,0.35);pointer-events:none;"></div>

  <div style="position:absolute;top:18px;left:22px;pointer-events:none;">
    <div style="font-size:9px;letter-spacing:0.14em;color:rgba(100,200,255,0.5);text-transform:uppercase;">Viral phenotype network · EID2 × Vir2vec · SVM predictions</div>
    <div id="count-label" style="font-size:12px;color:rgba(180,220,255,0.9);margin-top:3px;">${nodes.length.toLocaleString()} viruses · ${edges.length.toLocaleString()} connections</div>
  </div>

  <div style="position:absolute;top:18px;right:18px;display:flex;gap:5px;">
    <button class="mode-btn active" id="btn-all">All</button>
    <button class="mode-btn" id="btn-h2h">H2H risk</button>
    <button class="mode-btn" id="btn-zoo">Zoonotic</button>
    <button class="mode-btn" id="btn-both">Both risks</button>
    <button class="mode-btn" id="btn-labelled">Labelled</button>
    <button class="mode-btn" id="btn-edges">Show edges</button>
  </div>

  <div style="position:absolute;bottom:20px;left:20px;display:flex;flex-direction:column;gap:5px;pointer-events:none;">
    <div style="font-size:9px;letter-spacing:0.1em;color:rgba(100,200,255,0.3);text-transform:uppercase;margin-bottom:2px;">Node size = predicted risk</div>
    <div style="display:flex;align-items:center;gap:7px;font-size:9px;letter-spacing:0.08em;color:rgba(255,255,255,0.4);text-transform:uppercase;"><div style="width:7px;height:7px;border-radius:50%;background:#3b82f6;flex-shrink:0;"></div>Low risk</div>
    <div style="display:flex;align-items:center;gap:7px;font-size:9px;letter-spacing:0.08em;color:rgba(255,255,255,0.4);text-transform:uppercase;"><div style="width:7px;height:7px;border-radius:50%;background:#f97316;flex-shrink:0;"></div>H2H transmissible</div>
    <div style="display:flex;align-items:center;gap:7px;font-size:9px;letter-spacing:0.08em;color:rgba(255,255,255,0.4);text-transform:uppercase;"><div style="width:7px;height:7px;border-radius:50%;background:#10b981;flex-shrink:0;"></div>Zoonotic spillover</div>
    <div style="display:flex;align-items:center;gap:7px;font-size:9px;letter-spacing:0.08em;color:rgba(255,255,255,0.4);text-transform:uppercase;"><div style="width:7px;height:7px;border-radius:50%;background:#a855f7;flex-shrink:0;"></div>Both risks</div>
  </div>

  <div id="tt" style="position:fixed;background:rgba(4,8,16,0.97);border:1px solid rgba(100,200,255,0.25);border-radius:4px;padding:14px 18px;display:none;pointer-events:none;min-width:230px;z-index:9999;">
    <div id="tt-name" style="font-size:12px;color:#00ccff;margin-bottom:10px;letter-spacing:0.04em;font-weight:600;"></div>
    <div class="tt-row"><span>Family</span><span class="tt-val" id="tt-fam"></span></div>
    <div class="tt-row"><span>H2H predicted risk</span><span class="tt-val" id="tt-h2h"></span></div>
    <div class="tt-row"><span>Zoonotic predicted risk</span><span class="tt-val" id="tt-zoo"></span></div>
    <div class="tt-row"><span>Label source</span><span class="tt-val" id="tt-src"></span></div>
  </div>

  <div style="position:absolute;bottom:20px;right:20px;font-size:9px;color:rgba(100,200,255,0.2);letter-spacing:0.08em;text-transform:uppercase;pointer-events:none;">scroll to zoom · drag to pan · hover node</div>
</div>
`;

const style = document.createElement("style");
style.textContent = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { overflow: hidden; }
  .mode-btn {
    background: transparent; border: 1px solid rgba(100,200,255,0.15);
    color: rgba(100,200,255,0.45); font-size: 9px; font-family: 'SF Mono', monospace;
    letter-spacing: 0.1em; padding: 5px 10px; cursor: pointer;
    text-transform: uppercase; border-radius: 2px; transition: all 0.15s;
  }
  .mode-btn:hover, .mode-btn.active {
    background: rgba(100,200,255,0.08); border-color: rgba(100,200,255,0.4);
    color: rgba(100,200,255,0.95);
  }
  .tt-row {
    font-size: 10px; letter-spacing: 0.06em; color: rgba(180,200,255,0.5);
    display: flex; justify-content: space-between; gap: 20px;
    margin-top: 5px; text-transform: uppercase;
  }
  .tt-val { color: #ffffff; font-weight: 500; }
`;
document.head.appendChild(style);

const graph = new Graph({ multi: false, type: "undirected" });

nodes.forEach((n) => {
  graph.addNode(n.id, {
    x: n.x, y: n.y,
    size: n.size,
    color: n.color,
    originalColor: n.color,
    label: "",
    h2h: n.h2h, zoo: n.zoo,
    family: n.family || "Unknown",
    fullLabel: n.label || ("Taxid " + n.id),
    labelled: n.labelled,
    famous: n.famous,
  });
});

edges.forEach((e) => {
  if (graph.hasNode(e.source) && graph.hasNode(e.target) && !graph.hasEdge(e.source, e.target)) {
    graph.addEdge(e.source, e.target, {
      size: 0.2,
      color: "rgba(100,200,255,0.03)",
      hidden: true,
    });
  }
});

const renderer = new Sigma(graph, document.getElementById("sigma-container")!, {
  renderEdgeLabels: false,
  defaultEdgeType: "line",
  backgroundColor: "#060a12",
  minCameraRatio: 0.02,
  maxCameraRatio: 20,
});

const tt = document.getElementById("tt")!;
const ttName = document.getElementById("tt-name")!;
const ttFam = document.getElementById("tt-fam")!;
const ttH2h = document.getElementById("tt-h2h")!;
const ttZoo = document.getElementById("tt-zoo")!;
const ttSrc = document.getElementById("tt-src")!;

let hoveredNode: string | null = null;

renderer.getMouseCaptor().on("mousemovebody", (e) => {
  const graphPos = renderer.viewportToGraph(e);
  const camera = renderer.getCamera();
  const ratio = camera.ratio;

  let closest: string | null = null;
  let closestDist = Infinity;

  graph.forEachNode((node, attrs) => {
    if (graph.getNodeAttribute(node, "hidden")) return;
    const dx = attrs.x - graphPos.x;
    const dy = attrs.y - graphPos.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const nodeScreenSize = attrs.size / ratio;
    const threshold = Math.max(nodeScreenSize * 0.015, 0.15);
    if (dist < threshold && dist < closestDist) {
      closestDist = dist;
      closest = node;
    }
  });

  if (closest !== hoveredNode) {
    if (hoveredNode) {
      graph.setNodeAttribute(hoveredNode, "color", graph.getNodeAttribute(hoveredNode, "originalColor"));
    }
    hoveredNode = closest;
    if (hoveredNode) {
      graph.setNodeAttribute(hoveredNode, "color", "#ffffff");
    }
    renderer.refresh();
  }

  if (closest) {
    const a = graph.getNodeAttributes(closest);
    ttName.textContent = a.fullLabel;
    ttFam.textContent = a.family;
    ttH2h.textContent = (a.h2h * 100).toFixed(1) + "%";
    ttZoo.textContent = (a.zoo * 100).toFixed(1) + "%";
    ttSrc.textContent = a.labelled ? "Ground truth (EID2)" : "Pseudo-label";
    tt.style.display = "block";
    tt.style.left = Math.min(e.x + 16, window.innerWidth - 260) + "px";
    tt.style.top = Math.min(e.y - 10, window.innerHeight - 180) + "px";
  } else {
    tt.style.display = "none";
  }
});

function setFilter(f: string) {
  document.querySelectorAll(".mode-btn").forEach((b) => {
    if (b.id !== "btn-edges") b.classList.remove("active");
  });
  document.getElementById(`btn-${f}`)!.classList.add("active");
  graph.forEachNode((node, attrs) => {
    let visible = true;
    if (f === "h2h") visible = attrs.h2h > 0.6;
    else if (f === "zoo") visible = attrs.zoo > 0.5;
    else if (f === "both") visible = attrs.h2h > 0.6 && attrs.zoo > 0.5;
    else if (f === "labelled") visible = attrs.labelled;
    graph.setNodeAttribute(node, "hidden", !visible);
  });
  graph.forEachEdge((edge, _attrs, src, tgt) => {
    graph.setEdgeAttribute(edge, "hidden",
      graph.getNodeAttribute(src, "hidden") || graph.getNodeAttribute(tgt, "hidden") || !edgesVisible
    );
  });
  const vis = graph.nodes().filter(n => !graph.getNodeAttribute(n, "hidden"));
  document.getElementById("count-label")!.textContent = `${vis.length.toLocaleString()} viruses visible`;
}

let edgesVisible = false;
function toggleEdges() {
  edgesVisible = !edgesVisible;
  document.getElementById("btn-edges")!.classList.toggle("active", edgesVisible);
  graph.forEachEdge((edge, _attrs, src, tgt) => {
    graph.setEdgeAttribute(edge, "hidden",
      !edgesVisible || graph.getNodeAttribute(src, "hidden") || graph.getNodeAttribute(tgt, "hidden")
    );
  });
}

document.getElementById("btn-all")!.addEventListener("click", () => setFilter("all"));
document.getElementById("btn-h2h")!.addEventListener("click", () => setFilter("h2h"));
document.getElementById("btn-zoo")!.addEventListener("click", () => setFilter("zoo"));
document.getElementById("btn-both")!.addEventListener("click", () => setFilter("both"));
document.getElementById("btn-labelled")!.addEventListener("click", () => setFilter("labelled"));
document.getElementById("btn-edges")!.addEventListener("click", toggleEdges);
