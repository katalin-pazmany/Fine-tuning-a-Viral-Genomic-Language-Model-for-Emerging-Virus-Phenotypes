"""
Export Virus Network to Gephi GEXF Format
============================================
Converts virus_network_data.json into a GEXF file that Maya can
open directly in Gephi for custom visualisation and layout tuning.

GEXF (Graph Exchange XML Format) is Gephi's native format and
supports node/edge attributes, colours, and sizes.

Usage:
    conda activate viral-phenotype
    python scripts/export_to_gephi.py

Output:
    outputs/viral_network.gexf
"""

import json
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "outputs" / "virus_network_data.json"
OUT_PATH  = BASE_DIR / "outputs" / "viral_network.gexf"

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading network data...")
with open(DATA_PATH) as f:
    data = json.load(f)

nodes = data["nodes"]
edges = data["edges"]
print(f"  Nodes: {len(nodes)}")
print(f"  Edges: {len(edges)}")

# ── 2. BUILD NETWORKX GRAPH ───────────────────────────────────────────────────
print("\nBuilding graph...")
G = nx.Graph()

for n in nodes:
    # GEXF requires hex colour without '#' converted to RGB
    color_hex = n["color"].lstrip("#")
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)

    G.add_node(
        n["id"],
        label=n.get("label") or f"Taxid {n['id']}",
        family=n.get("family", "Unknown"),
        h2h_risk=float(n["h2h"]),
        zoonotic_risk=float(n["zoo"]),
        labelled=bool(n["labelled"]),
        famous=bool(n["famous"]),
        viz={"color": {"r": r, "g": g, "b": b}, "size": float(n["size"])},
        x=float(n["x"]) * 10,   # scale up for better Gephi default view
        y=float(n["y"]) * 10,
    )

for e in edges:
    if G.has_node(e["source"]) and G.has_node(e["target"]) and not G.has_edge(e["source"], e["target"]):
        G.add_edge(e["source"], e["target"], weight=float(e["weight"]))

print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 3. EXPORT TO GEXF ──────────────────────────────────────────────────────────
print("\nExporting to GEXF...")
nx.write_gexf(G, OUT_PATH)

size_mb = OUT_PATH.stat().st_size / 1e6
print(f"\nSaved: {OUT_PATH}")
print(f"  File size: {size_mb:.1f} MB")
print("\nDone! ✓")
print("\nTo open in Gephi:")
print("  1. Open Gephi")
print("  2. File > Open > select viral_network.gexf")
print("  3. Node positions, colours, and sizes are pre-set")
print("  4. Try Gephi's ForceAtlas2 layout for a refined circular layout")
