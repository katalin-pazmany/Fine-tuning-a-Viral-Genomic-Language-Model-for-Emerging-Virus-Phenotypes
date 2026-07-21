"""
Build virus_network_data.json (node/edge graph for the web viz + Gephi export)
==============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Reconstructs the network-viz/Gephi data file from the current embedding set.
The original one-off generator was never committed, so this rebuilds it
faithfully to the existing schema (verified against the 7-Jul file):

  nodes: id, label, family, x, y, size, color, h2h, zoo, labelled, famous
  edges: source, target, weight   (weight = cosine similarity)

Node layout (x, y)   : 2-D UMAP of the per-taxon mean-pooled Vir2vec embeddings
Node risk (h2h, zoo) : Tier-1 SVM (RBF, frozen embeddings) predicted probability
Node colour          : h2h >= 0.60 and/or zoo >= 0.50 ->
                         both -> #a855f7, h2h -> #f97316, zoo -> #10b981, low -> #3b82f6
Node size            : famous -> 8/10/12 (curated), labelled -> 4, else 2
Edges                : per-node cosine k-NN, kept when similarity >= --edge-cutoff

The two knobs Maya wants to experiment with are exposed as flags:
    --edge-cutoff   minimum cosine similarity for an edge to be drawn (default 0.50)
    --k             neighbours considered per node before the cutoff (default 8)

Usage (run where the FULL refseq_embeddings.npz lives, i.e. Barkla):
    conda activate viral-phenotype
    python scripts/build_network_data.py --edge-cutoff 0.55 --k 8

    # on Barkla, via the CPU submit wrapper (never the login node):
    sbatch scripts/slurm/submit_cpu_analysis.sh scripts/build_network_data.py --edge-cutoff 0.55

Outputs:
    outputs/virus_network_data.json          (copy into network-viz/src/ for the web viz)
    then:  python scripts/export_to_gephi.py  ->  outputs/viral_network.gexf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--edge-cutoff", type=float, default=0.50,
                    help="Minimum cosine similarity for an edge (Maya's cutoff knob).")
parser.add_argument("--k", type=int, default=8,
                    help="Neighbours considered per node before applying the cutoff.")
parser.add_argument("--umap-neighbors", type=int, default=15)
parser.add_argument("--umap-min-dist", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

CACHE_CSV = OUTPUTS_DIR / "accession_taxid_cache.csv"
METADATA_CSV = OUTPUTS_DIR / "refseq_metadata.csv"
EMBEDDINGS_NPZ = OUTPUTS_DIR / "refseq_embeddings.npz"
H2H_XLSX = BASE_DIR / "data" / "complete_h2h_labels.xlsx"
ZOO_XLSX = BASE_DIR / "data" / "complete_zoonotic_labels.xlsx"
OUT_JSON = OUTPUTS_DIR / "virus_network_data.json"

# Colour scheme (matches network-viz/src/main.ts legend).
COL_BOTH, COL_H2H, COL_ZOO, COL_LOW = "#a855f7", "#f97316", "#10b981", "#3b82f6"
H2H_THRESHOLD, ZOO_THRESHOLD = 0.60, 0.50

# Curated "famous" viruses with hand-assigned display sizes (from the original file).
FAMOUS = {
    10244: 8,    # monkeypox virus
    11082: 8,    # west nile virus
    11089: 8,    # yellow fever virus
    11137: 8,    # human coronavirus 229e
    11320: 12,   # influenza a virus
    11520: 10,   # influenza b virus
    11676: 12,   # human immunodeficiency virus 1
    12637: 10,   # dengue virus
    64320: 10,   # zika virus
    277944: 8,   # human coronavirus nl63
    333760: 8,   # human papillomavirus 16
}


# ── 1. LOAD EMBEDDINGS + METADATA + LABELS ────────────────────────────────────
print("Loading cache, metadata and embeddings...")
cache = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache["accession"], cache["taxid"].astype(int)))
stripped_to_taxid = {}
for acc_ver, tid in acc_to_taxid.items():
    stripped_to_taxid.setdefault(str(acc_ver).split(".")[0], tid)

def resolve_taxid(acc):
    return acc_to_taxid.get(acc) or stripped_to_taxid.get(str(acc).split(".")[0])

meta = pd.read_csv(METADATA_CSV)
meta["virus_taxid"] = meta["accession"].map(resolve_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

npz = np.load(EMBEDDINGS_NPZ)
embs, accs = npz["embeddings"], npz["accessions"]
acc_to_idx = {a: i for i, a in enumerate(accs.tolist())}
print(f"  Embeddings: {embs.shape}  |  metadata rows: {len(meta)}")

h2h_df = pd.read_excel(H2H_XLSX)
zoo_df = pd.read_excel(ZOO_XLSX)
labels_df = h2h_df.merge(zoo_df[["virus_taxid", "zoonotic"]], on="virus_taxid", how="left")
labels_df = labels_df.dropna(subset=["virus_family"])
labels_df["virus_family"] = labels_df["virus_family"].astype(str)
label_lookup = labels_df.set_index("virus_taxid")
labelled_taxids = set(labels_df["virus_taxid"])
# virus name column (for node labels), if present
name_col = next((c for c in ("virus", "virus_name", "species") if c in labels_df.columns), None)

# ── 2. AGGREGATE EMBEDDINGS PER TAXID (mean pool) ─────────────────────────────
print("Aggregating embeddings per taxid...")
taxid_vecs = {}
for _, row in meta.iterrows():
    idx = acc_to_idx.get(row["accession"])
    if idx is not None:
        taxid_vecs.setdefault(row["virus_taxid"], []).append(embs[idx])

taxids = sorted(taxid_vecs.keys())
X = np.vstack([np.mean(taxid_vecs[t], axis=0) for t in taxids]).astype(np.float32)
print(f"  Unique taxa (nodes): {len(taxids)}")

# ── 3. TIER-1 SVM RISK SCORES (fit on labelled subset, predict for all) ──────
print("Fitting Tier-1 SVMs for h2h / zoonotic risk...")
lab_idx = [i for i, t in enumerate(taxids) if t in label_lookup.index]
X_lab = X[lab_idx]
y_h2h = np.array([int(label_lookup.loc[taxids[i], "human_to_human"]) for i in lab_idx])
y_zoo = np.array([int(label_lookup.loc[taxids[i], "zoonotic"]) for i in lab_idx])

def make_svm():
    return Pipeline([("s", StandardScaler()),
                     ("c", SVC(kernel="rbf", probability=True,
                               class_weight="balanced", random_state=args.seed))])

clf_h2h = make_svm().fit(X_lab, y_h2h)
clf_zoo = make_svm().fit(X_lab, y_zoo)
proba_h2h = clf_h2h.predict_proba(X)[:, 1]
proba_zoo = clf_zoo.predict_proba(X)[:, 1]

# ── 4. UMAP LAYOUT ────────────────────────────────────────────────────────────
print("Running UMAP (2-D layout)...")
from umap import UMAP
reducer = UMAP(n_components=2, n_neighbors=args.umap_neighbors,
               min_dist=args.umap_min_dist, metric="cosine", random_state=args.seed)
XY = reducer.fit_transform(X)

# ── 5. BUILD NODES ────────────────────────────────────────────────────────────
def node_color(h, z):
    if h >= H2H_THRESHOLD and z >= ZOO_THRESHOLD:
        return COL_BOTH
    if h >= H2H_THRESHOLD:
        return COL_H2H
    if z >= ZOO_THRESHOLD:
        return COL_ZOO
    return COL_LOW

def node_size(tid, is_labelled):
    if tid in FAMOUS:
        return FAMOUS[tid]
    return 4 if is_labelled else 2

nodes = []
for i, tid in enumerate(taxids):
    is_lab = tid in labelled_taxids
    fam = str(label_lookup.loc[tid, "virus_family"]) if tid in label_lookup.index else "Unknown"
    label = ""
    if name_col and tid in label_lookup.index:
        val = label_lookup.loc[tid, name_col]
        label = "" if pd.isna(val) else str(val)
    nodes.append({
        "id": str(tid),
        "label": label,
        "family": fam,
        "x": round(float(XY[i, 0]), 3),
        "y": round(float(XY[i, 1]), 3),
        "size": node_size(tid, is_lab),
        "color": node_color(proba_h2h[i], proba_zoo[i]),
        "h2h": round(float(proba_h2h[i]), 3),
        "zoo": round(float(proba_zoo[i]), 3),
        "labelled": bool(is_lab),
        "famous": tid in FAMOUS,
    })

# ── 6. BUILD EDGES (cosine k-NN, thresholded) ────────────────────────────────
print(f"Building edges (k={args.k}, cutoff={args.edge_cutoff})...")
nn = NearestNeighbors(n_neighbors=min(args.k + 1, len(taxids)), metric="cosine")
nn.fit(X)
dist, nbr = nn.kneighbors(X)   # cosine distance = 1 - cosine similarity
edges = []
seen = set()
for i in range(len(taxids)):
    for j_pos in range(1, nbr.shape[1]):     # skip self (column 0)
        j = int(nbr[i, j_pos])
        sim = 1.0 - float(dist[i, j_pos])
        if sim < args.edge_cutoff:
            continue
        a, b = sorted((i, j))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append({"source": str(taxids[a]), "target": str(taxids[b]),
                      "weight": round(sim, 3)})

print(f"  Nodes: {len(nodes)}  |  Edges: {len(edges)}")

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
with open(OUT_JSON, "w") as f:
    json.dump({"nodes": nodes, "edges": edges}, f)
print(f"Saved: {OUT_JSON}")
print("Next: copy to network-viz/src/virus_network_data.json for the web viz, "
      "and run scripts/export_to_gephi.py to produce the GEXF for Gephi.")
