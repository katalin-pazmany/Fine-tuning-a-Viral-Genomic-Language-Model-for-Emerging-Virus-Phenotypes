"""
UMAP Visualisation of Vir2vec Embeddings
=========================================
Generates 2D UMAP projections of the Vir2vec embedding space,
coloured by phenotype labels (h2h and zoonotic) and virus family.

Outputs saved to outputs/results/umap_figures/

Usage:
    conda activate viral-phenotype
    pip install umap-learn
    python scripts/umap_visualisation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from umap import UMAP

BASE_DIR    = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "results" / "umap_figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_NPZ = BASE_DIR / "outputs" / "refseq_embeddings.npz"
METADATA_CSV   = BASE_DIR / "outputs" / "refseq_metadata.csv"
LABELS_XLSX    = BASE_DIR / "data" / "human_pathogens.xlsx"
CACHE_CSV      = BASE_DIR / "outputs" / "accession_taxid_cache.csv"

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading embeddings and labels...")
npz  = np.load(EMBEDDINGS_NPZ)
embs = npz["embeddings"]
accs = npz["accessions"]

cache_df     = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))

meta = pd.read_csv(METADATA_CSV)
meta["virus_taxid"] = meta["accession"].map(acc_to_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

human_df        = pd.read_excel(LABELS_XLSX, sheet_name="human")
interactions_df = pd.read_excel(LABELS_XLSX, sheet_name="interactions")

human_hosts = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
non_human   = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
human_df["zoonotic"] = human_df["virus_taxid"].isin(human_hosts & non_human).astype(int)

# ── 2. MATCH LABELS TO EMBEDDINGS ────────────────────────────────────────────
print("Matching labels to embeddings...")
acc_to_idx   = {a: i for i, a in enumerate(accs)}
label_lookup = human_df.set_index("virus_taxid")

matched_rows = []
for _, row in meta[meta["virus_taxid"].isin(set(human_df["virus_taxid"]))].iterrows():
    idx = acc_to_idx.get(row["accession"])
    if idx is not None:
        tid = row["virus_taxid"]
        matched_rows.append({
            "accession":    row["accession"],
            "virus_taxid":  tid,
            "emb_idx":      idx,
            "h2h":          label_lookup.loc[tid, "human_to_human"] if tid in label_lookup.index else np.nan,
            "zoonotic":     label_lookup.loc[tid, "zoonotic"] if tid in label_lookup.index else np.nan,
            "virus_family": label_lookup.loc[tid, "virus_family"] if tid in label_lookup.index else "Unknown",
        })

matched_df = pd.DataFrame(matched_rows).drop_duplicates(subset=["virus_taxid"])
matched_df = matched_df.dropna(subset=["h2h", "zoonotic"])
print(f"  Matched viruses for UMAP: {len(matched_df)}")

X_matched = embs[matched_df["emb_idx"].tolist()]

# ── 3. RUN UMAP ───────────────────────────────────────────────────────────────
print("Running UMAP (this may take a few minutes)...")
reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
               random_state=42, verbose=True)
X_2d = reducer.fit_transform(X_matched)
matched_df["umap_x"] = X_2d[:, 0]
matched_df["umap_y"] = X_2d[:, 1]
print("  UMAP complete!")

# ── 4. PLOT: H2H LABEL ───────────────────────────────────────────────────────
def plot_phenotype(df, label_col, title, fname, pos_label, neg_label,
                   pos_color="#e03131", neg_color="#1971c2"):
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    ax.scatter(neg["umap_x"], neg["umap_y"], c=neg_color, s=12, alpha=0.6,
               linewidths=0, label=neg_label)
    ax.scatter(pos["umap_x"], pos["umap_y"], c=pos_color, s=12, alpha=0.8,
               linewidths=0, label=pos_label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

plot_phenotype(
    matched_df, "h2h",
    "Vir2vec Embedding Space — Human-to-Human Transmissibility",
    "umap_h2h.png",
    pos_label=f"Transmissible (n={int((matched_df['h2h']==1).sum())})",
    neg_label=f"Non-transmissible (n={int((matched_df['h2h']==0).sum())})"
)

plot_phenotype(
    matched_df, "zoonotic",
    "Vir2vec Embedding Space — Zoonotic Spillover Potential",
    "umap_zoonotic.png",
    pos_label=f"Zoonotic (n={int((matched_df['zoonotic']==1).sum())})",
    neg_label=f"Non-zoonotic (n={int((matched_df['zoonotic']==0).sum())})"
)

# ── 5. PLOT: VIRUS FAMILY ────────────────────────────────────────────────────
print("Plotting by virus family...")
top_families = matched_df["virus_family"].value_counts().head(10).index.tolist()
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(10, 8))
other = matched_df[~matched_df["virus_family"].isin(top_families)]
ax.scatter(other["umap_x"], other["umap_y"], c="#cccccc", s=8, alpha=0.4,
           linewidths=0, label="Other")

for i, fam in enumerate(top_families):
    sub = matched_df[matched_df["virus_family"] == fam]
    ax.scatter(sub["umap_x"], sub["umap_y"], c=[colors[i]], s=15, alpha=0.8,
               linewidths=0, label=f"{fam} (n={len(sub)})")

ax.set_title("Vir2vec Embedding Space — Virus Family", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP 1", fontsize=11)
ax.set_ylabel("UMAP 2", fontsize=11)
ax.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "umap_family.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: umap_family.png")

print(f"\nAll UMAP figures saved to: {OUTPUTS_DIR}")
