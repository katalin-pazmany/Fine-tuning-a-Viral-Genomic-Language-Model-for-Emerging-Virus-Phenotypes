"""
UMAP figure of the Vir2vec embedding space (thesis figure)
==========================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Projects the frozen per-taxon Vir2vec embeddings of the deduplicated, labelled
viruses into 2D with UMAP, and draws the SAME layout three ways: coloured by virus
family, by zoonotic (strict) label, and by whether Vir2vec saw the virus during
pre-training. Reading the three panels together is the visual form of the thesis:
the space organises by family, phenotype is smeared across those family clusters,
and the seen/unseen split shows why extrapolation is hard.

Writes NEW files: outputs/figures_umap/embedding_umap_panels.png/.pdf
(does not touch any existing figure).
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
FIGDIR = OUT / "figures_umap"
FIGDIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# colourblind-safe categorical palette (Okabe-Ito) for families
FAM_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
              "#D55E00", "#F0E442", "#000000", "#999999", "#7570B3"]
POS_C, NEG_C = "#D55E00", "#0072B2"          # phenotype
SEEN_C, UNSEEN_C = "#999999", "#CC79A7"      # exposure
BG = "#FFFFFF"


def per_taxon_embeddings():
    d = np.load(OUT / "refseq_embeddings.npz")
    emb, sid = d["embeddings"], list(d["accessions"])
    cache = pd.read_csv(OUT / "accession_taxid_cache.csv")
    c = dict(zip(cache["accession"], cache["taxid"]))
    meta = pd.read_csv(OUT / "refseq_metadata.csv")
    meta["tx"] = meta["accession"].map(c)
    meta = meta.dropna(subset=["tx"]); meta["tx"] = meta["tx"].astype(int)
    im = {a: i for i, a in enumerate(sid)}
    meta = meta[meta["accession"].isin(im)].copy()
    meta["ei"] = meta["accession"].map(im)
    t2i = {}
    for t, e in zip(meta["tx"], meta["ei"]):
        t2i.setdefault(int(t), []).append(int(e))
    return {t: emb[ix].mean(0) for t, ix in t2i.items()}


def main():
    emb = per_taxon_embeddings()
    lab = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv").set_index("virus_taxid")
    seen = {int(r["virus_taxid"]): int(r["vir2vec_seen"])
            for r in csv.DictReader(open(OUT / "vir2vec_seen_labels.csv"))}
    dedup = {int(x) for x in open(OUT / "dedup" / "h2h_zoo_deduplicated_taxids.txt") if x.strip()}

    tx = [t for t in lab.index if t in emb and t in dedup]
    X = np.vstack([emb[t] for t in tx])
    fam = np.array([str(lab.loc[t, "virus_family"]) if isinstance(lab.loc[t, "virus_family"], str)
                    else "Unassigned" for t in tx])
    zoo = np.array([int(lab.loc[t, "zoonotic_strict"] or 0) for t in tx])
    sn = np.array([seen.get(t, -1) for t in tx])
    print(f"projecting {len(tx)} viruses with UMAP...")

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=SEED).fit_transform(X)

    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(BG)

    # Panel A: family (top 9 by count, rest grey)
    top = [f for f, _ in pd.Series(fam).value_counts().head(9).items()]
    axes[0].scatter(xy[:, 0], xy[:, 1], s=6, c="#DDDDDD", linewidths=0)
    for i, f in enumerate(top):
        m = fam == f
        axes[0].scatter(xy[m, 0], xy[m, 1], s=9, c=FAM_COLORS[i % len(FAM_COLORS)],
                        label=f, linewidths=0, alpha=0.9)
    axes[0].set_title("(a) Coloured by virus family", fontweight="bold")
    axes[0].legend(fontsize=6, markerscale=1.4, loc="upper right", framealpha=0.8, ncol=1)

    # Panel B: zoonotic label
    axes[1].scatter(xy[zoo == 0, 0], xy[zoo == 0, 1], s=7, c=NEG_C, linewidths=0, alpha=0.7,
                    label="non-zoonotic")
    axes[1].scatter(xy[zoo == 1, 0], xy[zoo == 1, 1], s=7, c=POS_C, linewidths=0, alpha=0.85,
                    label="zoonotic")
    axes[1].set_title("(b) Coloured by zoonotic label", fontweight="bold")
    axes[1].legend(fontsize=7, markerscale=1.4, loc="upper right", framealpha=0.8)

    # Panel C: seen / unseen
    axes[2].scatter(xy[sn == 1, 0], xy[sn == 1, 1], s=7, c=SEEN_C, linewidths=0, alpha=0.7,
                    label="seen in pre-training")
    axes[2].scatter(xy[sn == 0, 0], xy[sn == 0, 1], s=7, c=UNSEEN_C, linewidths=0, alpha=0.85,
                    label="unseen")
    axes[2].set_title("(c) Coloured by pre-training exposure", fontweight="bold")
    axes[2].legend(fontsize=7, markerscale=1.4, loc="upper right", framealpha=0.8)

    fig.suptitle("Vir2vec embedding space (UMAP projection of frozen embeddings)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"embedding_umap_panels.{ext}", bbox_inches="tight", facecolor="white")
    print(f"saved {FIGDIR/'embedding_umap_panels.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
