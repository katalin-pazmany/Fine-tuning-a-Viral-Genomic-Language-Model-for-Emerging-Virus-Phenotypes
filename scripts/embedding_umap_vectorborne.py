"""
UMAP figure of the Vir2vec embedding space for VECTOR-BORNE viruses
===================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Companion to embedding_umap_figure.py, for vector-borne-to-humans. Three panels:
by virus family, by vector-borne-to-humans label, and by vector group.

Renders a white version (name.png/.pdf) for the dissertation and a dark-blue version
(name_dark.png) matching the galaxy figures for slides.
"""
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
plt.rcParams["font.family"] = "Arial"

THEMES = {
    "": dict(bg="white", fg="black", bgdot="#DDDDDD", lface="white", lalpha=0.8,
             fam=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000", "#999999", "#7570B3"],
             pos="#D55E00", neg="#0072B2", grp=["#009E73", "#D55E00", "#0072B2", "#CC79A7", "#999999"], exts=("png", "pdf")),
    "_dark": dict(bg="#0A0E28", fg="#E8EAF2", bgdot="#2A2E48", lface="#12162E", lalpha=0.9,
                  fam=["#4DA6E8", "#FFB020", "#2FD48F", "#E48FC0", "#7FD0F5", "#FF8A3D", "#FFE84D", "#FFFFFF", "#AAB0C4", "#B0A8E8"],
                  pos="#FF8A3D", neg="#4DA6E8", grp=["#2FD48F", "#FF8A3D", "#4DA6E8", "#E48FC0", "#AAB0C4"], exts=("png",)),
}


def per_taxon_embeddings():
    d = np.load(OUT / "refseq_embeddings.npz")
    emb, sid = d["embeddings"], list(d["accessions"])
    cache = pd.read_csv(OUT / "accession_taxid_cache.csv")
    c = dict(zip(cache["accession"], cache["taxid"]))
    strip = {str(a).split(".")[0]: t for a, t in c.items()}
    meta = pd.read_csv(OUT / "refseq_metadata.csv")
    meta["tx"] = meta["accession"].map(lambda a: c.get(a) or strip.get(str(a).split(".")[0]))
    meta = meta.dropna(subset=["tx"]); meta["tx"] = meta["tx"].astype(int)
    im = {a: i for i, a in enumerate(sid)}
    meta = meta[meta["accession"].isin(im)].copy()
    meta["ei"] = meta["accession"].map(im)
    t2i = {}
    for t, e in zip(meta["tx"], meta["ei"]):
        t2i.setdefault(int(t), []).append(int(e))
    return {t: emb[ix].mean(0) for t, ix in t2i.items()}


def style_legend(ax, th, fs=7):
    leg = ax.legend(fontsize=fs, markerscale=1.4, loc="upper right", framealpha=th["lalpha"])
    leg.get_frame().set_facecolor(th["lface"]); leg.get_frame().set_edgecolor(th["fg"])
    for t in leg.get_texts():
        t.set_color(th["fg"])


def render(xy, fam, vb, grp, top, gtop, suffix, th):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor(th["bg"])
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(th["bg"])

    axes[0].scatter(xy[:, 0], xy[:, 1], s=6, c=th["bgdot"], linewidths=0)
    for i, f in enumerate(top):
        m = fam == f
        axes[0].scatter(xy[m, 0], xy[m, 1], s=9, c=th["fam"][i % len(th["fam"])], label=f, linewidths=0, alpha=0.9)
    axes[0].set_title("(a) Coloured by virus family", fontweight="bold", color=th["fg"])
    style_legend(axes[0], th, fs=6)

    axes[1].scatter(xy[vb == 0, 0], xy[vb == 0, 1], s=7, c=th["neg"], linewidths=0, alpha=0.7, label="not vector-borne to humans")
    axes[1].scatter(xy[vb == 1, 0], xy[vb == 1, 1], s=10, c=th["pos"], linewidths=0, alpha=0.9, label="vector-borne to humans")
    axes[1].set_title("(b) Coloured by vector-borne label", fontweight="bold", color=th["fg"])
    style_legend(axes[1], th)

    for i, g in enumerate(gtop):
        m = grp == g
        axes[2].scatter(xy[m, 0], xy[m, 1], s=8, c=th["grp"][i % len(th["grp"])], label=g, linewidths=0, alpha=0.85)
    axes[2].set_title("(c) Coloured by vector group", fontweight="bold", color=th["fg"])
    style_legend(axes[2], th)

    fig.suptitle("Vir2vec embedding space — vector-borne viruses (UMAP projection)",
                 fontweight="bold", y=1.01, color=th["fg"])
    fig.tight_layout()
    for ext in th["exts"]:
        fig.savefig(FIGDIR / f"embedding_umap_vectorborne{suffix}.{ext}", bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)


def main():
    emb = per_taxon_embeddings()
    lab = pd.read_csv(OUT / "vector_borne_virus_list_for_review.csv").set_index("virus_taxid")
    dedup = {int(x) for x in open(OUT / "dedup" / "vector_borne_deduplicated_taxids.txt") if x.strip()}
    tx = [t for t in lab.index if t in emb and t in dedup]
    X = np.vstack([emb[t] for t in tx])
    fam = np.array([str(lab.loc[t, "virus_family"]) if isinstance(lab.loc[t, "virus_family"], str) else "Unassigned" for t in tx])
    vb = np.array([int(lab.loc[t, "vector_borne_to_humans"] or 0) for t in tx])
    grp = np.array([str(lab.loc[t, "vector_groups"]).split(";")[0].split(",")[0].strip().lower()
                    if isinstance(lab.loc[t, "vector_groups"], str) else "unknown" for t in tx])
    print(f"projecting {len(tx)} vector-borne viruses with UMAP...")
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(X)
    top = [f for f, _ in pd.Series(fam).value_counts().head(9).items()]
    gtop = [g for g, _ in pd.Series(grp).value_counts().head(5).items()]
    for suffix, th in THEMES.items():
        render(xy, fam, vb, grp, top, gtop, suffix, th)
    print(f"saved embedding_umap_vectorborne.png/.pdf (white) + _dark.png (navy), {len(tx)} viruses")


if __name__ == "__main__":
    main()
