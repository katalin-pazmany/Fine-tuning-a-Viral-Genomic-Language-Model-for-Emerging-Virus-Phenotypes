"""
Interactive 3D view of the Vir2vec embedding space (for presentation / defence)
==============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Projects the frozen per-taxon Vir2vec embeddings of the deduplicated, labelled
viruses into 3D with UMAP and renders a self-contained, rotatable Plotly scatter.

  colour     : virus family, using the SAME palette as the Gephi galaxy figure
               (dropdown re-colours by zoonotic label or pre-training exposure).
  node size  : how confidently Vir2vec predicts that virus's HUMAN-TO-HUMAN label
               under 5-fold cross-validation -- the predicted probability of the
               virus's TRUE class. Bigger node = the model reliably gets it right.
  hover      : virus name, family, phenotype, seen/unseen, predictability %.

Intended as an interactive companion to the static thesis figures -- rotate the
cloud in the viva to show family structure and where the model is (and isn't)
confident. Writes a self-contained file: outputs/figures_umap/embedding_3d.html
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
FIGDIR = OUT / "figures_umap"
FIGDIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# Neon family palette (Tron look): brightened versions of the Gephi galaxy hues,
# so families stay recognisable but glow on black. Rest -> dim slate.
FAM_HEX = {
    "Picornaviridae":   "#E64DFF",   # neon magenta
    "Papillomaviridae": "#4DFF88",   # neon green
    "Anelloviridae":    "#29D3FF",   # neon cyan
    "Adenoviridae":     "#FFFFFF",   # white
    "Peribunyaviridae": "#FFB020",   # neon amber
    "Flaviviridae":     "#FF476F",   # neon rose
    "Caliciviridae":    "#2BFFC6",   # neon teal
}
GREY = "#5A6470"


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
    name = [str(lab.loc[t, "virus"]) for t in tx]
    fam = [str(lab.loc[t, "virus_family"]) if isinstance(lab.loc[t, "virus_family"], str) else "Unassigned"
           for t in tx]
    zoo = [int(lab.loc[t, "zoonotic_strict"] or 0) for t in tx]
    sn = [seen.get(t, -1) for t in tx]
    y_h2h = np.array([int(lab.loc[t, "human_to_human"] or 0) for t in tx])

    # node size = 5-fold CV predicted probability of the TRUE human-to-human class
    print("cross-validating human-to-human predictability for node size...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    proba = cross_val_predict(clf, X, y_h2h, method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    p_true = proba[np.arange(len(tx)), y_h2h]          # prob assigned to correct label
    # predictability is heavily skewed (most viruses ~1.0), so a linear size map
    # looks uniform; rank-transform it to get a visible gradient (biggest = most
    # reliably predicted, smallest = the model's hardest cases)
    rank01 = p_true.argsort().argsort() / (len(p_true) - 1)
    size = 4.0 + 16.0 * rank01

    print(f"projecting {len(tx)} viruses into 3D with UMAP...")
    xyz = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                    metric="cosine", random_state=SEED).fit_transform(X)

    # k-nearest-neighbour edges (same graph as the galaxy) -> a glowing 3D web
    K = 6
    _, idx = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X).kneighbors(X)
    ex, ey, ez = [], [], []
    seen_e = set()
    for i in range(len(tx)):
        for j in idx[i][1:]:
            e = (min(i, j), max(i, j))
            if e in seen_e:
                continue
            seen_e.add(e)
            ex += [xyz[i, 0], xyz[j, 0], None]
            ey += [xyz[i, 1], xyz[j, 1], None]
            ez += [xyz[i, 2], xyz[j, 2], None]
    print(f"  {len(seen_e)} edges")

    POS, NEG = "#FF3D7F", "#29D3FF"          # phenotype positive / negative (neon)
    colors = {
        "Family": [FAM_HEX.get(f, GREY) for f in fam],
        "Human-to-human": [POS if h else NEG for h in y_h2h],
        "Zoonotic": [POS if z else NEG for z in zoo],
        "Pre-training exposure": ["#4A5568" if s == 1 else ("#FF47E0" if s == 0 else "#222833") for s in sn],
    }
    hover = [f"<b>{name[i]}</b><br>family: {fam[i]}<br>"
             f"zoonotic: {'yes' if zoo[i] else 'no'}<br>"
             f"seen by Vir2vec: {'yes' if sn[i] == 1 else ('no' if sn[i] == 0 else 'n/a')}<br>"
             f"H2H predictability: {p_true[i]*100:.0f}%"
             for i in range(len(tx))]

    def layer(sz, op, cols, hov):
        return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                            marker=dict(size=sz, color=cols, opacity=op, line=dict(width=0)),
                            text=hover if hov else None, hoverinfo="text" if hov else "none",
                            showlegend=False)

    base = colors["Family"]
    fig = go.Figure()
    # glowing neon web (drawn first / underneath the nodes)
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="#2CC8E0", width=1),
                               opacity=0.22, hoverinfo="none", showlegend=False))
    fig.add_trace(layer(size * 2.4, 0.05, base, False))   # outer glow  (trace 1)
    fig.add_trace(layer(size * 1.5, 0.09, base, False))   # inner glow  (trace 2)
    fig.add_trace(layer(size, 1.0, base, True))           # bright core (trace 3)

    # dropdown recolours the three node layers (traces 1,2,3); web stays constant
    buttons = [dict(label=k, method="restyle", args=[{"marker.color": [v, v, v]}, [1, 2, 3]])
               for k, v in colors.items()]

    axis = dict(showbackground=True, backgroundcolor="#03060d",
                gridcolor="#0e3a47", zerolinecolor="#124a5a",
                showspikes=False, showticklabels=False,
                title=dict(font=dict(color="#3fd0e0", size=11, family="monospace")))
    fig.update_layout(
        title=dict(text="VIR2VEC &#183; EMBEDDING SPACE &#47;&#47; 3D UMAP"
                        "<br><span style='font-size:11px;color:#4a8493'>colour = family &#183; "
                        "size = human-to-human predictability &#183; drag to rotate</span>",
                   font=dict(color="#63E6FF", size=17, family="monospace"), x=0.03, xanchor="left"),
        updatemenus=[dict(buttons=buttons, x=0.02, y=0.90, xanchor="left", yanchor="top",
                          bgcolor="#08151f", bordercolor="#1d5a6a",
                          font=dict(color="#8fe6ff", family="monospace"))],
        scene=dict(
            xaxis={**axis, "title": {"text": "UMAP-1"}},
            yaxis={**axis, "title": {"text": "UMAP-2"}},
            zaxis={**axis, "title": {"text": "UMAP-3"}},
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))),
        margin=dict(l=0, r=0, t=58, b=0), template="plotly_dark",
        paper_bgcolor="#02040a")

    out = FIGDIR / "embedding_3d.html"
    fig.write_html(out, include_plotlyjs="inline", full_html=True)
    print(f"saved {out}  ({out.stat().st_size // 1024} KB, self-contained)")
    print(f"H2H predictability: median {np.median(p_true)*100:.0f}%, "
          f"range {p_true.min()*100:.0f}-{p_true.max()*100:.0f}%")


if __name__ == "__main__":
    main()
