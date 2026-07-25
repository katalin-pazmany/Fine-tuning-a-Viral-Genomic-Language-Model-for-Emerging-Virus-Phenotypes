"""
"Galaxy" network of the Vir2vec embedding space, coloured by virus family + legend
==================================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Same content as the Gephi render, but coloured *strictly by virus family* (so a
legend is truthful) and generated end-to-end in Python for reproducibility:

  layout   : UMAP(2D) of the frozen per-taxon Vir2vec embeddings (same seed as the
             thesis UMAP figure), so node positions match the analysis.
  edges    : each virus linked to its K nearest neighbours in cosine space, drawn
             faint + additive on black so overlaps build luminous cores.
  glow     : multi-scale Gaussian-blur bloom, Screen-composited over the sharp render.
  legend   : family -> colour swatches, drawn AFTER the bloom so text stays crisp.

Writes: outputs/figures_umap/viral_network_galaxy_legend.png (+ a no-legend variant).
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageFilter, ImageDraw, ImageFont

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
FIGDIR = OUT / "figures_umap"
FIGDIR.mkdir(parents=True, exist_ok=True)
SEED = 42
K = 6
N_FAM = 10                       # families that get their own colour; rest -> grey

# bright, saturated, well-separated on black
PALETTE = ["#00E676", "#FF9100", "#E040FB", "#00B0FF", "#FF1744",
           "#FFEA00", "#1DE9B6", "#FF80AB", "#B388FF", "#40C4FF"]
GREY = "#6E6E6E"


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


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def screen(x, y):
    return 1.0 - (1.0 - x) * (1.0 - y)


def bloom(rgb):
    """rgb: float array HxWx3 in [0,1] on black -> add multi-scale halo."""
    hi = np.clip((rgb - 0.06) / 0.94, 0, 1)
    hi_im = Image.fromarray((hi * 255).astype(np.uint8))
    out = rgb.copy()
    for radius, gain in [(4, 0.9), (12, 0.7), (30, 0.45)]:
        g = np.asarray(hi_im.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0
        out = screen(out, np.clip(g * gain, 0, 1))
    return np.clip(out, 0, 1)


def main():
    emb = per_taxon_embeddings()
    lab = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv").set_index("virus_taxid")
    dedup = {int(x) for x in open(OUT / "dedup" / "h2h_zoo_deduplicated_taxids.txt") if x.strip()}

    tx = [t for t in lab.index if t in emb and t in dedup]
    X = np.vstack([emb[t] for t in tx])
    fam = np.array([str(lab.loc[t, "virus_family"]) if isinstance(lab.loc[t, "virus_family"], str)
                    else "Unassigned" for t in tx])
    print(f"{len(tx)} viruses")

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=SEED).fit_transform(X)

    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)

    # family -> colour (top N by count get a palette colour; rest grey)
    counts = pd.Series(fam).value_counts()
    top = list(counts.head(N_FAM).index)
    fam_hex = {f: PALETTE[i] for i, f in enumerate(top)}
    node_rgb = np.array([hex2rgb(fam_hex.get(f, GREY)) for f in fam])
    deg = np.array([len(set(idx[i][1:]) | {j for j in range(len(tx)) if i in idx[j][1:]})
                    for i in range(len(tx))], dtype=float)

    # --- render network on black ---
    W, H = 2600, 2600
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    fig.patch.set_facecolor("black"); ax.set_axis_off()

    segs, cols = [], []
    for i in range(len(tx)):
        for j in idx[i][1:]:
            segs.append([xy[i], xy[j]])
            cols.append((*node_rgb[i], 0.07))          # source-family colour, faint
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4))

    s = 4 + 26 * (deg - deg.min()) / (deg.ptp() + 1e-9)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=node_rgb, alpha=0.95, linewidths=0)
    ax.autoscale(); ax.margins(0.03)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)

    glowed = (bloom(buf) * 255).astype(np.uint8)
    Image.fromarray(glowed).save(FIGDIR / "viral_network_galaxy_nolegend.png")

    # --- draw legend AFTER bloom so it stays crisp ---
    img = Image.fromarray(glowed).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        f_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
        f_item = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except Exception:
        f_title = f_item = ImageFont.load_default()

    x0, y0, dy, sw = 60, 60, 52, 34
    d.text((x0, y0), "Virus family", fill="white", font=f_title)
    y = y0 + 64
    for f in top:
        r, g, b = [int(255 * v) for v in hex2rgb(fam_hex[f])]
        d.rectangle([x0, y + 6, x0 + sw, y + 6 + sw], fill=(r, g, b))
        d.text((x0 + sw + 16, y), f"{f}  ({counts[f]})", fill="white", font=f_item)
        y += dy
    n_other = int(counts.iloc[N_FAM:].sum()) if len(counts) > N_FAM else 0
    if n_other:
        r, g, b = [int(255 * v) for v in hex2rgb(GREY)]
        d.rectangle([x0, y + 6, x0 + sw, y + 6 + sw], fill=(r, g, b))
        d.text((x0 + sw + 16, y), f"other families  ({n_other})", fill="white", font=f_item)

    out = FIGDIR / "viral_network_galaxy_legend.png"
    img.save(out)
    print(f"saved {out}  {img.size}")


if __name__ == "__main__":
    main()
