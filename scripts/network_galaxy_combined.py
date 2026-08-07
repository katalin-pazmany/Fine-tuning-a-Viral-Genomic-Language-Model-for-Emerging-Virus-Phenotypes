"""
Combined "galaxy" of the whole Vir2vec viral embedding space (all phenotypes)
=============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

One overview network of every labelled, deduplicated virus across all three phenotype
sets (human-to-human / zoonotic + vector-borne, unioned), positioned by UMAP of their
frozen Vir2vec embeddings, linked to nearest neighbours, coloured by virus family, with
glow + legend. The single "here is the whole viral world the model sees, and it
organises it by family" figure. Phenotype-specific structure is carried by the separate
UMAP panels and results charts.

Writes: outputs/figures_umap/viral_network_galaxy_combined_legend.png
        outputs/viral_network_combined.gexf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import networkx as nx
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
SEED, K, N_FAM = 42, 6, 12
PALETTE = ["#00E676", "#FF9100", "#E040FB", "#00B0FF", "#FF1744", "#FFEA00",
           "#1DE9B6", "#FF80AB", "#B388FF", "#40C4FF", "#FFD180", "#69F0AE"]
GREY = "#6E6E6E"


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


def hex2rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def screen(x, y): return 1.0 - (1.0 - x) * (1.0 - y)


def bloom(rgb):
    hi = np.clip((rgb - 0.06) / 0.94, 0, 1)
    him = Image.fromarray((hi * 255).astype(np.uint8))
    out = rgb.copy()
    for r, g in [(4, 0.9), (12, 0.7), (30, 0.45)]:
        b = np.asarray(him.filter(ImageFilter.GaussianBlur(r))).astype(np.float32) / 255.0
        out = screen(out, np.clip(b * g, 0, 1))
    return np.clip(out, 0, 1)


def font(sz, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", sz, index=2 if bold else 7)
    except Exception:
        return ImageFont.load_default()


def family_map():
    fam = {}
    h = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv")
    for t, f in zip(h["virus_taxid"], h["virus_family"]):
        if isinstance(f, str) and f.strip():
            fam[int(t)] = f
    v = pd.read_csv(OUT / "vector_borne_virus_list_for_review.csv")
    for t, f in zip(v["virus_taxid"], v["virus_family"]):
        if int(t) not in fam and isinstance(f, str) and f.strip():
            fam[int(t)] = f
    return fam


def main():
    emb = per_taxon_embeddings()
    fam_all = family_map()
    dedup = set()
    for p in ("h2h_zoo_deduplicated_taxids.txt", "vector_borne_deduplicated_taxids.txt"):
        dedup |= {int(x) for x in open(OUT / "dedup" / p) if x.strip()}

    tx = [t for t in dedup if t in emb and t in fam_all]
    X = np.vstack([emb[t] for t in tx])
    fam = [fam_all[t] for t in tx]
    print(f"{len(tx)} viruses (union of all phenotype sets), {len(set(fam))} families")

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(X)
    _, idx = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X).kneighbors(X)

    counts = pd.Series(fam).value_counts()
    top = list(counts.head(N_FAM).index)
    fam_hex = {f: PALETTE[i] for i, f in enumerate(top)}
    node_rgb = np.array([hex2rgb(fam_hex.get(f, GREY)) for f in fam])
    deg = np.array([sum(1 for j in range(len(tx)) if (i in idx[j][1:] or j in idx[i][1:]))
                    for i in range(len(tx))], dtype=float)

    G = nx.Graph()
    for i, t in enumerate(tx):
        G.add_node(int(t), family=fam[i])
    for i, t in enumerate(tx):
        for j in idx[i][1:]:
            G.add_edge(int(t), int(tx[j]))
    for n, dd in G.degree():
        G.nodes[n]["degree"] = int(dd)
    nx.write_gexf(G, OUT / "viral_network_combined.gexf")

    W = H = 2800
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    fig.patch.set_facecolor("black"); ax.set_axis_off()
    seg_all, col_all, lens = [], [], []
    for i in range(len(tx)):
        for j in idx[i][1:]:
            seg_all.append([xy[i], xy[j]]); col_all.append((*node_rgb[i], 0.06))
            lens.append(float(np.linalg.norm(xy[i] - xy[j])))
    thr = np.percentile(lens, 98)          # drop the longest 2% (spikes to far outliers)
    segs = [s for s, l in zip(seg_all, lens) if l <= thr]
    cols = [c for c, l in zip(col_all, lens) if l <= thr]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.35))
    s = 3.5 + 24 * (deg - deg.min()) / (np.ptp(deg) + 1e-9)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=node_rgb, alpha=0.95, linewidths=0)
    # frame the dense bulk (a few far outliers otherwise squash everything into a corner)
    xlo, xhi = np.percentile(xy[:, 0], [2.5, 97.5]); ylo, yhi = np.percentile(xy[:, 1], [2.5, 97.5])
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = max(xhi - xlo, yhi - ylo) / 2 * 1.06
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    net = Image.fromarray((bloom(buf) * 255).astype(np.uint8)).convert("RGB")

    panel = int(net.width * 1.05)
    canvas = Image.new("RGB", (net.width + panel, net.height), (0, 0, 0))
    canvas.paste(net, (0, 0))
    d = ImageDraw.Draw(canvas)
    S = net.height; px = net.width + int(S * 0.05)
    d.multiline_text((px, int(S*0.09)), "The Vir2vec viral\nembedding space",
                     fill=(174, 190, 235), font=font(int(S*0.050), True), spacing=int(S*0.010))
    d.multiline_text((px, int(S*0.24)), f"{len(tx):,} human-associated viruses across all\n"
                     "phenotypes, linked to nearest neighbours\nand coloured by family",
                     fill=(150, 150, 160), font=font(int(S*0.025)), spacing=int(S*0.012))
    ly = int(S*0.42); sw = int(S*0.028); dy = int(S*0.0392)
    d.text((px, ly - int(S*0.05)), "Virus family", fill=(235, 235, 235), font=font(int(S*0.034), True))
    for f in top:
        c = tuple(int(255*v) for v in hex2rgb(fam_hex[f]))
        d.rounded_rectangle([px, ly, px+sw, ly+sw], radius=int(sw*0.22), fill=c)
        d.text((px+sw+int(S*0.016), ly-int(S*0.002)), f, fill=(235, 235, 235), font=font(int(S*0.028)))
        d.text((px+int(S*0.52), ly-int(S*0.002)), str(int(counts[f])), fill=(150, 150, 160), font=font(int(S*0.026)))
        ly += dy
    n_other = int(counts.iloc[N_FAM:].sum()) if len(counts) > N_FAM else 0
    if n_other:
        c = tuple(int(255*v) for v in hex2rgb(GREY))
        d.rounded_rectangle([px, ly, px+sw, ly+sw], radius=int(sw*0.22), fill=c)
        d.text((px+sw+int(S*0.016), ly-int(S*0.002)), "other families", fill=(235, 235, 235), font=font(int(S*0.028)))
        d.text((px+int(S*0.52), ly-int(S*0.002)), str(n_other), fill=(150, 150, 160), font=font(int(S*0.026)))

    out = FIGDIR / "viral_network_galaxy_combined_legend.png"
    canvas.save(out)
    print(f"saved {out}  {canvas.size}")


if __name__ == "__main__":
    main()
