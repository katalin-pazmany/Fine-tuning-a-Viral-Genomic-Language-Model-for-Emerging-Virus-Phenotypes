"""
"Galaxy" network of the Vir2vec embedding space for VECTOR-BORNE viruses
=======================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Vector-borne companion to network_galaxy_labeled.py, for visual parity with the
human-to-human / zoonotic galaxy. Nodes are the deduplicated vector-borne viruses,
positioned by UMAP of their frozen Vir2vec embeddings, linked to nearest neighbours,
coloured by virus family, with a glow + legend. Also writes a GEXF so the same network
can be laid out in Gephi if an exact style match to the h2h galaxy is wanted.

Writes: outputs/figures_umap/viral_network_galaxy_vectorborne_legend.png
        outputs/viral_network_vectorborne.gexf
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
SEED, K, N_FAM = 42, 6, 10
PALETTE = ["#00E676", "#FF9100", "#E040FB", "#00B0FF", "#FF1744",
           "#FFEA00", "#1DE9B6", "#FF80AB", "#B388FF", "#40C4FF"]
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
    p = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def main():
    emb = per_taxon_embeddings()
    lab = pd.read_csv(OUT / "vector_borne_virus_list_for_review.csv").set_index("virus_taxid")
    dedup = {int(x) for x in open(OUT / "dedup" / "vector_borne_deduplicated_taxids.txt") if x.strip()}
    tx = [t for t in lab.index if t in emb and t in dedup]
    X = np.vstack([emb[t] for t in tx])
    fam = [str(lab.loc[t, "virus_family"]) if isinstance(lab.loc[t, "virus_family"], str)
           else "Unassigned" for t in tx]
    print(f"{len(tx)} vector-borne viruses")

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(X)
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)

    counts = pd.Series(fam).value_counts()
    top = list(counts.head(N_FAM).index)
    fam_hex = {f: PALETTE[i] for i, f in enumerate(top)}
    node_rgb = np.array([hex2rgb(fam_hex.get(f, GREY)) for f in fam])
    deg = np.array([sum(1 for j in range(len(tx)) if (i in idx[j][1:] or j in idx[i][1:]))
                    for i in range(len(tx))], dtype=float)

    # GEXF for optional Gephi
    G = nx.Graph()
    for i, t in enumerate(tx):
        G.add_node(int(t), label=str(lab.loc[t, "virus_name"]) if isinstance(lab.loc[t, "virus_name"], str) else str(t),
                   family=fam[i], vector_borne=int(lab.loc[t, "vector_borne_to_humans"] or 0))
    for i, t in enumerate(tx):
        for j, dist in zip(idx[i][1:], _[i][1:]):
            G.add_edge(int(t), int(tx[j]), weight=round(float(1 - dist), 4))
    for n, dd in G.degree():
        G.nodes[n]["degree"] = int(dd)
    nx.write_gexf(G, OUT / "viral_network_vectorborne.gexf")

    # render
    W = H = 2600
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black")
    fig.patch.set_facecolor("black"); ax.set_axis_off()
    segs, cols = [], []
    for i in range(len(tx)):
        for j in idx[i][1:]:
            segs.append([xy[i], xy[j]]); cols.append((*node_rgb[i], 0.07))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4))
    s = 4 + 26 * (deg - deg.min()) / (np.ptp(deg) + 1e-9)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=node_rgb, alpha=0.95, linewidths=0)
    ax.autoscale(); ax.margins(0.03)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    # re-base the black render onto a dark-blue background, then bloom (matches combined/h2h)
    BG = np.array([10, 14, 40], np.float32) / 255.0
    base = BG[None, None, :] + buf * (1 - BG[None, None, :])
    him = Image.fromarray((np.clip(buf, 0, 1) * 255).astype(np.uint8))
    outb = base.copy()
    for _r, _g in [(4, 0.9), (12, 0.7), (30, 0.42)]:
        _b = np.asarray(him.filter(ImageFilter.GaussianBlur(_r))).astype(np.float32) / 255.0
        outb = screen(outb, np.clip(_b * _g, 0, 1))
    img = Image.fromarray((np.clip(outb, 0, 1) * 255).astype(np.uint8)).convert("RGB")

    # title + legend panel (matches the h2h titled galaxy)
    net = img
    panel = int(net.width * 1.05)
    canvas = Image.new("RGB", (net.width + panel, net.height), (10, 14, 40))
    canvas.paste(net, (0, 0))
    d = ImageDraw.Draw(canvas)
    S = net.height; px = net.width + int(S * 0.05)
    d.multiline_text((px, int(S*0.11)), "The Vir2vec viral\nembedding space\n(vector-borne)",
                     fill=(174, 190, 235), font=font(int(S*0.048), True), spacing=int(S*0.010))
    d.multiline_text((px, int(S*0.34)), f"{len(tx)} vector-borne viruses, linked to their\n"
                     "nearest neighbours and coloured by family",
                     fill=(150, 150, 160), font=font(int(S*0.026)), spacing=int(S*0.013))
    ly = int(S*0.48); sw = int(S*0.030); dy = int(S*0.0435)
    d.text((px, ly - int(S*0.052)), "Virus family", fill=(235, 235, 235), font=font(int(S*0.036), True))
    for f in top:
        c = tuple(int(255*v) for v in hex2rgb(fam_hex[f]))
        d.rounded_rectangle([px, ly, px+sw, ly+sw], radius=int(sw*0.22), fill=c)
        d.text((px+sw+int(S*0.018), ly-int(S*0.002)), f, fill=(235, 235, 235), font=font(int(S*0.030)))
        d.text((px+int(S*0.52), ly-int(S*0.002)), str(int(counts[f])), fill=(150, 150, 160), font=font(int(S*0.028)))
        ly += dy
    n_other = int(counts.iloc[N_FAM:].sum()) if len(counts) > N_FAM else 0
    if n_other:
        c = tuple(int(255*v) for v in hex2rgb(GREY))
        d.rounded_rectangle([px, ly, px+sw, ly+sw], radius=int(sw*0.22), fill=c)
        d.text((px+sw+int(S*0.018), ly-int(S*0.002)), "other families", fill=(235, 235, 235), font=font(int(S*0.030)))
        d.text((px+int(S*0.52), ly-int(S*0.002)), str(n_other), fill=(150, 150, 160), font=font(int(S*0.028)))

    out = FIGDIR / "viral_network_galaxy_vectorborne_legend.png"
    canvas.save(out)
    print(f"saved {out}  {canvas.size}")
    print(f"saved {OUT/'viral_network_vectorborne.gexf'} (for optional Gephi)")


if __name__ == "__main__":
    main()
