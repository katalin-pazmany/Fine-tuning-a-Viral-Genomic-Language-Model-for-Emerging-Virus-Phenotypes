"""
Build a Gephi network of the Vir2vec embedding space (for figures)
==================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Produces a compact, on-message network for Gephi from the SAME per-taxon Vir2vec
embeddings used in the analysis. Nodes are the deduplicated, labelled viruses;
edges connect each virus to its k nearest neighbours in embedding space (cosine
similarity), so a force-directed layout (ForceAtlas2 in Gephi) clusters viruses
the model represents as similar. Every node carries attributes that can be used to
colour the layout: virus family, the three phenotype labels, and whether the virus
was seen by Vir2vec during pre-training.

The intended figure: lay the graph out once, then colour it (i) by family and
(ii) by a phenotype. If it clusters cleanly by family but the phenotype is smeared
across those clusters, that is the visual form of the dissertation's central point
--- the representation organises viruses by taxonomy, and phenotype rides along.

Writes a NEW file: outputs/viral_network_labelled.gexf (does not touch the existing
outputs/viral_network.gexf).
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
K = 6            # nearest neighbours per node
SEED = 42
OUT_FILE = OUT / "viral_network_labelled.gexf"


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
    seqlen = dict(zip(meta["accession"], meta["seq_length"]))
    t2i, t2len = {}, {}
    for a, t, e in zip(meta["accession"], meta["tx"], meta["ei"]):
        t2i.setdefault(int(t), []).append(int(e))
        t2len[int(t)] = max(t2len.get(int(t), 0), int(seqlen.get(a, 0)))
    return {t: emb[ix].mean(0) for t, ix in t2i.items()}, t2len


def main():
    emb, seqlen = per_taxon_embeddings()
    lab = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv").set_index("virus_taxid")
    seen = {int(r["virus_taxid"]): int(r["vir2vec_seen"])
            for r in csv.DictReader(open(OUT / "vir2vec_seen_labels.csv"))}
    dedup = {int(x) for x in open(OUT / "dedup" / "h2h_zoo_deduplicated_taxids.txt") if x.strip()}

    tx = [t for t in lab.index if t in emb and t in dedup]
    X = np.vstack([emb[t] for t in tx])
    print(f"nodes: {len(tx)} labelled, deduplicated viruses with embeddings")

    # k-nearest-neighbour graph in cosine space
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)

    G = nx.Graph()
    for i, t in enumerate(tx):
        row = lab.loc[t]
        fam = row.get("virus_family")
        fam = str(fam) if isinstance(fam, str) and fam.strip() else "Unassigned"
        G.add_node(int(t),
                   label=str(row.get("virus", "")) or str(t),
                   family=fam,
                   genus=str(row.get("virus_genus", "")) if isinstance(row.get("virus_genus"), str) else "",
                   h2h=int(row.get("human_to_human", 0) or 0),
                   zoonotic_relaxed=int(row.get("zoonotic_relaxed", 0) or 0),
                   zoonotic_strict=int(row.get("zoonotic_strict", 0) or 0),
                   vir2vec_seen=int(seen.get(int(t), -1)),
                   genome_length=int(seqlen.get(int(t), 0)))
    for i, t in enumerate(tx):
        for j, sim in zip(idx[i][1:], dist[i][1:]):     # skip self (first neighbour)
            u, v = int(t), int(tx[j])
            w = float(1.0 - sim)                          # cosine similarity as weight
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=round(w, 4))

    # degree as an extra size attribute
    for n, d in G.degree():
        G.nodes[n]["degree"] = int(d)

    nx.write_gexf(G, OUT_FILE)
    print(f"edges: {G.number_of_edges()} (k={K} nearest neighbours, cosine)")
    print(f"families represented: {len(set(nx.get_node_attributes(G, 'family').values()))}")
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
