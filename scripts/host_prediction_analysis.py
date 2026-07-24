"""
Host-group prediction from Vir2vec embeddings (exploratory / supplementary)
==========================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Follow-on suggested by the project supervisor: rather than predicting human-relevant
phenotypes, can the frozen Vir2vec representation predict which *host group* a virus
is known to infect? Host association is plausibly more intrinsic to the virus, and
less confounded by human-research effort, than zoonotic status.

Binary tasks (does the virus infect any host in the group?) are derived from the
VIRION virus--host association edgelist, with host taxa resolved to taxonomic class
via the NCBI taxonomy. Only the reservoir groups with a usable class balance are run:
birds (Aves), bats (Chiroptera) and rodents (Rodentia).

Each task is evaluated exactly as in the main analysis: frozen Vir2vec embeddings with
the four Tier-1 classifiers, against the composition baselines (GC+length, k-mer,
codon usage), under stratified 5-fold cross-validation (AUROC and AUPRC).

Writes NEW files only (nothing existing is overwritten):
    outputs/host_labels.csv               -- per-virus host-group labels
    outputs/host_prediction_results.json  -- results

Prerequisites: the NCBI taxonomy dump (nodes.dmp, names.dmp, merged.dmp). By default
these are read from a local scratch directory; set TAXDUMP_DIR to point elsewhere.
"""

import csv
import itertools
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
csv.field_size_limit(10 ** 7)
SEED = 42
BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
TAXDUMP_DIR = Path(os.environ.get(
    "TAXDUMP_DIR",
    "/private/tmp/claude-501/-Users-puszedli-viral-phenotype-finetuning/"
    "6c6f82a5-2bf6-42bf-84df-2cc23e5b906a/scratchpad"))

# Reservoir groups with a usable class balance (name -> NCBI taxon name to match).
HOST_GROUPS = {"birds": "Aves", "bats": "Chiroptera", "rodents": "Rodentia"}


# ── taxonomy ──────────────────────────────────────────────────────────────────
def load_taxonomy():
    parent, name, merged = {}, {}, {}
    for line in open(TAXDUMP_DIR / "nodes.dmp"):
        p = [x.strip() for x in line.split("|")]
        parent[int(p[0])] = int(p[1])
    for line in open(TAXDUMP_DIR / "merged.dmp"):
        p = [x.strip() for x in line.split("|") if x.strip()]
        if len(p) >= 2:
            merged[int(p[0])] = int(p[1])
    for line in open(TAXDUMP_DIR / "names.dmp"):
        p = [x.strip() for x in line.split("|")]
        if len(p) > 3 and p[3] == "scientific name":
            name[int(p[0])] = p[1]
    return parent, name, merged


def make_lineage_test(parent, name, merged):
    cache = {}
    def has_ancestor(taxid, target):
        key = (taxid, target)
        if key in cache:
            return cache[key]
        t = merged.get(taxid, taxid)
        for _ in range(60):
            if name.get(t) == target:
                cache[key] = True
                return True
            if t not in parent or parent[t] == t:
                cache[key] = False
                return False
            t = parent[t]
        cache[key] = False
        return False
    return has_ancestor


# ── features ──────────────────────────────────────────────────────────────────
def per_taxon_embeddings():
    d = np.load(OUT / "refseq_embeddings.npz")
    emb, sid = d["embeddings"], list(d["accessions"])
    cache = pd.read_csv(OUT / "accession_taxid_cache.csv")
    c = dict(zip(cache["accession"], cache["taxid"]))
    meta = pd.read_csv(OUT / "refseq_metadata.csv")
    meta["tx"] = meta["accession"].map(c)
    meta = meta.dropna(subset=["tx"])
    meta["tx"] = meta["tx"].astype(int)
    im = {a: i for i, a in enumerate(sid)}
    meta = meta[meta["accession"].isin(im)].copy()
    meta["ei"] = meta["accession"].map(im)
    t2i = {}
    for t, e in zip(meta["tx"], meta["ei"]):
        t2i.setdefault(int(t), []).append(int(e))
    return {t: emb[ix].mean(0) for t, ix in t2i.items()}


_COMP = str.maketrans("ACGT", "TGCA")
_STOPS = {"TAA", "TAG", "TGA"}
_K4 = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=4))}
_COD = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=3))}


def composition_features():
    seqs, tid, buf = {}, None, []
    for fa in ["h2h_zoo_representative.fasta", "vector_borne_representative.fasta"]:
        for line in open(OUT / "dedup" / fa):
            if line.startswith(">"):
                if tid is not None:
                    seqs.setdefault(tid, "".join(buf))
                tid, buf = int(line[1:].strip()), []
            else:
                buf.append(line.strip().upper())
        if tid is not None:
            seqs.setdefault(tid, "".join(buf))
        tid, buf = None, []
    gc_len, kmer, codon = {}, {}, {}
    for t, s in seqs.items():
        s = "".join(ch for ch in s if ch in "ACGT")
        acgt = len(s)
        gc = (s.count("G") + s.count("C")) / max(1, acgt)
        v = np.zeros(256)
        for i in range(len(s) - 3):
            j = _K4.get(s[i:i + 4])
            if j is not None:
                v[j] += 1
        if v.sum():
            v /= v.sum()
        c = np.zeros(64)
        for strand in (s, s.translate(_COMP)[::-1]):
            for frame in range(3):
                orf = []
                for i in range(frame, len(strand) - 2, 3):
                    cod = strand[i:i + 3]
                    if cod in _STOPS:
                        if len(orf) >= 30:
                            for cc in orf:
                                c[_COD[cc]] += 1
                        orf = []
                    elif cod in _COD:
                        orf.append(cod)
                if len(orf) >= 30:
                    for cc in orf:
                        c[_COD[cc]] += 1
        if c.sum():
            c /= c.sum()
        gc_len[t], kmer[t], codon[t] = [gc, np.log10(max(1, acgt))], v, c
    return gc_len, kmer, codon


# ── evaluation ────────────────────────────────────────────────────────────────
def pipe(clf):
    return Pipeline([("s", StandardScaler()), ("c", clf)])


def classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    }


def cv_scores(X, y, clf):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(pipe(clf), X, y, cv=skf, method="predict_proba")[:, 1]
    return {"AUROC": round(roc_auc_score(y, p), 4), "AUPRC": round(average_precision_score(y, p), 4)}


def main():
    parent, name, merged = load_taxonomy()
    has_ancestor = make_lineage_test(parent, name, merged)
    emb = per_taxon_embeddings()
    gc_len, kmer, codon = composition_features()

    # VIRION: virus taxid -> set of host taxids
    v2h = {}
    for r in csv.DictReader(open(BASE / "data" / "virion_edgelist.csv")):
        if r["VirusTaxID"].isdigit() and r["HostTaxID"].isdigit():
            v2h.setdefault(int(r["VirusTaxID"]), set()).add(int(r["HostTaxID"]))

    usable = sorted(t for t in v2h if t in emb and t in gc_len)
    print(f"usable viruses (embedding + genome + host records): {len(usable)}")

    # per-virus labels
    labels = {g: {} for g in HOST_GROUPS}
    for t in usable:
        for g, target in HOST_GROUPS.items():
            labels[g][t] = int(any(has_ancestor(h, target) for h in v2h[t]))
    with open(OUT / "host_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["virus_taxid", "virus_name"] + [f"infects_{g}" for g in HOST_GROUPS])
        for t in usable:
            w.writerow([t, name.get(merged.get(t, t), "")] + [labels[g][t] for g in HOST_GROUPS])

    feats = {"Vir2vec": {t: emb[t] for t in usable},
             "GC+Length": {t: gc_len[t] for t in usable},
             "k-mer(k=4)": {t: kmer[t] for t in usable},
             "Codon-usage": {t: codon[t] for t in usable}}
    results = {}
    for g in HOST_GROUPS:
        y = np.array([labels[g][t] for t in usable])
        task = {"n": len(usable), "positive": int(y.sum())}
        for cname, clf in classifiers().items():
            X = np.vstack([feats["Vir2vec"][t] for t in usable])
            task[f"Vir2vec / {cname}"] = cv_scores(X, y, clf)
        task["GC+Length"] = cv_scores(np.array([feats["GC+Length"][t] for t in usable]), y,
                                      LogisticRegression(max_iter=1000, class_weight="balanced"))
        task["k-mer(k=4)"] = cv_scores(np.array([feats["k-mer(k=4)"][t] for t in usable]), y,
                                       GradientBoostingClassifier(n_estimators=200, random_state=SEED))
        task["Codon-usage"] = cv_scores(np.array([feats["Codon-usage"][t] for t in usable]), y,
                                        GradientBoostingClassifier(n_estimators=200, random_state=SEED))
        results[g] = task
        best_v = max((task[k]["AUROC"] for k in task if k.startswith("Vir2vec")))
        print(f"  infects {g:8s} (n={len(usable)}, pos={int(y.sum())}): "
              f"Vir2vec best={best_v}  k-mer={task['k-mer(k=4)']['AUROC']}  "
              f"codon={task['Codon-usage']['AUROC']}  GC={task['GC+Length']['AUROC']}")

    (OUT / "host_prediction_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT / 'host_prediction_results.json'} and {OUT / 'host_labels.csv'}")


if __name__ == "__main__":
    main()
