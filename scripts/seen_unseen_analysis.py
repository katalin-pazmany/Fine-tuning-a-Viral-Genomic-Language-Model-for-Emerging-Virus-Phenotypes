"""
Seen-vs-unseen (out-of-distribution) evaluation against Vir2vec's pre-training
==============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Tests whether the frozen Vir2vec representation predicts phenotypes for viruses the
model *never saw during pre-training*, and how it compares to cheap composition
baselines in that setting. Uses the SEEN/UNSEEN split from
outputs/vir2vec_seen_labels.csv (see scripts/make_vir2vec_seen_labels.py).

For each phenotype and each method it reports, on the CD-HIT-deduplicated data:
  * CV (all)     — standard stratified 5-fold out-of-fold performance
  * CV (seen)    — same folds, restricted to the seen subset
  * CV (unseen)  — same folds, restricted to the unseen subset
  * train-SEEN -> test-UNSEEN — the prospective test: fit only on viruses Vir2vec
    saw, evaluate only on genuinely novel ones
Both AUROC and AUPRC are reported (base rates differ between seen and unseen).

Methods: Vir2vec frozen embeddings (LR / RF / GB / SVM) and the composition
baselines (GC-content+length; tetranucleotide k-mer, k=4).

Usage:
    conda activate viral-phenotype
    python scripts/seen_unseen_analysis.py
Writes: outputs/seen_unseen_results.json
"""

import itertools
import json
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
BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
TASKS = ["human_to_human", "zoonotic_relaxed", "zoonotic_strict"]
SEED = 42


def pipe(clf):
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    }


# ── Features ──────────────────────────────────────────────────────────────────
def vir2vec_embeddings():
    d = np.load(OUT / "refseq_embeddings.npz")
    emb, sid = d["embeddings"], list(d["accessions"])
    meta = pd.read_csv(OUT / "refseq_metadata.csv")
    cache = pd.read_csv(OUT / "accession_taxid_cache.csv")
    c = dict(zip(cache["accession"], cache["taxid"]))
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
_CODONS = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=3))}


def revcomp(s):
    return s.translate(_COMP)[::-1]


def codon_usage(seq, min_orf=30):
    """64-dim codon-frequency vector from open reading frames (>=min_orf codons,
    between stop codons) across all six frames. A reading-frame-aware baseline —
    the biologically pointed test of whether Vir2vec just captures codon bias."""
    counts = np.zeros(64)
    for strand in (seq, revcomp(seq)):
        for frame in range(3):
            orf = []
            for i in range(frame, len(strand) - 2, 3):
                c = strand[i:i + 3]
                if c in _STOPS:
                    if len(orf) >= min_orf:
                        for cc in orf:
                            counts[_CODONS[cc]] += 1
                    orf = []
                elif c in _CODONS:
                    orf.append(c)
            if len(orf) >= min_orf:
                for cc in orf:
                    counts[_CODONS[cc]] += 1
    if counts.sum() > 0:
        counts /= counts.sum()
    return counts


def composition_features():
    seqs, tid, buf = {}, None, []
    for line in open(OUT / "dedup" / "h2h_zoo_representative.fasta"):
        if line.startswith(">"):
            if tid is not None:
                seqs[tid] = "".join(buf)
            tid, buf = int(line[1:].strip()), []
        else:
            buf.append(line.strip().upper())
    if tid is not None:
        seqs[tid] = "".join(buf)
    kidx = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=4))}
    gc_len, kmer, codon = {}, {}, {}
    for t, s in seqs.items():
        acgt = sum(s.count(b) for b in "ACGT")
        gc = (s.count("G") + s.count("C")) / max(1, acgt)
        v = np.zeros(256)
        for i in range(len(s) - 3):
            k = s[i:i + 4]
            if k in kidx:
                v[kidx[k]] += 1
        if v.sum() > 0:
            v /= v.sum()
        gc_len[t], kmer[t], codon[t] = [gc, np.log10(max(1, len(s)))], v, codon_usage(s)
    return gc_len, kmer, codon


def scores(y, p, mask):
    yy, pp = y[mask], p[mask]
    if len(set(yy)) < 2:
        return {"AUROC": None, "AUPRC": None, "n": int(mask.sum()), "pos": int(yy.sum())}
    return {"AUROC": round(roc_auc_score(yy, pp), 4),
            "AUPRC": round(average_precision_score(yy, pp), 4),
            "n": int(mask.sum()), "pos": int(yy.sum())}


def evaluate(name, X, y, seen, clf):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(pipe(clf), X, y, cv=skf, method="predict_proba")[:, 1]
    allm = np.ones(len(y), bool)
    res = {"cv_all": scores(y, p, allm),
           "cv_seen": scores(y, p, seen == 1),
           "cv_unseen": scores(y, p, seen == 0)}
    tr, te = seen == 1, seen == 0
    if len(set(y[tr])) > 1 and len(set(y[te])) > 1:
        c = pipe(clf.__class__(**clf.get_params()))
        c.fit(X[tr], y[tr])
        pu = c.predict_proba(X[te])[:, 1]
        res["train_seen_test_unseen"] = {
            "AUROC": round(roc_auc_score(y[te], pu), 4),
            "AUPRC": round(average_precision_score(y[te], pu), 4),
            "n": int(te.sum()), "pos": int(y[te].sum())}
    return res


def main():
    emb = vir2vec_embeddings()
    gc_len, kmer, codon = composition_features()
    lab = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv").set_index("virus_taxid")
    seen_map = dict(pd.read_csv(OUT / "vir2vec_seen_labels.csv").values)
    dedup = {int(x) for x in open(OUT / "dedup" / "h2h_zoo_deduplicated_taxids.txt") if x.strip()}

    results = {}
    for col in TASKS:
        tx = [t for t in lab.index if t in emb and t in gc_len and t in dedup]
        y = np.array([int(lab.loc[t, col]) for t in tx])
        seen = np.array([int(seen_map.get(t, 0)) for t in tx])
        feats = {"Vir2vec": np.vstack([emb[t] for t in tx]),
                 "GC+Length": np.array([gc_len[t] for t in tx]),
                 "k-mer(k=4)": np.array([kmer[t] for t in tx]),
                 "Codon-usage": np.array([codon[t] for t in tx])}
        task_res = {"n": len(tx), "n_seen": int(seen.sum()), "n_unseen": int((seen == 0).sum())}
        for cname, clf in classifiers().items():
            task_res[f"Vir2vec / {cname}"] = evaluate(cname, feats["Vir2vec"], y, seen, clf)
        task_res["GC+Length"] = evaluate("GC+Length", feats["GC+Length"], y, seen,
                                         LogisticRegression(max_iter=1000, class_weight="balanced"))
        task_res["k-mer(k=4)"] = evaluate("k-mer", feats["k-mer(k=4)"], y, seen,
                                          GradientBoostingClassifier(n_estimators=200, random_state=SEED))
        task_res["Codon-usage"] = evaluate("Codon-usage", feats["Codon-usage"], y, seen,
                                           GradientBoostingClassifier(n_estimators=200, random_state=SEED))
        results[col] = task_res
        print(f"{col}: n={len(tx)} seen={seen.sum()} unseen={(seen == 0).sum()} — done")

    (OUT / "seen_unseen_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT / 'seen_unseen_results.json'}")


if __name__ == "__main__":
    main()
