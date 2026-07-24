"""
Significance and pre-training-exposure checks for host-group prediction
======================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Follow-up to scripts/host_prediction_analysis.py. That script found that the frozen
Vir2vec representation beat every composition baseline on all three host-group tasks
under cross-validation --- the opposite of the human-phenotype tasks. This script asks
whether that margin is real, and whether it survives on viruses Vir2vec never saw.

For each task it computes:
  * out-of-fold CV predictions for Vir2vec (best of four classifiers) and each baseline;
  * paired bootstrap 95% CIs on every AUROC and on each (Vir2vec - baseline) difference,
    with a two-sided bootstrap p-value;
  * the same CV predictions scored separately on the SEEN and UNSEEN subsets;
  * a train-on-seen / test-on-unseen prospective test (reported with its sample size,
    as the seen subset here is small).

Writes a NEW file only: outputs/host_significance_results.json
"""

import csv
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
csv.field_size_limit(10 ** 7)
SEED, N_BOOT = 42, 2000
BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
TASKS = ["birds", "bats", "rodents"]
rng = np.random.default_rng(SEED)

_COMP = str.maketrans("ACGT", "TGCA")
_STOPS = {"TAA", "TAG", "TGA"}
_K4 = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=4))}
_COD = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=3))}


def pipe(clf):
    return Pipeline([("s", StandardScaler()), ("c", clf)])


def classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    }


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
        gc = (s.count("G") + s.count("C")) / max(1, len(s))
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
        gc_len[t], kmer[t], codon[t] = [gc, np.log10(max(1, len(s)))], v, c
    return gc_len, kmer, codon


def ci(vals):
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main():
    emb = per_taxon_embeddings()
    gc_len, kmer, codon = composition_features()
    rows = list(csv.DictReader(open(OUT / "host_labels.csv")))
    seen_map = {int(r["virus_taxid"]): int(r["vir2vec_seen"])
                for r in csv.DictReader(open(OUT / "vir2vec_seen_labels.csv"))}
    tx = [int(r["virus_taxid"]) for r in rows]
    seen = np.array([seen_map.get(t, -1) for t in tx])   # -1 = unknown

    results = {}
    for task in TASKS:
        y = np.array([int(r[f"infects_{task}"]) for r in rows])
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        preds = {}
        # Vir2vec: keep the best-performing classifier for this task
        Xv = np.vstack([emb[t] for t in tx])
        best_name, best_p, best_auc = None, None, -1
        for cname, clf in classifiers().items():
            p = cross_val_predict(pipe(clf), Xv, y, cv=skf, method="predict_proba")[:, 1]
            a = roc_auc_score(y, p)
            if a > best_auc:
                best_name, best_p, best_auc = cname, p, a
        preds[f"Vir2vec ({best_name})"] = best_p
        for nm, feat, clf in [
            ("k-mer(k=4)", kmer, GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
            ("Codon-usage", codon, GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
            ("GC+Length", gc_len, LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]:
            X = np.array([feat[t] for t in tx])
            preds[nm] = cross_val_predict(pipe(clf), X, y, cv=skf, method="predict_proba")[:, 1]

        vkey = f"Vir2vec ({best_name})"
        n = len(y)
        boots = {m: np.empty(N_BOOT) for m in preds}
        diffs = {m: np.empty(N_BOOT) for m in preds if m != vkey}
        idx = np.arange(n)
        for b in range(N_BOOT):
            s = rng.choice(idx, n, replace=True)
            if len(set(y[s])) < 2:
                for m in preds:
                    boots[m][b] = np.nan
                for m in diffs:
                    diffs[m][b] = np.nan
                continue
            aucs = {m: roc_auc_score(y[s], preds[m][s]) for m in preds}
            for m in preds:
                boots[m][b] = aucs[m]
            for m in diffs:
                diffs[m][b] = aucs[vkey] - aucs[m]   # positive => Vir2vec ahead

        t = {"n": int(n), "positive": int(y.sum()), "vir2vec_model": best_name, "methods": {}, "vs_baselines": {}}
        for m in preds:
            v = boots[m][~np.isnan(boots[m])]
            t["methods"][m] = {"AUROC": round(float(roc_auc_score(y, preds[m])), 4),
                               "AUPRC": round(float(average_precision_score(y, preds[m])), 4),
                               "CI95": ci(v)}
        for m in diffs:
            dv = diffs[m][~np.isnan(diffs[m])]
            t["vs_baselines"][m] = {"delta_AUROC": round(float(dv.mean()), 4),
                                    "CI95": ci(dv),
                                    "p_value": round(float(2 * min((dv <= 0).mean(), (dv >= 0).mean())), 4)}

        # seen / unseen breakdown on the same CV predictions
        sub = {}
        for lab, mask in [("seen", seen == 1), ("unseen", seen == 0)]:
            if mask.sum() and len(set(y[mask])) > 1:
                sub[lab] = {"n": int(mask.sum()), "positive": int(y[mask].sum()),
                            **{m: round(float(roc_auc_score(y[mask], preds[m][mask])), 4) for m in preds}}
        t["cv_by_exposure"] = sub

        # prospective: train on seen, test on unseen
        tr, te = seen == 1, seen == 0
        if tr.sum() > 20 and len(set(y[tr])) > 1 and len(set(y[te])) > 1:
            pro = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "train_positive": int(y[tr].sum())}
            c = pipe(classifiers()[best_name]); c.fit(Xv[tr], y[tr])
            pro[vkey] = round(float(roc_auc_score(y[te], c.predict_proba(Xv[te])[:, 1])), 4)
            for nm, feat, clf in [
                ("k-mer(k=4)", kmer, GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
                ("Codon-usage", codon, GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
                ("GC+Length", gc_len, LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]:
                X = np.array([feat[t_] for t_ in tx])
                c = pipe(clf); c.fit(X[tr], y[tr])
                pro[nm] = round(float(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1])), 4)
            t["train_seen_test_unseen"] = pro

        results[task] = t
        d = t["vs_baselines"]
        print(f"{task:8s} n={n} pos={int(y.sum())} | Vir2vec({best_name}) {t['methods'][vkey]['AUROC']} "
              f"{t['methods'][vkey]['CI95']}")
        for m, v in d.items():
            flag = "SIG" if v["p_value"] < 0.05 else "ns"
            print(f"           vs {m:14s} delta={v['delta_AUROC']:+.4f} CI{v['CI95']} p={v['p_value']} [{flag}]")

    (OUT / "host_significance_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT / 'host_significance_results.json'}")


if __name__ == "__main__":
    main()
