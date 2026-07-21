"""
Bootstrap confidence intervals & significance for the seen/unseen prospective test
==================================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Quantifies whether the composition baselines really beat Vir2vec on genuinely
unseen viruses, or whether the gap is noise. For each phenotype, every method is
trained on the SEEN viruses and scored on the UNSEEN ones (as in §4.8); the UNSEEN
test set is then resampled with replacement (paired across methods, so every
bootstrap replicate uses the same viruses for all methods) to obtain 95% CIs on
each AUROC and on the paired differences (baseline - Vir2vec). A two-sided
bootstrap p-value is reported for each difference.

Vir2vec uses its best-of-four classifier per task (the conservative choice — it
gives the language model its strongest showing before comparing).

Usage:
    conda activate viral-phenotype
    python scripts/seen_unseen_significance.py
Writes: outputs/seen_unseen_significance.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from seen_unseen_analysis import (
    vir2vec_embeddings, composition_features, classifiers, pipe, OUT, BASE, TASKS, SEED,
)

warnings.filterwarnings("ignore")
N_BOOT = 2000
rng = np.random.default_rng(SEED)


def fit_predict(X, y, seen, clf):
    tr, te = seen == 1, seen == 0
    c = pipe(clf.__class__(**clf.get_params()))
    c.fit(X[tr], y[tr])
    return c.predict_proba(X[te])[:, 1], y[te]


def ci(vals):
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main():
    emb = vir2vec_embeddings()
    gc_len, kmer, codon = composition_features()
    lab = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv").set_index("virus_taxid")
    seen_map = dict(pd.read_csv(OUT / "vir2vec_seen_labels.csv").values)
    dedup = {int(x) for x in open(OUT / "dedup" / "h2h_zoo_deduplicated_taxids.txt") if x.strip()}
    gb = lambda: GradientBoostingClassifier(n_estimators=200, random_state=SEED)

    results = {}
    for col in TASKS:
        tx = [t for t in lab.index if t in emb and t in gc_len and t in dedup]
        y = np.array([int(lab.loc[t, col]) for t in tx])
        seen = np.array([int(seen_map.get(t, 0)) for t in tx])

        # Vir2vec: pick best-of-four by unseen AUROC (conservative — its best shot)
        v_preds = {}
        for name, clf in classifiers().items():
            p, yte = fit_predict(np.vstack([emb[t] for t in tx]), y, seen, clf)
            v_preds[name] = p
        best = max(v_preds, key=lambda n: roc_auc_score(yte, v_preds[n]))
        preds = {f"Vir2vec ({best})": v_preds[best]}
        preds["k-mer(k=4)"], _ = fit_predict(np.array([kmer[t] for t in tx]), y, seen, gb())
        preds["Codon-usage"], _ = fit_predict(np.array([codon[t] for t in tx]), y, seen, gb())
        preds["GC+Length"], _ = fit_predict(np.array([gc_len[t] for t in tx]), y, seen,
                                             LogisticRegression(max_iter=1000, class_weight="balanced"))

        n = len(yte)
        idx = np.arange(n)
        boots = {m: np.empty(N_BOOT) for m in preds}
        vir_key = f"Vir2vec ({best})"
        diffs = {m: np.empty(N_BOOT) for m in preds if m != vir_key}
        for b in range(N_BOOT):
            s = rng.choice(idx, n, replace=True)
            if len(set(yte[s])) < 2:
                for m in preds:
                    boots[m][b] = np.nan
                for m in diffs:
                    diffs[m][b] = np.nan
                continue
            aucs = {m: roc_auc_score(yte[s], preds[m][s]) for m in preds}
            for m in preds:
                boots[m][b] = aucs[m]
            for m in diffs:
                diffs[m][b] = aucs[m] - aucs[vir_key]

        task_res = {"n_unseen": int(n), "pos_unseen": int(yte.sum()), "vir2vec_model": best, "methods": {}}
        for m in preds:
            v = boots[m][~np.isnan(boots[m])]
            task_res["methods"][m] = {"AUROC": round(float(roc_auc_score(yte, preds[m])), 4),
                                      "CI95": ci(v)}
        task_res["vs_vir2vec"] = {}
        for m in diffs:
            dv = diffs[m][~np.isnan(diffs[m])]
            p_two = 2 * min((dv <= 0).mean(), (dv >= 0).mean())
            task_res["vs_vir2vec"][m] = {"delta_AUROC": round(float(dv.mean()), 4),
                                         "CI95": ci(dv), "p_value": round(float(p_two), 4)}
        results[col] = task_res
        print(f"{col}: Vir2vec({best}) vs baselines on {n} unseen taxa — done")

    (OUT / "seen_unseen_significance.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT / 'seen_unseen_significance.json'}")


if __name__ == "__main__":
    main()
