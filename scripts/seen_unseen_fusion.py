"""
Does Vir2vec add anything on top of codon usage? Feature-fusion test (unseen viruses)
=====================================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

The prospective test (§4.8) showed that on genuinely unseen viruses Vir2vec is beaten
by codon usage for human-to-human transmission. This asks the natural follow-up: if we
*concatenate* Vir2vec's embedding with the codon-usage vector, does the combination beat
codon usage alone? A clean test holds the classifier fixed (Gradient Boosting) and varies
only the feature set:

    Codon            : 64-dim codon usage          (= the §4.8 baseline)
    Fusion           : codon usage  +  Vir2vec      (486-dim)
    Vir2vec (ref)    : Vir2vec embedding, best-of-4  (= the §4.8 language-model arm)

All three are trained on SEEN viruses and scored on UNSEEN ones, with 2000 paired
bootstrap resamples for 95% CIs and a two-sided p-value on Fusion - Codon (and Fusion -
Vir2vec). If Fusion does not significantly exceed Codon, Vir2vec adds no signal beyond
composition even when handed to the classifier alongside it.

Writes: outputs/seen_unseen_fusion.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    _, kmer, codon = composition_features()
    lab = pd.read_csv(BASE / "data" / "phenotype_labels_with_strict_zoonotic.csv").set_index("virus_taxid")
    seen_map = dict(pd.read_csv(OUT / "vir2vec_seen_labels.csv").values)
    dedup = {int(x) for x in open(OUT / "dedup" / "h2h_zoo_deduplicated_taxids.txt") if x.strip()}
    gb = lambda: GradientBoostingClassifier(n_estimators=200, random_state=SEED)

    results = {}
    for col in TASKS:
        tx = [t for t in lab.index if t in emb and t in codon and t in dedup]
        y = np.array([int(lab.loc[t, col]) for t in tx])
        seen = np.array([int(seen_map.get(t, 0)) for t in tx])

        E = np.vstack([emb[t] for t in tx])
        C = np.array([codon[t] for t in tx])
        F = np.hstack([C, E])                       # fusion: codon + Vir2vec

        preds = {}
        preds["Codon"], yte = fit_predict(C, y, seen, gb())
        preds["Fusion"], _ = fit_predict(F, y, seen, gb())
        # Vir2vec reference: best-of-four (its strongest showing), as in §4.8
        vp = {}
        for name, clf in classifiers().items():
            vp[name], _ = fit_predict(E, y, seen, clf)
        best = max(vp, key=lambda n: roc_auc_score(yte, vp[n]))
        preds[f"Vir2vec ({best})"] = vp[best]
        vir_key = f"Vir2vec ({best})"

        n = len(yte); idx = np.arange(n)
        boots = {m: np.empty(N_BOOT) for m in preds}
        pairs = {"Fusion - Codon": ("Fusion", "Codon"),
                 "Fusion - Vir2vec": ("Fusion", vir_key)}
        dboot = {k: np.empty(N_BOOT) for k in pairs}
        for b in range(N_BOOT):
            s = rng.choice(idx, n, replace=True)
            if len(set(yte[s])) < 2:
                for m in preds: boots[m][b] = np.nan
                for k in pairs: dboot[k][b] = np.nan
                continue
            a = {m: roc_auc_score(yte[s], preds[m][s]) for m in preds}
            for m in preds: boots[m][b] = a[m]
            for k, (x1, x2) in pairs.items(): dboot[k][b] = a[x1] - a[x2]

        res = {"n_unseen": int(n), "pos_unseen": int(yte.sum()), "vir2vec_model": best,
               "AUROC": {}, "delta": {}}
        for m in preds:
            v = boots[m][~np.isnan(boots[m])]
            res["AUROC"][m] = {"point": round(float(roc_auc_score(yte, preds[m])), 4), "CI95": ci(v)}
        for k in pairs:
            dv = dboot[k][~np.isnan(dboot[k])]
            p_two = 2 * min((dv <= 0).mean(), (dv >= 0).mean())
            res["delta"][k] = {"mean": round(float(dv.mean()), 4), "CI95": ci(dv),
                               "p_value": round(float(p_two), 4)}
        results[col] = res

        cod = res["AUROC"]["Codon"]["point"]; fus = res["AUROC"]["Fusion"]["point"]
        d = res["delta"]["Fusion - Codon"]
        print(f"{col:18s}  Codon {cod:.3f} | Fusion {fus:.3f} | Vir2vec {res['AUROC'][vir_key]['point']:.3f}"
              f"  | Fusion-Codon {d['mean']:+.3f} (p={d['p_value']:.3f})")

    (OUT / "seen_unseen_fusion.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT / 'seen_unseen_fusion.json'}")


if __name__ == "__main__":
    main()
