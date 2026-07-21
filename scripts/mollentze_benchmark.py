# Mollentze et al. (2021) benchmark evaluation (Ticket 8).
# Trains Tier-1-style classifiers on Vir2vec embeddings for the 891
# matched Mollentze species, predicting InfectsHumans, and compares
# against their reported phylogenetic-baseline AUC of 0.773.
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, brier_score_loss, roc_curve
)

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
OUT_DIR = OUTPUTS_DIR / "mollentze_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Mollentze scope-check data (accession, taxid, InfectsHumans)...")
mdf = pd.read_csv(DATA_DIR / "mollentze_scope_check.csv")
mdf = mdf.dropna(subset=["resolved_taxid"])
mdf["resolved_taxid"] = mdf["resolved_taxid"].astype(int)
print(f"  {len(mdf)} species with a resolved taxid")

print("\nLoading embeddings...")
data = np.load(OUTPUTS_DIR / "refseq_embeddings.npz")
embeddings = data["embeddings"]
seq_ids = list(data["accessions"])
acc_to_idx = {a: i for i, a in enumerate(seq_ids)}

mdf["embedding_idx"] = mdf["accession"].map(acc_to_idx)
n_missing_emb = mdf["embedding_idx"].isna().sum()
print(f"  Missing embedding despite scope-check: {n_missing_emb} (dropping)")
mdf = mdf.dropna(subset=["embedding_idx"])
mdf["embedding_idx"] = mdf["embedding_idx"].astype(int)

print("\nAggregating embeddings per species taxid (mean pooling)...")
taxid_to_indices = {}
for _, row in mdf.iterrows():
    taxid_to_indices.setdefault(row["resolved_taxid"], []).append(row["embedding_idx"])

taxid_to_label = mdf.groupby("resolved_taxid")["InfectsHumans"].first().to_dict()
taxids_agg = sorted(taxid_to_indices.keys())
X = np.array([embeddings[taxid_to_indices[t]].mean(axis=0) for t in taxids_agg])
y = np.array([int(bool(taxid_to_label[t])) for t in taxids_agg])
print(f"  {len(taxids_agg)} unique species, {y.sum()} InfectsHumans=True / {(y==0).sum()} False")

CLASSIFIERS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ]),
}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def youden_threshold(y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)
    return thresholds[np.argmax(tpr - fpr)]

print(f"\n{'='*70}\nTask: Mollentze InfectsHumans (n={len(y)})\n{'='*70}")
print("Mollentze et al.'s own reported phylogenetic-baseline AUC: 0.773 "
      "(their actual genomic-feature model beats this — see paper for exact figure)")
results = {}
for name, clf in CLASSIFIERS.items():
    print(f"  Training {name}...", end=" ", flush=True)
    probs = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]
    threshold = youden_threshold(y, probs)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "AUROC": round(roc_auc_score(y, probs), 4),
        "AUPRC": round(average_precision_score(y, probs), 4),
        "MCC": round(matthews_corrcoef(y, preds), 4),
        "F1_macro": round(f1_score(y, preds, average="macro"), 4),
        "Brier": round(brier_score_loss(y, probs), 4),
        "Threshold": round(float(threshold), 4),
    }
    results[name] = metrics
    print(f"AUROC={metrics['AUROC']:.4f}  MCC={metrics['MCC']:.4f}")

out_path = OUT_DIR / "mollentze_benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

best = max(results, key=lambda k: results[k]["AUROC"])
print(f"\nBest: {best}  AUROC={results[best]['AUROC']:.4f}  "
      f"(vs Mollentze's phylo-baseline 0.773 — "
      f"{'BEATS' if results[best]['AUROC'] > 0.773 else 'does not beat'} it)")
print("Done! ✓")
