"""
Tier 1 Pipeline: Vector-Borne Phenotype (Frozen Vir2vec Embeddings + Shallow Classifiers)
===========================================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Same methodology as pipeline/tier1_pipeline.py (H2H / zoonotic), applied to the
vector-borne phenotype: does a mosquito/tick-associated virus cross into a
vertebrate (human-infecting) host, or does it stay insect/arachnid-only?

Usage:
    conda activate viral-phenotype
    python pipeline/tier1_vector_borne.py

Requires:
    outputs/refseq_embeddings.npz     — full embedding set (must include the
                                         ZOVER mosquito/tick sequences)
    outputs/refseq_metadata.csv
    data/vector_borne_labels.csv      — from scripts/derive_vector_borne_labels.py

Outputs (written to outputs/results_vector_borne/):
    tier1_results.json
    tier1_figures/
    labels_matched.csv
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dedup', action='store_true',
                     help='Filter to CD-HIT-deduplicated taxa to remove near-duplicate-genome '
                          'leakage across CV folds')
args = parser.parse_args()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / ("results_vector_borne_dedup" if args.dedup else "results_vector_borne")
FIGURES_DIR = RESULTS_DIR / "tier1_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "vector_borne_deduplicated_taxids.txt"

EMBEDDINGS_PATH = OUTPUTS_DIR / "refseq_embeddings.npz"
METADATA_PATH   = OUTPUTS_DIR / "refseq_metadata.csv"
CACHE_PATH      = OUTPUTS_DIR / "accession_taxid_cache.csv"
LABELS_PATH     = BASE_DIR / "data" / "vector_borne_labels.csv"

# ── 1. LOAD EMBEDDINGS ───────────────────────────────────────────────────────
print("Loading embeddings...")
data = np.load(EMBEDDINGS_PATH)
embeddings = data["embeddings"]
seq_ids    = data["accessions"]

meta = pd.read_csv(METADATA_PATH)
print(f"  Embeddings: {embeddings.shape}")
print(f"  Metadata rows: {len(meta)}")

# ── 2. LOAD VECTOR-BORNE LABELS ───────────────────────────────────────────────
print("\nLoading vector-borne labels...")
label_df = pd.read_csv(LABELS_PATH)
print(f"  Labelled taxa: {len(label_df)}")
print(f"  vector_borne=1: {label_df['vector_borne'].sum()}  "
      f"vector_borne=0: {(label_df['vector_borne'] == 0).sum()}")

# ── 3. MATCH: accession -> taxid via cached lookup ───────────────────────────
print("\nResolving accession -> taxid (from cache)...")
cache_df = pd.read_csv(CACHE_PATH)
cache = dict(zip(cache_df["accession"], cache_df["taxid"]))

# Try exact (versioned) match first, then fall back to stripped-accession
# match — the cache may hold a different version suffix than what's in
# refseq_metadata.csv (e.g. NCBI resolved a since-revised AccessionVersion).
meta["virus_taxid"] = meta["accession"].map(cache)
exact_matched = meta["virus_taxid"].notna().sum()

stripped_to_taxid = {}
for acc_ver, tid in cache.items():
    stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)

still_missing = meta["virus_taxid"].isna()
meta.loc[still_missing, "virus_taxid"] = (
    meta.loc[still_missing, "accession"].str.split(".").str[0].map(stripped_to_taxid)
)

n_missing = meta["virus_taxid"].isna().sum()
print(f"  Resolved: {len(meta) - n_missing}/{len(meta)} accessions "
      f"({exact_matched} exact, {len(meta) - n_missing - exact_matched} via stripped match)")
if n_missing > 0:
    print(f"  {n_missing} accessions not in cache — dropping them "
          f"(run scripts/derive_vector_borne_labels.py first to extend the cache)")
    meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

# Aggregate embeddings per virus taxid (mean pooling across accessions)
print("Aggregating embeddings per virus taxid (mean pooling)...")
valid_accs = set(seq_ids.tolist())
meta = meta[meta["accession"].isin(valid_accs)].copy()
meta["embedding_idx"] = meta["accession"].map({a: i for i, a in enumerate(seq_ids)})

taxid_to_indices = {}
for idx, row in meta.iterrows():
    tid = row["virus_taxid"]
    taxid_to_indices.setdefault(tid, []).append(int(row["embedding_idx"]))

taxids_agg, embeddings_agg = [], []
for tid, indices in taxid_to_indices.items():
    taxids_agg.append(tid)
    embeddings_agg.append(embeddings[indices].mean(axis=0))

taxids_agg     = np.array(taxids_agg)
embeddings_agg = np.array(embeddings_agg)
print(f"  Unique virus taxids with embeddings: {len(taxids_agg)}")

# ── 4. INTERSECT WITH VECTOR-BORNE LABELS ────────────────────────────────────
print("\nMatching to vector-borne labels...")
label_taxids = set(label_df["virus_taxid"])
emb_taxids   = set(taxids_agg.tolist())
matched      = label_taxids & emb_taxids
print(f"  Labelled taxa:        {len(label_taxids)}")
print(f"  Taxa with embeddings: {len(emb_taxids)}")
print(f"  Matched (both):       {len(matched)}")
if len(matched) == 0:
    raise SystemExit("No matches found — check that refseq_embeddings.npz includes "
                      "the ZOVER sequences and that the taxid cache is up to date.")

if args.dedup:
    if not DEDUP_TAXIDS_PATH.exists():
        raise SystemExit(f"--dedup requested but {DEDUP_TAXIDS_PATH} not found — "
                          f"run scripts/dedup_sequences.py --dataset vector_borne first.")
    with open(DEDUP_TAXIDS_PATH) as f:
        dedup_taxids = set(int(line.strip()) for line in f if line.strip())
    n_before = len(matched)
    matched = matched & dedup_taxids
    print(f"  --dedup: filtered {n_before} -> {len(matched)} taxa "
          f"({n_before - len(matched)} near-duplicate taxa removed)")

matched_order = sorted(matched)
label_lookup  = label_df.set_index("virus_taxid")["vector_borne"]
emb_lookup    = {tid: emb for tid, emb in zip(taxids_agg.tolist(), embeddings_agg)}

X = np.array([emb_lookup[t] for t in matched_order])
y = np.array([label_lookup.loc[t] for t in matched_order])

print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
print(f"  vector_borne — {y.sum()} pos / {(y == 0).sum()} neg ({y.mean():.1%} positive)")

matched_df = pd.DataFrame({"virus_taxid": matched_order, "vector_borne": y})
matched_df.to_csv(RESULTS_DIR / "labels_matched.csv", index=False)
print(f"\n  Saved: {RESULTS_DIR / 'labels_matched.csv'}")

# ── 5. CLASSIFIERS ───────────────────────────────────────────────────────────
CLASSIFIERS = {
    "Random (Baseline)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DummyClassifier(strategy="stratified", random_state=42))
    ]),
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
    """Optimal decision threshold via Youden's J statistic (max TPR - FPR).

    A fixed 0.5 threshold collapses predictions to the majority class on
    imbalanced tasks like this one (8.5% positive) — Youden's J picks the
    ROC point that best trades off sensitivity/specificity instead.
    """
    fpr, tpr, thresholds = roc_curve(y, probs)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

def evaluate(X, y, task_name):
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    results, all_probs, all_thresholds = {}, {}, {}
    for clf_name, clf in CLASSIFIERS.items():
        print(f"  Training {clf_name}...", end=" ", flush=True)
        probs = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]
        # Keep the naive baseline at 0.5 (it's a reference floor, not a model
        # to threshold-tune); pick Youden's J threshold for real classifiers.
        threshold = 0.5 if clf_name == "Random (Baseline)" else youden_threshold(y, probs)
        preds = (probs >= threshold).astype(int)
        metrics = {
            "AUROC":  round(roc_auc_score(y, probs), 4),
            "AUPRC":  round(average_precision_score(y, probs), 4),
            "MCC":    round(matthews_corrcoef(y, preds), 4),
            "F1_macro":    round(f1_score(y, preds, average="macro"), 4),
            "F1_weighted": round(f1_score(y, preds, average="weighted"), 4),
            "Brier":  round(brier_score_loss(y, probs), 4),
            "Threshold": round(float(threshold), 4),
        }
        results[clf_name] = metrics
        all_probs[clf_name] = probs
        all_thresholds[clf_name] = threshold
        print(f"AUROC={metrics['AUROC']:.3f}  AUPRC={metrics['AUPRC']:.3f}  MCC={metrics['MCC']:.3f}  (thresh={threshold:.3f})")
    return results, all_probs, all_thresholds

def plot_confusion_matrix(y, all_probs, all_thresholds, task_name, out_dir):
    fig, axes = plt.subplots(1, len(all_probs), figsize=(5 * len(all_probs), 5))
    fig.suptitle(f"{task_name} — Confusion Matrices", fontsize=13)
    for ax, (clf_name, probs) in zip(axes, all_probs.items()):
        preds = (probs >= all_thresholds[clf_name]).astype(int)
        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(clf_name, fontsize=9)
    plt.tight_layout()
    fname = out_dir / f"{task_name.replace(' ', '_')}_confusion_matrices.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")

def plot_roc_pr(y, all_probs, task_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{task_name} — ROC & Precision-Recall Curves", fontsize=13)
    for clf_name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        axes[0].plot(fpr, tpr, label=f"{clf_name} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y, probs)
        ap = average_precision_score(y, probs)
        axes[1].plot(rec, prec, label=f"{clf_name} (AP={ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8)
    baseline = y.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", lw=0.8, label=f"Baseline ({baseline:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    fname = out_dir / f"{task_name.replace(' ', '_')}_roc_pr.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")

def plot_calibration(y, all_probs, task_name, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"{task_name} — Calibration / Reliability Diagram")
    for clf_name, probs in all_probs.items():
        fraction_of_positives, mean_predicted = calibration_curve(y, probs, n_bins=10)
        ax.plot(mean_predicted, fraction_of_positives, marker="o", label=clf_name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = out_dir / f"{task_name.replace(' ', '_')}_calibration.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")

# ── 6. RUN ────────────────────────────────────────────────────────────────────
all_results = {}
results_vb, probs_vb, thresholds_vb = evaluate(X, y, "Vector-Borne")
plot_roc_pr(y, probs_vb, "Vector-Borne", FIGURES_DIR)
plot_confusion_matrix(y, probs_vb, thresholds_vb, "Vector-Borne", FIGURES_DIR)
plot_calibration(y, probs_vb, "Vector-Borne", FIGURES_DIR)
all_results["vector_borne"] = results_vb

# ── 6b. TICK-ONLY HOLD-OUT (host-type analogue of the Flaviviridae hold-out) ──
print("\n" + "=" * 70)
print("HOLD-OUT: tick-only viruses (never trained on)")
print("=" * 70)
host_type_lookup = label_df.set_index("virus_taxid")["host_types"]
host_types_arr = np.array([host_type_lookup.loc[t] for t in matched_order])
holdout_mask = np.array(["tick" in h and "mosquito" not in h for h in host_types_arr])
train_mask = ~holdout_mask
print(f"  Hold-out: {holdout_mask.sum()} taxa, train: {train_mask.sum()} taxa")

if holdout_mask.sum() >= 5:
    X_train_ho, X_test_ho = X[train_mask], X[holdout_mask]
    y_train_ho, y_test_ho = y[train_mask], y[holdout_mask]
    print(f"  Hold-out: {y_test_ho.sum()} pos / {(y_test_ho==0).sum()} neg")
    for clf_name, clf in CLASSIFIERS.items():
        if clf_name == "Random (Baseline)":
            continue
        clf.fit(X_train_ho, y_train_ho)
        probs = clf.predict_proba(X_test_ho)[:, 1]
        try:
            auroc = roc_auc_score(y_test_ho, probs)
            auprc = average_precision_score(y_test_ho, probs)
            print(f"    {clf_name:<25} AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
            all_results.setdefault("holdout_tick_only", {})[clf_name] = {
                "AUROC": round(auroc, 4), "AUPRC": round(auprc, 4)}
        except Exception as e:
            print(f"    {clf_name}: {e}")

# ── 7. PRINT SUMMARY TABLE ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for task, res in all_results.items():
    print(f"\n{task}")
    header = f"{'Classifier':<25} {'AUROC':>7} {'AUPRC':>7} {'MCC':>7} {'F1mac':>7} {'Brier':>7} {'Thresh':>7}"
    print(header)
    print("-" * len(header))
    for clf, m in res.items():
        if 'MCC' in m:
            print(f"{clf:<25} {m['AUROC']:>7.4f} {m['AUPRC']:>7.4f} {m['MCC']:>7.4f} {m['F1_macro']:>7.4f} {m['Brier']:>7.4f} {m['Threshold']:>7.4f}")
        else:
            print(f"{clf:<25} {m['AUROC']:>7.4f} {m['AUPRC']:>7.4f}")

# ── 8. SAVE RESULTS JSON ──────────────────────────────────────────────────────
results_path = RESULTS_DIR / "tier1_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved: {results_path}")
print("Done! ✓")
