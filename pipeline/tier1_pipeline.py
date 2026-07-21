"""
Tier 1 Pipeline: Frozen Vir2vec Embeddings + Shallow Classifiers
=================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Usage:
    conda activate viral-phenotype
    python tier1_pipeline.py

Outputs (written to outputs/):
    tier1_results.json          — per-model metrics for both tasks
    tier1_figures/              — ROC curves, PR curves, calibration plots
    labels_matched.csv          — matched virus labels (inspect this first!)

What this script does:
    1. Loads your refseq_embeddings.npz + refseq_metadata.csv
    2. Derives labels from the reference human_pathogens.xlsx (both tasks)
    3. Matches viruses by virus_taxid
    4. Trains 4 shallow classifiers for each phenotype task
    5. Evaluates with AUROC, AUPRC, MCC, F1, Brier score
    6. Saves plots and a results JSON
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--strict', action='store_true', help='Use strict zoonotic labels')
parser.add_argument('--dedup', action='store_true',
                     help='Filter to CD-HIT-deduplicated taxa (outputs/dedup/h2h_zoo_deduplicated_taxids.txt) '
                          'to remove near-duplicate-genome leakage across CV folds')
args = parser.parse_args()
LABEL_TYPE = 'strict' if args.strict else 'relaxed'
print(f'Zoonotic label type: {LABEL_TYPE}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── scikit-learn classifiers & metrics ──────────────────────────────────────
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
_results_name = ("results_strict" if args.strict else "results") + ("_dedup" if args.dedup else "")
RESULTS_DIR = BASE_DIR / "outputs" / _results_name
FIGURES_DIR = RESULTS_DIR / "tier1_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "h2h_zoo_deduplicated_taxids.txt"

EMBEDDINGS_PATH = OUTPUTS_DIR / "refseq_embeddings.npz"
METADATA_PATH   = OUTPUTS_DIR / "refseq_metadata.csv"
LABELS_PATH     = BASE_DIR / "data" / "human_pathogens.xlsx"   # reference dataset file

# ── 1. LOAD EMBEDDINGS ───────────────────────────────────────────────────────
print("Loading embeddings...")
data = np.load(EMBEDDINGS_PATH)
embeddings = data["embeddings"]          # shape: (N_sequences, 768)
seq_ids    = data["accessions"]          # accession IDs matching metadata rows

meta = pd.read_csv(METADATA_PATH)
print(f"  Embeddings: {embeddings.shape}")
print(f"  Metadata rows: {len(meta)}")
print(f"  Metadata columns: {list(meta.columns)}")

# ── 2. DERIVE LABELS FROM MAYA'S FILE ────────────────────────────────────────
print("\nLoading phenotype labels...")
if args.strict:
    strict_df = pd.read_csv(BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv")
    human_df  = strict_df.rename(columns={"zoonotic_strict": "zoonotic"})
else:
    human_df       = pd.read_excel(LABELS_PATH, sheet_name="human")
    interactions_df = pd.read_excel(LABELS_PATH, sheet_name="interactions")
    
    # Task A: human_to_human (directly from the 'human' sheet)
    # Task B: zoonotic  (= virus appears with BOTH human AND non-human hosts in interactions)
    human_host_taxids     = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
    non_human_host_taxids = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
    zoonotic_taxids       = human_host_taxids & non_human_host_taxids
    
    human_df["zoonotic"] = human_df["virus_taxid"].isin(zoonotic_taxids).astype(int)

print(f"  human_to_human — pos: {human_df['human_to_human'].sum()}  neg: {(human_df['human_to_human']==0).sum()}")
print(f"  zoonotic       — pos: {human_df['zoonotic'].sum()}  neg: {(human_df['zoonotic']==0).sum()}")

# ── 3. MATCH: accession → taxid via NCBI (with local cache) ─────────────────
import time
import xml.etree.ElementTree as ET
from Bio import Entrez

Entrez.email = "your_email@liverpool.ac.uk"   # ← replace with your email

CACHE_PATH = OUTPUTS_DIR / "accession_taxid_cache.csv"

def load_cache():
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
        return dict(zip(df["accession"], df["taxid"]))
    return {}

def save_cache(cache):
    pd.DataFrame(list(cache.items()), columns=["accession","taxid"]).to_csv(CACHE_PATH, index=False)

def fetch_taxids_batch(accessions, cache, batch_size=200):
    """Fetch taxids via NCBI esummary (nuccore) — returns TaxId directly."""
    to_fetch = [a for a in accessions if a not in cache]
    print(f"  {len(cache)} accessions already cached, fetching {len(to_fetch)} from NCBI...")
    total_batches = (len(to_fetch) - 1) // batch_size + 1 if to_fetch else 0
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i+batch_size]
        print(f"    Batch {i//batch_size + 1}/{total_batches} ({len(batch)} accessions)...", end=" ", flush=True)
        resolved = 0
        try:
            # esearch to get UIDs for the batch
            handle = Entrez.esearch(db="nuccore", term=" OR ".join(batch), retmax=len(batch))
            search_record = Entrez.read(handle)
            handle.close()
            uids = search_record["IdList"]
            if uids:
                # esummary to get TaxId for each UID
                handle2 = Entrez.esummary(db="nuccore", id=",".join(uids))
                summaries = Entrez.read(handle2)
                handle2.close()
                for s in summaries:
                    acc_ver = s.get("AccessionVersion", "")
                    taxid   = s.get("TaxId", None)
                    if acc_ver and taxid:
                        cache[acc_ver] = int(taxid)
                        resolved += 1
            print(f"done ({resolved}/{len(batch)} resolved)")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.34)  # NCBI rate limit: 3 req/sec without API key
        save_cache(cache)
    return cache

print("\nResolving accession → taxid (NCBI lookup with local cache)...")
print("  Available columns:", list(meta.columns))
cache = load_cache()
accessions = meta["accession"].tolist()
cache = fetch_taxids_batch(accessions, cache)
save_cache(cache)

meta["virus_taxid"] = meta["accession"].map(cache)
n_missing = meta["virus_taxid"].isna().sum()
print(f"  Resolved: {len(meta) - n_missing}/{len(meta)} accessions")
if n_missing > 0:
    print(f"  ⚠️  {n_missing} accessions could not be resolved — dropping them")
    meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

# For viruses with multiple sequences, we average their embeddings (mean pooling)
# This gives one embedding vector per virus species.
print("Aggregating embeddings per virus taxid (mean pooling)...")
# Only use accessions that exist in the embeddings array
valid_accs = set(seq_ids.tolist())
meta = meta[meta["accession"].isin(valid_accs)].copy()
meta["embedding_idx"] = meta["accession"].map({a: i for i, a in enumerate(seq_ids)})

taxid_to_indices = {}
for idx, row in meta.iterrows():
    tid = row["virus_taxid"]
    taxid_to_indices.setdefault(tid, []).append(int(row["embedding_idx"]))

taxids_agg   = []
embeddings_agg = []
for tid, indices in taxid_to_indices.items():
    taxids_agg.append(tid)
    embeddings_agg.append(embeddings[indices].mean(axis=0))

taxids_agg   = np.array(taxids_agg)
embeddings_agg = np.array(embeddings_agg)          # shape: (unique_viruses, 768)
print(f"  Unique virus taxids in RefSeq embeddings: {len(taxids_agg)}")

# ── 4. INTERSECT WITH MAYA'S LABELS ─────────────────────────────────────────
print("\nMatching to the reference labels...")
label_taxids = set(human_df["virus_taxid"])
emb_taxids   = set(taxids_agg.tolist())
matched      = label_taxids & emb_taxids
print(f"  Labelled viruses (reference): {len(label_taxids)}")
print(f"  Viruses with embeddings: {len(emb_taxids)}")
print(f"  Matched (both):          {len(matched)}")
if len(matched) == 0:
    print("\n  ⚠️  No matches found! Check your metadata taxid column.")
    print("  Sample metadata taxids:", list(taxids_agg[:5]))
    print("  Sample label taxids:", list(human_df['virus_taxid'].head(5)))
    raise SystemExit("Fix the taxid mapping before proceeding.")

if args.dedup:
    if not DEDUP_TAXIDS_PATH.exists():
        raise SystemExit(f"--dedup requested but {DEDUP_TAXIDS_PATH} not found — "
                          f"run scripts/dedup_sequences.py --dataset h2h_zoo first.")
    with open(DEDUP_TAXIDS_PATH) as f:
        dedup_taxids = set(int(line.strip()) for line in f if line.strip())
    n_before = len(matched)
    matched = matched & dedup_taxids
    print(f"  --dedup: filtered {n_before} -> {len(matched)} taxa "
          f"({n_before - len(matched)} near-duplicate taxa removed)")

# Build X, y_h2h, y_zoo for matched viruses
matched_order = sorted(matched)
label_lookup  = human_df.set_index("virus_taxid")[["human_to_human", "zoonotic"]]
emb_lookup    = {tid: emb for tid, emb in zip(taxids_agg.tolist(), embeddings_agg)}

X     = np.array([emb_lookup[t]                    for t in matched_order])
y_h2h = np.array([label_lookup.loc[t, "human_to_human"] for t in matched_order])
y_zoo = np.array([label_lookup.loc[t, "zoonotic"]       for t in matched_order])

print(f"\nFinal dataset shape: X={X.shape}, y_h2h={y_h2h.shape}, y_zoo={y_zoo.shape}")
print(f"  h2h  — {y_h2h.sum()} pos / {(y_h2h==0).sum()} neg ({y_h2h.mean():.1%} positive)")
print(f"  zoo  — {y_zoo.sum()} pos / {(y_zoo==0).sum()} neg ({y_zoo.mean():.1%} positive)")

# Save matched labels for inspection
matched_df = pd.DataFrame({
    "virus_taxid": matched_order,
    "human_to_human": y_h2h,
    "zoonotic": y_zoo
})
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

    A fixed 0.5 threshold understates MCC/F1 on imbalanced tasks (Zoonotic
    is ~31% positive, more so under strict labels) — Youden's J picks the
    ROC point that best trades off sensitivity/specificity instead.
    """
    fpr, tpr, thresholds = roc_curve(y, probs)
    return thresholds[np.argmax(tpr - fpr)]


def evaluate(X, y, task_name):
    """Cross-validated evaluation for all classifiers on one task."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    results = {}
    all_probs = {}
    all_thresholds = {}

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
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{task_name} — Confusion Matrices", fontsize=13)
    for ax, (clf_name, probs) in zip(axes, all_probs.items()):
        preds = (probs >= all_thresholds[clf_name]).astype(int)
        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(clf_name, fontsize=9)
    plt.tight_layout()
    task_safe = task_name.replace(" ", "_")
    fname = out_dir / f"{task_safe}_confusion_matrices.png"
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

    axes[0].plot([0,1],[0,1],"k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8)

    baseline = y.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", lw=0.8, label=f"Baseline ({baseline:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    fname = out_dir / f"{task_name.replace(' ','_')}_roc_pr.png"
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
    fname = out_dir / f"{task_name.replace(' ','_')}_calibration.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")


# ── 6. RUN BOTH TASKS ────────────────────────────────────────────────────────
all_results = {}

results_h2h, probs_h2h, thresholds_h2h = evaluate(X, y_h2h, "Human-to-Human Transmissibility")
plot_roc_pr(y_h2h, probs_h2h, "Human-to-Human", FIGURES_DIR)
plot_confusion_matrix(y_h2h, probs_h2h, thresholds_h2h, "Human-to-Human", FIGURES_DIR)
plot_calibration(y_h2h, probs_h2h, "Human-to-Human", FIGURES_DIR)
all_results["human_to_human"] = results_h2h

results_zoo, probs_zoo, thresholds_zoo = evaluate(X, y_zoo, "Zoonotic Spillover")
plot_roc_pr(y_zoo, probs_zoo, "Zoonotic Spillover", FIGURES_DIR)
plot_confusion_matrix(y_zoo, probs_zoo, thresholds_zoo, "Zoonotic Spillover", FIGURES_DIR)
plot_calibration(y_zoo, probs_zoo, "Zoonotic Spillover", FIGURES_DIR)
all_results["zoonotic"] = results_zoo

# ── 7. PHYLOGENETIC / FAMILY HOLD-OUTS ───────────────────────────────────────
# Flaviviridae is the headline H2H generalisation test (an emerging-RNA-virus
# family withheld entirely). It cannot, however, evaluate the zoonotic task:
# Flaviviridae is almost entirely zoonotic-positive, so the held-out set is
# near-class-homogeneous and AUROC is degenerate. Adenoviridae is added as a
# class-balanced hold-out (≈23 zoonotic-pos / 38 zoonotic-neg after dedup) so the
# zoonotic phenotype gets a meaningful out-of-distribution test too.
# Keys: Flaviviridae results are saved unsuffixed (holdout_h2h / holdout_zoonotic)
# for backward compatibility; Adenoviridae results carry an "_adenoviridae" suffix.
family_lookup = human_df.set_index("virus_taxid")["virus_family"]

HOLDOUTS = [
    ("Flaviviridae", [("H2H", y_h2h), ("Zoonotic", y_zoo)]),
    ("Adenoviridae", [("H2H", y_h2h), ("Zoonotic", y_zoo)]),
]

for HOLDOUT_FAMILY, tasks in HOLDOUTS:
    print("")
    print("=" * 70)
    print(f"PHYLOGENETIC HOLD-OUT: {HOLDOUT_FAMILY}")
    print("=" * 70)
    holdout_mask = np.array([family_lookup.get(t, "") == HOLDOUT_FAMILY for t in matched_order])
    train_mask = ~holdout_mask
    print(f"  Training viruses: {train_mask.sum()}")
    print(f"  Hold-out viruses: {holdout_mask.sum()}")
    if holdout_mask.sum() < 5:
        print("  Fewer than 5 hold-out taxa; skipping this family.")
        continue

    X_train_ho, X_test_ho = X[train_mask], X[holdout_mask]
    fam_suffix = "" if HOLDOUT_FAMILY == "Flaviviridae" else f"_{HOLDOUT_FAMILY.lower()}"
    for task_name, y_all in tasks:
        y_train_ho = y_all[train_mask]
        y_test_ho  = y_all[holdout_mask]
        print(f"Task: {task_name} | Hold-out: {y_test_ho.sum()} pos / {(y_test_ho==0).sum()} neg")
        for clf_name, clf in CLASSIFIERS.items():
            if clf_name == "Random (Baseline)": continue
            clf.fit(X_train_ho, y_train_ho)
            probs = clf.predict_proba(X_test_ho)[:, 1]
            try:
                auroc = roc_auc_score(y_test_ho, probs)
                auprc = average_precision_score(y_test_ho, probs)
                print(f"    {clf_name:<25} AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
                all_results.setdefault(f"holdout_{task_name.lower()}{fam_suffix}", {})[clf_name] = {"AUROC": round(auroc,4), "AUPRC": round(auprc,4)}
            except Exception as e:
                print(f"    {clf_name}: {e}")

# ── 8. PRINT SUMMARY TABLE  ───────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for task, res in all_results.items():
    print(f"\n{task}")
    header = f"{'Classifier':<25} {'AUROC':>7} {'AUPRC':>7} {'MCC':>7} {'F1mac':>7} {'Brier':>7}"
    print(header)
    print("-" * len(header))
    for clf, m in res.items():
        if 'MCC' in m:
            print(f"{clf:<25} {m['AUROC']:>7.4f} {m['AUPRC']:>7.4f} {m['MCC']:>7.4f} {m['F1_macro']:>7.4f} {m['Brier']:>7.4f}")
        else:
            print(f"{clf:<25} {m['AUROC']:>7.4f} {m['AUPRC']:>7.4f}")

# ── 8. SAVE RESULTS JSON ─────────────────────────────────────────────────────
results_path = RESULTS_DIR / "tier1_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved: {results_path}")
print("Done! ✓")
