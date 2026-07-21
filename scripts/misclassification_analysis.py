"""
Misclassification-by-Group Analysis
=====================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

The supervisor's request: confusion matrices give aggregate TP/FP/TN/FN counts, but
don't say *which* viruses are misclassified or whether errors concentrate
in specific groups — especially "mixed" groups that contain both positive
and negative examples (the hard cases where embedding space likely overlaps).

For H2H/Zoonotic, the grouping variable is virus_family (from the reference dataset).
For Vector-Borne there's no family info without an extra NCBI taxonomy
lookup, so this uses host_types (mosquito/tick/both) instead — a more
directly available and arguably more relevant grouping for that phenotype.

Reuses the best Tier 1 classifier's out-of-fold predictions (same
5-fold cross_val_predict already used for the headline Tier 1 numbers) —
no retraining beyond what Tier 1 already does, just joins predictions
back to virus identity instead of only reporting aggregate metrics.

Decision thresholds match what's already reported elsewhere: 0.5 for
H2H/Zoonotic (matches the existing tier1_pipeline.py confusion matrices),
Youden's J for Vector-Borne (matches tier1_vector_borne.py's fix).

Usage:
    conda activate viral-phenotype
    python scripts/misclassification_analysis.py

Outputs (written to outputs/misclassification/):
    {task}_per_virus.csv   — virus_taxid, group, true label, predicted, correct
    {task}_by_group.csv    — per-group counts, error rate, mixed-group flag
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dedup', action='store_true',
                     help='Filter to CD-HIT-deduplicated taxa to remove near-duplicate-genome '
                          'leakage across CV folds')
args = parser.parse_args()

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_DIR = OUTPUTS_DIR / ("misclassification_dedup" if args.dedup else "misclassification")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "h2h_zoo_deduplicated_taxids.txt"

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


def best_oof_predictions(X, y, use_youden: bool):
    """Cross-validated probs from the best-AUROC classifier, plus predictions."""
    best_name, best_probs, best_auroc = None, None, -1
    for name, clf in CLASSIFIERS.items():
        probs = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]
        auroc = roc_auc_score(y, probs)
        if auroc > best_auroc:
            best_name, best_probs, best_auroc = name, probs, auroc
    threshold = youden_threshold(y, best_probs) if use_youden else 0.5
    preds = (best_probs >= threshold).astype(int)
    return best_name, best_probs, preds, threshold, best_auroc


def analyse_task(task_label: str, taxids, X, y, groups, group_col_name: str, use_youden: bool):
    print(f"\n{'='*70}")
    print(f"Task: {task_label}")
    print(f"{'='*70}")
    clf_name, probs, preds, threshold, auroc = best_oof_predictions(X, y, use_youden)
    print(f"  Best classifier: {clf_name} (AUROC={auroc:.4f}, threshold={threshold:.4f})")

    per_virus = pd.DataFrame({
        "virus_taxid": taxids,
        group_col_name: groups,
        "true_label": y,
        "predicted_prob": np.round(probs, 4),
        "predicted_label": preds,
    })
    per_virus["correct"] = (per_virus["true_label"] == per_virus["predicted_label"])
    per_virus["error_type"] = np.select(
        [
            (per_virus.true_label == 1) & (per_virus.predicted_label == 0),
            (per_virus.true_label == 0) & (per_virus.predicted_label == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )

    fname_pv = OUT_DIR / f"{task_label.replace(' ', '_')}_per_virus.csv"
    per_virus.to_csv(fname_pv, index=False)
    print(f"  Saved: {fname_pv.relative_to(BASE_DIR)}")

    # ── Group-level breakdown ────────────────────────────────────────────────
    by_group = per_virus.groupby(group_col_name).agg(
        n_viruses=("virus_taxid", "count"),
        n_positive=("true_label", "sum"),
        n_errors=("correct", lambda s: (~s).sum()),
    ).reset_index()
    by_group["n_negative"] = by_group["n_viruses"] - by_group["n_positive"]
    by_group["error_rate"] = (by_group["n_errors"] / by_group["n_viruses"]).round(3)
    by_group["mixed_group"] = (by_group["n_positive"] > 0) & (by_group["n_negative"] > 0)
    by_group = by_group.sort_values("error_rate", ascending=False)

    fname_bg = OUT_DIR / f"{task_label.replace(' ', '_')}_by_group.csv"
    by_group.to_csv(fname_bg, index=False)
    print(f"  Saved: {fname_bg.relative_to(BASE_DIR)}")

    mixed = by_group[by_group["mixed_group"]]
    pure = by_group[~by_group["mixed_group"]]
    print(f"\n  {group_col_name}: {len(by_group)} groups total "
          f"({len(mixed)} mixed-label, {len(pure)} pure-label)")
    if len(mixed):
        print(f"  Mixed-group mean error rate: {mixed['error_rate'].mean():.3f}  "
              f"(weighted by size: {(mixed['n_errors'].sum()/mixed['n_viruses'].sum()):.3f})")
    if len(pure):
        print(f"  Pure-group   mean error rate: {pure['error_rate'].mean():.3f}  "
              f"(weighted by size: {(pure['n_errors'].sum()/pure['n_viruses'].sum()):.3f})")
    print(f"\n  Top 5 worst groups (min 3 viruses):")
    worst = by_group[by_group["n_viruses"] >= 3].head(5)
    for _, r in worst.iterrows():
        tag = "MIXED" if r["mixed_group"] else "pure"
        print(f"    {r[group_col_name]:<30} n={r['n_viruses']:<4} "
              f"errors={r['n_errors']:<3} rate={r['error_rate']:.2f}  [{tag}]")


# ══════════════════════════════════════════════════════════════════════════
# H2H / ZOONOTIC (relaxed + strict) — grouped by virus_family
# ══════════════════════════════════════════════════════════════════════════
def load_h2h_zoo_data(strict: bool):
    data = np.load(OUTPUTS_DIR / "refseq_embeddings.npz")
    embeddings = data["embeddings"]
    seq_ids = data["accessions"]
    meta = pd.read_csv(OUTPUTS_DIR / "refseq_metadata.csv")

    if strict:
        human_df = pd.read_csv(BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv")
        human_df = human_df.rename(columns={"zoonotic_strict": "zoonotic"})
    else:
        human_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="human")
        interactions_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="interactions")
        human_host_taxids = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
        non_human_taxids = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
        human_df["zoonotic"] = human_df["virus_taxid"].isin(human_host_taxids & non_human_taxids).astype(int)

    cache_df = pd.read_csv(OUTPUTS_DIR / "accession_taxid_cache.csv")
    cache = dict(zip(cache_df["accession"], cache_df["taxid"]))
    meta["virus_taxid"] = meta["accession"].map(cache)
    meta = meta.dropna(subset=["virus_taxid"])
    meta["virus_taxid"] = meta["virus_taxid"].astype(int)

    valid_accs = set(seq_ids.tolist())
    meta = meta[meta["accession"].isin(valid_accs)].copy()
    meta["embedding_idx"] = meta["accession"].map({a: i for i, a in enumerate(seq_ids)})

    taxid_to_indices = {}
    for _, row in meta.iterrows():
        taxid_to_indices.setdefault(row["virus_taxid"], []).append(int(row["embedding_idx"]))

    taxids_agg = list(taxid_to_indices.keys())
    embeddings_agg = np.array([embeddings[idx].mean(axis=0) for idx in taxid_to_indices.values()])

    label_taxids = set(human_df["virus_taxid"])
    matched = sorted(set(taxids_agg) & label_taxids)

    if args.dedup:
        if not DEDUP_TAXIDS_PATH.exists():
            raise SystemExit(f"--dedup requested but {DEDUP_TAXIDS_PATH} not found — "
                              f"run scripts/dedup_sequences.py --dataset h2h_zoo first.")
        with open(DEDUP_TAXIDS_PATH) as f:
            dedup_taxids = set(int(line.strip()) for line in f if line.strip())
        n_before = len(matched)
        matched = [t for t in matched if t in dedup_taxids]
        print(f"  --dedup: filtered {n_before} -> {len(matched)} taxa")

    label_lookup = human_df.set_index("virus_taxid")[["human_to_human", "zoonotic", "virus_family"]]
    emb_lookup = {t: e for t, e in zip(taxids_agg, embeddings_agg)}

    X = np.array([emb_lookup[t] for t in matched])
    y_h2h = np.array([label_lookup.loc[t, "human_to_human"] for t in matched])
    y_zoo = np.array([label_lookup.loc[t, "zoonotic"] for t in matched])
    family = np.array([label_lookup.loc[t, "virus_family"] for t in matched])
    return matched, X, y_h2h, y_zoo, family


# human_to_human is the same column regardless of --strict (only the
# zoonotic label has a relaxed/strict distinction — the supervisor's 22 lab-artefact
# removals), so H2H only needs to run once.
taxids, X, y_h2h, y_zoo, family = load_h2h_zoo_data(strict=False)
print(f"\n\n{'#'*70}\n# H2H\n{'#'*70}")
analyse_task("H2H", taxids, X, y_h2h, family, "virus_family", use_youden=False)

for strict, label_tag in [(False, "relaxed"), (True, "strict")]:
    print(f"\n\n{'#'*70}\n# ZOONOTIC — {label_tag.upper()}\n{'#'*70}")
    taxids, X, y_h2h, y_zoo, family = load_h2h_zoo_data(strict)
    analyse_task(f"Zoonotic_{label_tag}", taxids, X, y_zoo, family, "virus_family", use_youden=False)


# ══════════════════════════════════════════════════════════════════════════
# VECTOR-BORNE — grouped by host_types (mosquito/tick/both)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'#'*70}\n# VECTOR-BORNE\n{'#'*70}")

data = np.load(OUTPUTS_DIR / "refseq_embeddings.npz")
embeddings = data["embeddings"]
seq_ids = data["accessions"]
meta = pd.read_csv(OUTPUTS_DIR / "refseq_metadata.csv")
label_df = pd.read_csv(BASE_DIR / "data" / "vector_borne_labels.csv")

cache_df = pd.read_csv(OUTPUTS_DIR / "accession_taxid_cache.csv")
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"]))
stripped_to_taxid = {}
for acc_ver, tid in acc_to_taxid.items():
    stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)

meta["virus_taxid"] = meta["accession"].map(acc_to_taxid)
still_missing = meta["virus_taxid"].isna()
meta.loc[still_missing, "virus_taxid"] = meta.loc[still_missing, "accession"].str.split(".").str[0].map(stripped_to_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

valid_accs = set(seq_ids.tolist())
meta = meta[meta["accession"].isin(valid_accs)].copy()
meta["embedding_idx"] = meta["accession"].map({a: i for i, a in enumerate(seq_ids)})

taxid_to_indices = {}
for _, row in meta.iterrows():
    taxid_to_indices.setdefault(row["virus_taxid"], []).append(int(row["embedding_idx"]))

taxids_agg = list(taxid_to_indices.keys())
embeddings_agg = np.array([embeddings[idx].mean(axis=0) for idx in taxid_to_indices.values()])

label_taxids = set(label_df["virus_taxid"])
matched = sorted(set(taxids_agg) & label_taxids)

if args.dedup:
    VB_DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "vector_borne_deduplicated_taxids.txt"
    if not VB_DEDUP_TAXIDS_PATH.exists():
        raise SystemExit(f"--dedup requested but {VB_DEDUP_TAXIDS_PATH} not found — "
                          f"run scripts/dedup_sequences.py --dataset vector_borne first.")
    with open(VB_DEDUP_TAXIDS_PATH) as f:
        vb_dedup_taxids = set(int(line.strip()) for line in f if line.strip())
    n_before = len(matched)
    matched = [t for t in matched if t in vb_dedup_taxids]
    print(f"  --dedup: filtered {n_before} -> {len(matched)} taxa")

label_lookup = label_df.set_index("virus_taxid")[["vector_borne", "host_types"]]
emb_lookup = {t: e for t, e in zip(taxids_agg, embeddings_agg)}

X_vb = np.array([emb_lookup[t] for t in matched])
y_vb = np.array([label_lookup.loc[t, "vector_borne"] for t in matched])
host_types_vb = np.array([label_lookup.loc[t, "host_types"] for t in matched])

analyse_task("Vector-Borne", matched, X_vb, y_vb, host_types_vb, "host_types", use_youden=True)

print(f"\n\nDone! All per-virus and per-group CSVs saved under {OUT_DIR}/")
