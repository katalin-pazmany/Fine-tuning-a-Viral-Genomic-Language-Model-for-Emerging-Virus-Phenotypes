"""
Baseline Comparisons: Majority-Class, GC-Content+Length, k-mer
==================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Three baselines to contextualise what Vir2vec's embeddings are actually
buying you over trivial/cheap alternatives:

    1. Majority-class baseline — always predicts the majority label.
       The true "floor": MCC=0 by construction. Distinct from the
       stratified-random DummyClassifier already used as the Tier 1
       "Random (Baseline)" row, which introduces controlled randomness
       rather than a fixed constant prediction.

    2. GC-content + length baseline — 2 features (GC%, log10 genome
       length) per virus, logistic regression. Answers "is Vir2vec
       learning real biology, or just picking up genome-length/family
       fingerprints a reviewer would ask about first?"

    3. k-mer composition baseline — tetranucleotide (k=4, 256-dim)
       frequency vector per virus, Gradient Boosting (sklearn, not
       LightGBM — avoids a new cluster dependency; same tree-boosting
       family). A much stronger/more convincing baseline than GC-content
       alone.

All three are evaluated identically to Tier 1 (5-fold CV, same metrics,
Youden's J threshold) AND on the same Flaviviridae phylogenetic hold-out
already used for Vir2vec, so the numbers are directly comparable.

Per-virus features are computed from the LONGEST single FASTA record
found for that taxid (a representative-genome choice, not an aggregate
across all deposited accessions — avoids inflating features for
well-studied viruses with many redundant submissions).

Usage:
    conda activate viral-phenotype
    python scripts/baseline_comparisons.py

Outputs (written to outputs/baselines/):
    baseline_results.json
    figures/  — ROC/PR comparison plots (Vir2vec Tier 1 best vs baselines)
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
from Bio import SeqIO

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, brier_score_loss, roc_curve
)

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_DIR = OUTPUTS_DIR / ("baselines_dedup" if args.dedup else "baselines")
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "h2h_zoo_deduplicated_taxids.txt"

FASTA_DIR = BASE_DIR / "data" / "raw_matched"
CACHE_CSV = OUTPUTS_DIR / "accession_taxid_cache.csv"
K = 4  # tetranucleotide k-mers

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── 1. LOAD LABELS (same pattern as misclassification_analysis.py) ──────────
def load_h2h_zoo_labels(strict: bool):
    if strict:
        strict_df = pd.read_csv(BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv")
        human_df = strict_df.rename(columns={"zoonotic_strict": "zoonotic"})
    else:
        human_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="human")
        interactions_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="interactions")
        human_host_taxids = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
        non_human_taxids = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
        human_df["zoonotic"] = human_df["virus_taxid"].isin(human_host_taxids & non_human_taxids).astype(int)
    return human_df


# ── 2. RESOLVE ACCESSION -> TAXID (exact + stripped fallback) ────────────────
print("Loading accession->taxid cache...")
cache_df = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))
stripped_to_taxid = {}
for acc_ver, tid in acc_to_taxid.items():
    stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)

def resolve_taxid(acc: str):
    return acc_to_taxid.get(acc) or stripped_to_taxid.get(acc.split(".")[0])


# ── 3. FIND THE LONGEST SEQUENCE PER TAXID (representative genome) ──────────
print("Scanning FASTA files for the longest sequence per virus taxid "
      "(this may take a minute)...")
label_df = load_h2h_zoo_labels(strict=False)  # union of taxids across relaxed/strict is the same set
label_taxids = set(label_df["virus_taxid"])

taxid_best_seq = {}  # taxid -> (length, seq_str)
n_scanned = 0
for fasta_path in sorted(FASTA_DIR.glob("*.fasta")):
    tid = resolve_taxid(fasta_path.stem)
    if not tid or tid not in label_taxids:
        continue
    n_scanned += 1
    try:
        record = next(SeqIO.parse(fasta_path, "fasta"))
    except StopIteration:
        continue
    seq = str(record.seq).upper()
    if tid not in taxid_best_seq or len(seq) > taxid_best_seq[tid][0]:
        taxid_best_seq[tid] = (len(seq), seq)
    if n_scanned % 2000 == 0:
        print(f"  Scanned {n_scanned} matching FASTA files...")

print(f"  Found representative sequences for {len(taxid_best_seq)} virus taxa "
      f"(out of {len(label_taxids)} labelled)")


# ── 4. FEATURE EXTRACTION ─────────────────────────────────────────────────────
def gc_length_features(seq: str) -> np.ndarray:
    seq = seq.replace("N", "")
    length = len(seq)
    gc = (seq.count("G") + seq.count("C")) / max(length, 1)
    return np.array([gc, np.log10(max(length, 1))])

BASES = "ACGT"
KMERS = ["".join(p) for p in __import__("itertools").product(BASES, repeat=K)]
KMER_INDEX = {k: i for i, k in enumerate(KMERS)}

def kmer_features(seq: str) -> np.ndarray:
    seq = "".join(c for c in seq if c in BASES)
    counts = np.zeros(len(KMERS))
    for i in range(len(seq) - K + 1):
        idx = KMER_INDEX.get(seq[i:i+K])
        if idx is not None:
            counts[idx] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts

# Codon-usage baseline (Ticket 4): 64-dim codon frequencies from open reading
# frames (>=30 codons, between stop codons) across all six frames. Reading-frame
# aware, so it is a genuine codon-usage signature rather than trinucleotide counts.
_COMP = str.maketrans("ACGT", "TGCA")
_STOPS = {"TAA", "TAG", "TGA"}
CODONS = ["".join(p) for p in __import__("itertools").product(BASES, repeat=3)]
CODON_INDEX = {c: i for i, c in enumerate(CODONS)}

def codon_features(seq: str, min_orf: int = 30) -> np.ndarray:
    seq = "".join(c for c in seq if c in BASES)
    counts = np.zeros(64)
    for strand in (seq, seq.translate(_COMP)[::-1]):
        for frame in range(3):
            orf = []
            for i in range(frame, len(strand) - 2, 3):
                c = strand[i:i+3]
                if c in _STOPS:
                    if len(orf) >= min_orf:
                        for cc in orf:
                            counts[CODON_INDEX[cc]] += 1
                    orf = []
                elif c in CODON_INDEX:
                    orf.append(c)
            if len(orf) >= min_orf:
                for cc in orf:
                    counts[CODON_INDEX[cc]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts

print("\nExtracting GC+length, k-mer and codon-usage features...")
taxids_ordered = sorted(taxid_best_seq.keys())
gc_len_matrix = np.array([gc_length_features(taxid_best_seq[t][1]) for t in taxids_ordered])
kmer_matrix = np.array([kmer_features(taxid_best_seq[t][1]) for t in taxids_ordered])
codon_matrix = np.array([codon_features(taxid_best_seq[t][1]) for t in taxids_ordered])
print(f"  GC+length matrix: {gc_len_matrix.shape}")
print(f"  k-mer matrix:      {kmer_matrix.shape}")
print(f"  codon matrix:      {codon_matrix.shape}")


# ── 5. BASELINES ──────────────────────────────────────────────────────────────
def youden_threshold(y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)
    return thresholds[np.argmax(tpr - fpr)]

BASELINES = {
    "Majority-Class": ("majority", None),
    "GC-Content + Length": ("model", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])),
    "k-mer Composition (k=4)": ("model", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])),
    "Codon Usage": ("model", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])),
}
FEATURES = {
    "Majority-Class": None,
    "GC-Content + Length": gc_len_matrix,
    "k-mer Composition (k=4)": kmer_matrix,
    "Codon Usage": codon_matrix,
}


def evaluate_baseline(name, y):
    kind, clf = BASELINES[name]
    X = FEATURES[name]
    if kind == "majority":
        dummy = DummyClassifier(strategy="most_frequent")
        probs = cross_val_predict(dummy, np.zeros((len(y), 1)), y, cv=CV, method="predict_proba")[:, 1]
    else:
        probs = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]

    # Majority-class gives constant probabilities — AUROC/AUPRC are
    # undefined/degenerate for a constant predictor; report them as NaN
    # rather than a misleading 0.5, and use a fixed 0.5 threshold since
    # there's no ROC curve to derive Youden's J from.
    if probs.std() == 0:
        auroc, auprc, threshold = float("nan"), float("nan"), 0.5
    else:
        auroc = roc_auc_score(y, probs)
        auprc = average_precision_score(y, probs)
        threshold = youden_threshold(y, probs)
    preds = (probs >= threshold).astype(int)
    return {
        "AUROC": round(auroc, 4) if not np.isnan(auroc) else None,
        "AUPRC": round(auprc, 4) if not np.isnan(auprc) else None,
        "MCC": round(matthews_corrcoef(y, preds), 4),
        "F1_macro": round(f1_score(y, preds, average="macro"), 4),
        "Brier": round(brier_score_loss(y, probs), 4),
        "Threshold": round(float(threshold), 4),
    }, probs, threshold


# ── 6. PHYLOGENETIC HOLD-OUT — Ticket 7 ──────────────────────────────────────
# Flaviviridae is the H2H generalisation family; Adenoviridae is the class-balanced
# zoonotic generalisation family (Flaviviridae is ~all zoonotic-positive and cannot
# evaluate the zoonotic task). Both are computed so Vir2vec and the baselines are
# compared on the identical hold-out.
def evaluate_holdout(name, y_all, family_all, holdout_family):
    kind, clf_template = BASELINES[name]
    holdout_mask = (family_all == holdout_family)
    train_mask = ~holdout_mask
    if holdout_mask.sum() < 5:
        return None

    y_train, y_test = y_all[train_mask], y_all[holdout_mask]
    if kind == "majority":
        dummy = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((train_mask.sum(), 1)), y_train)
        probs_test = dummy.predict_proba(np.zeros((holdout_mask.sum(), 1)))[:, 1]
    else:
        X = FEATURES[name]
        clf = clf_template
        clf.fit(X[train_mask], y_train)
        probs_test = clf.predict_proba(X[holdout_mask])[:, 1]

    if probs_test.std() == 0 or len(set(y_test)) < 2:
        return {"AUROC": None, "AUPRC": None, "n_holdout": int(holdout_mask.sum())}
    return {
        "AUROC": round(roc_auc_score(y_test, probs_test), 4),
        "AUPRC": round(average_precision_score(y_test, probs_test), 4),
        "n_holdout": int(holdout_mask.sum()),
    }


# ── 7. RUN EVERYTHING ─────────────────────────────────────────────────────────
dedup_taxids = None
if args.dedup:
    if not DEDUP_TAXIDS_PATH.exists():
        raise SystemExit(f"--dedup requested but {DEDUP_TAXIDS_PATH} not found — "
                          f"run scripts/dedup_sequences.py --dataset h2h_zoo first.")
    with open(DEDUP_TAXIDS_PATH) as f:
        dedup_taxids = set(int(line.strip()) for line in f if line.strip())

all_results = {}

for strict, label_tag in [(False, "relaxed"), (True, "strict")]:
    human_df = load_h2h_zoo_labels(strict)
    label_lookup = human_df.set_index("virus_taxid")[["human_to_human", "zoonotic", "virus_family"]]
    matched = [t for t in taxids_ordered if t in label_lookup.index]
    if dedup_taxids is not None:
        n_before = len(matched)
        matched = [t for t in matched if t in dedup_taxids]
        print(f"  --dedup ({label_tag}): filtered {n_before} -> {len(matched)} taxa")

    # Re-index feature matrices to the matched, label-available subset
    idx_map = {t: i for i, t in enumerate(taxids_ordered)}
    sel = [idx_map[t] for t in matched]
    gc_len_sub = gc_len_matrix[sel]
    kmer_sub = kmer_matrix[sel]
    codon_sub = codon_matrix[sel]
    FEATURES["GC-Content + Length"] = gc_len_sub
    FEATURES["k-mer Composition (k=4)"] = kmer_sub
    FEATURES["Codon Usage"] = codon_sub

    family_all = np.array([label_lookup.loc[t, "virus_family"] for t in matched])

    tasks = [("human_to_human", "H2H")] if not strict else []  # H2H identical regardless of strict
    tasks += [("zoonotic", f"Zoonotic_{label_tag}")]

    for label_col, task_name in tasks:
        y_all = np.array([label_lookup.loc[t, label_col] for t in matched])
        print(f"\n{'='*70}\nTask: {task_name}  (n={len(y_all)}, {y_all.sum()} pos)\n{'='*70}")

        task_results = {}
        for name in BASELINES:
            metrics, probs, threshold = evaluate_baseline(name, y_all)
            holdout_flavi = evaluate_holdout(name, y_all, family_all, "Flaviviridae")
            holdout_adeno = evaluate_holdout(name, y_all, family_all, "Adenoviridae")
            metrics["Holdout_Flaviviridae"] = holdout_flavi
            metrics["Holdout_Adenoviridae"] = holdout_adeno
            task_results[name] = metrics
            auroc_str = f"{metrics['AUROC']:.4f}" if metrics["AUROC"] is not None else "n/a (constant)"
            fa = holdout_flavi["AUROC"] if holdout_flavi else "n/a"
            ad = holdout_adeno["AUROC"] if holdout_adeno else "n/a"
            print(f"  {name:<28} AUROC={auroc_str}  MCC={metrics['MCC']:.4f}  "
                  f"(holdout Flavi={fa}  Adeno={ad})")
        all_results[task_name] = task_results

out_path = OUT_DIR / "baseline_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")
print("Done! ✓")
