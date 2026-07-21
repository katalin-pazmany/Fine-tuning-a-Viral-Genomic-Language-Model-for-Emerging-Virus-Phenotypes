"""
Vector-Borne Baseline Comparisons (Tickets 2, 3, 5, 6, 7 — Vector-Borne)
============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Same baseline suite as scripts/baseline_comparisons.py + scripts/blast_baseline.py
(majority-class, GC-content+length, k-mer composition, BLAST nearest-neighbour),
applied to the Vector-Borne phenotype. ZOVER-derived viruses don't have
virus_family info without an extra taxonomy lookup, so the generalisation
hold-out (Ticket 7) uses host_type instead: train on mosquito-only +
mosquito/tick-dual viruses, hold out all tick-only viruses — the same
"can it generalise beyond what it trained on" logic as the Flaviviridae
hold-out, just grouped by vector ecology instead of taxonomy.

Usage:
    conda activate viral-phenotype
    python scripts/vector_borne_baselines.py --dedup

Requires:
    outputs/dedup/vector_borne_representative.fasta   — from dedup_sequences.py
    data/vector_borne_labels.csv

Outputs (written to outputs/baselines_vector_borne[_dedup]/):
    baseline_results.json   — majority/GC+length/k-mer
    blast_results.json      — BLAST nearest-neighbour
"""

import argparse
import itertools
import json
import subprocess
import tempfile
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
OUT_DIR = OUTPUTS_DIR / ("baselines_vector_borne_dedup" if args.dedup else "baselines_vector_borne")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_FASTA = OUTPUTS_DIR / "dedup" / "vector_borne_representative.fasta"
DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "vector_borne_deduplicated_taxids.txt"
LABELS_PATH = BASE_DIR / "data" / "vector_borne_labels.csv"
K = 4

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 1. LOAD REPRESENTATIVE SEQUENCES ─────────────────────────────────────────
if not REPRESENTATIVE_FASTA.exists():
    raise SystemExit(
        f"{REPRESENTATIVE_FASTA} not found — run "
        "'python scripts/dedup_sequences.py --dataset vector_borne' first "
        "(it builds this file as a side effect even if you don't use --dedup)."
    )

print(f"Loading representative sequences from {REPRESENTATIVE_FASTA}...")
taxid_seq = {}
for record in SeqIO.parse(REPRESENTATIVE_FASTA, "fasta"):
    taxid_seq[int(record.id)] = str(record.seq)
print(f"  {len(taxid_seq)} representative sequences loaded")

# ── 2. LOAD LABELS ─────────────────────────────────────────────────────────────
label_df = pd.read_csv(LABELS_PATH)
label_lookup = label_df.set_index("virus_taxid")[["vector_borne", "host_types"]]
matched = [t for t in taxid_seq if t in label_lookup.index]

if args.dedup:
    if not DEDUP_TAXIDS_PATH.exists():
        raise SystemExit(f"--dedup requested but {DEDUP_TAXIDS_PATH} not found — "
                          f"run scripts/dedup_sequences.py --dataset vector_borne first.")
    with open(DEDUP_TAXIDS_PATH) as f:
        dedup_taxids = set(int(line.strip()) for line in f if line.strip())
    n_before = len(matched)
    matched = [t for t in matched if t in dedup_taxids]
    print(f"  --dedup: filtered {n_before} -> {len(matched)} taxa")

matched = sorted(matched)
y_all = np.array([label_lookup.loc[t, "vector_borne"] for t in matched])
host_types_all = np.array([label_lookup.loc[t, "host_types"] for t in matched])
print(f"  Matched: {len(matched)} taxa, {y_all.sum()} pos / {(y_all==0).sum()} neg")
print(f"  Host types: mosquito={sum('mosquito' in h for h in host_types_all)}  "
      f"tick={sum('tick' in h for h in host_types_all)}")

# ── 3. FEATURES (GC+length, k-mer) ────────────────────────────────────────────
def gc_length_features(seq: str) -> np.ndarray:
    seq = seq.upper().replace("N", "")
    length = len(seq)
    gc = (seq.count("G") + seq.count("C")) / max(length, 1)
    return np.array([gc, np.log10(max(length, 1))])

BASES = "ACGT"
KMERS = ["".join(p) for p in itertools.product(BASES, repeat=K)]
KMER_INDEX = {k: i for i, k in enumerate(KMERS)}

def kmer_features(seq: str) -> np.ndarray:
    seq = "".join(c for c in seq.upper() if c in BASES)
    counts = np.zeros(len(KMERS))
    for i in range(len(seq) - K + 1):
        idx = KMER_INDEX.get(seq[i:i+K])
        if idx is not None:
            counts[idx] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts

# Codon-usage baseline (Ticket 4): 64-dim codon frequencies from open reading
# frames (>=30 codons, between stops) across all six frames — reading-frame aware.
_COMP = str.maketrans("ACGT", "TGCA")
_STOPS = {"TAA", "TAG", "TGA"}
CODONS = ["".join(p) for p in itertools.product(BASES, repeat=3)]
CODON_INDEX = {c: i for i, c in enumerate(CODONS)}

def codon_features(seq: str, min_orf: int = 30) -> np.ndarray:
    seq = "".join(c for c in seq.upper() if c in BASES)
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
gc_len_matrix = np.array([gc_length_features(taxid_seq[t]) for t in matched])
kmer_matrix = np.array([kmer_features(taxid_seq[t]) for t in matched])
codon_matrix = np.array([codon_features(taxid_seq[t]) for t in matched])

# ── 4. HOLD-OUT SPLIT: tick-only vs rest (Ticket 7, host-type analogue) ──────
holdout_mask = np.array(["tick" in h and "mosquito" not in h for h in host_types_all])
train_mask = ~holdout_mask
print(f"\nHold-out (tick-only): {holdout_mask.sum()} taxa, train: {train_mask.sum()} taxa")

def youden_threshold(y, probs):
    fpr, tpr, thresholds = roc_curve(y, probs)
    return thresholds[np.argmax(tpr - fpr)]

# ── 5. MAJORITY / GC+LENGTH / K-MER BASELINES ────────────────────────────────
BASELINES = {
    "Majority-Class": ("majority", None, None),
    "GC-Content + Length": ("model", gc_len_matrix, Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])),
    "k-mer Composition (k=4)": ("model", kmer_matrix, Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])),
    "Codon Usage": ("model", codon_matrix, Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])),
}

all_results = {}
print(f"\n{'='*60}\nSklearn baselines\n{'='*60}")
for name, (kind, X, clf_template) in BASELINES.items():
    if kind == "majority":
        dummy = DummyClassifier(strategy="most_frequent")
        probs = cross_val_predict(dummy, np.zeros((len(y_all), 1)), y_all, cv=CV, method="predict_proba")[:, 1]
    else:
        probs = cross_val_predict(clf_template, X, y_all, cv=CV, method="predict_proba")[:, 1]

    if probs.std() == 0:
        auroc, auprc, threshold = float("nan"), float("nan"), 0.5
    else:
        auroc = roc_auc_score(y_all, probs)
        auprc = average_precision_score(y_all, probs)
        threshold = youden_threshold(y_all, probs)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "AUROC": round(auroc, 4) if not np.isnan(auroc) else None,
        "AUPRC": round(auprc, 4) if not np.isnan(auprc) else None,
        "MCC": round(matthews_corrcoef(y_all, preds), 4),
        "F1_macro": round(f1_score(y_all, preds, average="macro"), 4),
        "Brier": round(brier_score_loss(y_all, probs), 4),
        "Threshold": round(float(threshold), 4),
    }

    # Hold-out: tick-only viruses, never trained on
    holdout_metrics = None
    if kind != "majority" and holdout_mask.sum() >= 5 and len(set(y_all[train_mask])) > 1:
        clf_ho = clf_template
        clf_ho.fit(X[train_mask], y_all[train_mask])
        probs_ho = clf_ho.predict_proba(X[holdout_mask])[:, 1]
        y_ho = y_all[holdout_mask]
        if len(set(y_ho)) > 1 and probs_ho.std() > 0:
            holdout_metrics = {
                "AUROC": round(roc_auc_score(y_ho, probs_ho), 4),
                "AUPRC": round(average_precision_score(y_ho, probs_ho), 4),
                "n_holdout": int(holdout_mask.sum()),
            }
        else:
            holdout_metrics = {"AUROC": None, "n_holdout": int(holdout_mask.sum())}
    metrics["Holdout_TickOnly"] = holdout_metrics
    all_results[name] = metrics

    auroc_str = f"{metrics['AUROC']:.4f}" if metrics["AUROC"] is not None else "n/a"
    ho_str = f"{holdout_metrics['AUROC']:.4f}" if holdout_metrics and holdout_metrics.get("AUROC") is not None else "n/a"
    print(f"  {name:<28} AUROC={auroc_str}  MCC={metrics['MCC']:.4f}  (holdout AUROC={ho_str})")

out_path = OUT_DIR / "baseline_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")

# ── 6. BLAST BASELINE ─────────────────────────────────────────────────────────
def write_fasta(taxids, path):
    with open(path, "w") as f:
        for t in taxids:
            f.write(f">{t}\n{taxid_seq[t]}\n")

def blast_top_hits(query_taxids, db_taxids, tmpdir):
    query_fasta = tmpdir / "query.fasta"
    db_fasta = tmpdir / "db.fasta"
    write_fasta(query_taxids, query_fasta)
    write_fasta(db_taxids, db_fasta)
    db_prefix = tmpdir / "blastdb"
    subprocess.run(["makeblastdb", "-in", str(db_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
                    capture_output=True, check=True)
    result = subprocess.run(
        ["blastn", "-query", str(query_fasta), "-db", str(db_prefix),
         "-outfmt", "6 qseqid sseqid bitscore", "-max_target_seqs", "1", "-num_threads", "8"],
        capture_output=True, text=True, check=True)
    best_hit, best_score = {}, {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        qid, sid, score = line.split("\t")
        qid, sid, score = int(qid), int(sid), float(score)
        if qid not in best_score or score > best_score[qid]:
            best_score[qid] = score
            best_hit[qid] = sid
    return best_hit

print(f"\n{'='*60}\nBLAST nearest-neighbour baseline\n{'='*60}")
matched_arr = np.array(matched)
taxid_to_pos = {t: i for i, t in enumerate(matched)}
all_probs = np.full(len(matched_arr), np.nan)
with tempfile.TemporaryDirectory() as tmpdir_str:
    tmpdir = Path(tmpdir_str)
    for fold, (train_idx, test_idx) in enumerate(CV.split(matched_arr, y_all)):
        train_t = matched_arr[train_idx].tolist()
        test_t = matched_arr[test_idx].tolist()
        best_hit = blast_top_hits(test_t, train_t, tmpdir)
        for i, t in zip(test_idx, test_t):
            if t in best_hit and best_hit[t] in taxid_to_pos:
                all_probs[i] = y_all[taxid_to_pos[best_hit[t]]]
        print(f"  Fold {fold+1}/5: {len(best_hit)}/{len(test_t)} queries got a BLAST hit")

no_hit = np.isnan(all_probs)
print(f"  No-hit queries: {no_hit.sum()}/{len(all_probs)}")
majority = int(round(y_all.mean()))
all_probs[no_hit] = majority
preds = (all_probs >= 0.5).astype(int)
blast_metrics = {
    "AUROC": round(roc_auc_score(y_all, all_probs), 4),
    "AUPRC": round(average_precision_score(y_all, all_probs), 4),
    "MCC": round(matthews_corrcoef(y_all, preds), 4),
    "F1_macro": round(f1_score(y_all, preds, average="macro"), 4),
    "n_no_hit": int(no_hit.sum()),
}
print(f"  AUROC={blast_metrics['AUROC']:.4f}  MCC={blast_metrics['MCC']:.4f}")

# BLAST hold-out: tick-only
train_t = matched_arr[train_mask].tolist()
test_t = matched_arr[holdout_mask].tolist()
y_ho = y_all[holdout_mask]
with tempfile.TemporaryDirectory() as tmpdir_str:
    tmpdir = Path(tmpdir_str)
    best_hit = blast_top_hits(test_t, train_t, tmpdir)
probs_ho = np.array([y_all[taxid_to_pos[best_hit[t]]] if t in best_hit and best_hit[t] in taxid_to_pos else np.nan for t in test_t])
no_hit_ho = np.isnan(probs_ho)
probs_ho[no_hit_ho] = int(round(y_all[train_mask].mean()))
if len(set(y_ho)) > 1 and probs_ho.std() > 0:
    holdout_blast = {
        "AUROC": round(roc_auc_score(y_ho, probs_ho), 4),
        "AUPRC": round(average_precision_score(y_ho, probs_ho), 4),
        "n_holdout": len(test_t),
        "n_no_hit": int(no_hit_ho.sum()),
    }
else:
    holdout_blast = {"AUROC": None, "n_holdout": len(test_t), "n_no_hit": int(no_hit_ho.sum())}
blast_metrics["Holdout_TickOnly"] = holdout_blast
print(f"  Hold-out (tick-only): AUROC={holdout_blast.get('AUROC')}  no-hit={holdout_blast['n_no_hit']}/{holdout_blast['n_holdout']}")

blast_out_path = OUT_DIR / "blast_results.json"
with open(blast_out_path, "w") as f:
    json.dump({"vector_borne": blast_metrics}, f, indent=2)
print(f"\nSaved: {blast_out_path}")
print("Done! ✓")
