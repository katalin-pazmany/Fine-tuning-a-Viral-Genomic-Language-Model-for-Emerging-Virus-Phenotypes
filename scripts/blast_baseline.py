"""
BLAST Nearest-Neighbour Baseline (Ticket 5)
==============================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

For every virus in a held-out fold, BLAST its representative genome
against the other folds' genomes and assign the label of its single
top hit (nearest neighbour by local alignment). Answers the most
obvious reviewer question: "couldn't you just BLAST it?"

Evaluated with the same 5-fold CV split as Tier 1 / the other baselines
(same random_state=42), so results are directly comparable. Also run on
the Flaviviridae phylogenetic hold-out (Ticket 7) — a BLAST classifier
should collapse to near-chance there if it's relying on close homology
rather than anything more general, which is exactly the point of that
comparison.

Usage:
    conda activate viral-phenotype
    python scripts/blast_baseline.py

Requires:
    outputs/dedup/h2h_zoo_representative.fasta   — from dedup_sequences.py
    (falls back to re-deriving representative sequences if that's missing)

Outputs (written to outputs/baselines/):
    blast_results.json
"""

import argparse
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_DIR = OUTPUTS_DIR / ("baselines_dedup" if args.dedup else "baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_TAXIDS_PATH = OUTPUTS_DIR / "dedup" / "h2h_zoo_deduplicated_taxids.txt"

REPRESENTATIVE_FASTA = OUTPUTS_DIR / "dedup" / "h2h_zoo_representative.fasta"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_h2h_zoo_labels(strict: bool):
    if strict:
        strict_df = pd.read_csv(BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv")
        return strict_df.rename(columns={"zoonotic_strict": "zoonotic"})
    human_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="human")
    interactions_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="interactions")
    human_host_taxids = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
    non_human_taxids = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
    human_df["zoonotic"] = human_df["virus_taxid"].isin(human_host_taxids & non_human_taxids).astype(int)
    return human_df


if not REPRESENTATIVE_FASTA.exists():
    raise SystemExit(
        f"{REPRESENTATIVE_FASTA} not found — run "
        "'python scripts/dedup_sequences.py --dataset h2h_zoo' first "
        "(it builds this file as a side effect even if you don't use the "
        "dedup output itself)."
    )

print(f"Loading representative sequences from {REPRESENTATIVE_FASTA}...")
taxid_seq = {}
for record in SeqIO.parse(REPRESENTATIVE_FASTA, "fasta"):
    taxid_seq[int(record.id)] = str(record.seq)
print(f"  {len(taxid_seq)} representative sequences loaded")


def write_fasta(taxids, path):
    with open(path, "w") as f:
        for t in taxids:
            f.write(f">{t}\n{taxid_seq[t]}\n")


def blast_top_hits(query_taxids, db_taxids, tmpdir):
    """Returns {query_taxid: best_hit_taxid} via blastn, top hit by bitscore."""
    query_fasta = tmpdir / "query.fasta"
    db_fasta = tmpdir / "db.fasta"
    write_fasta(query_taxids, query_fasta)
    write_fasta(db_taxids, db_fasta)

    db_prefix = tmpdir / "blastdb"
    subprocess.run(
        ["makeblastdb", "-in", str(db_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
        capture_output=True, check=True,
    )
    result = subprocess.run(
        ["blastn", "-query", str(query_fasta), "-db", str(db_prefix),
         "-outfmt", "6 qseqid sseqid bitscore", "-max_target_seqs", "1",
         "-num_threads", "8"],
        capture_output=True, text=True, check=True,
    )

    best_hit = {}
    best_score = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        qid, sid, score = line.split("\t")
        qid, sid, score = int(qid), int(sid), float(score)
        if qid not in best_score or score > best_score[qid]:
            best_score[qid] = score
            best_hit[qid] = sid
    return best_hit


def evaluate_task(y_lookup, matched_taxids, task_name):
    print(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")
    y_all = np.array([y_lookup[t] for t in matched_taxids])
    matched_taxids = np.array(matched_taxids)

    all_probs = np.full(len(matched_taxids), np.nan)
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for fold, (train_idx, test_idx) in enumerate(CV.split(matched_taxids, y_all)):
            train_taxids = matched_taxids[train_idx].tolist()
            test_taxids = matched_taxids[test_idx].tolist()
            best_hit = blast_top_hits(test_taxids, train_taxids, tmpdir)
            for i, t in zip(test_idx, test_taxids):
                if t in best_hit:
                    all_probs[i] = y_lookup[best_hit[t]]
            print(f"  Fold {fold+1}/5: {len(best_hit)}/{len(test_taxids)} queries got a BLAST hit")

    no_hit = np.isnan(all_probs)
    print(f"  No-hit queries (no BLAST match at all): {no_hit.sum()}/{len(all_probs)}")
    # No-hit queries: fall back to majority class (can't be "no prediction" for AUROC)
    majority = int(round(y_all.mean()))
    all_probs[no_hit] = majority

    preds = (all_probs >= 0.5).astype(int)
    metrics = {
        "AUROC": round(roc_auc_score(y_all, all_probs), 4),
        "AUPRC": round(average_precision_score(y_all, all_probs), 4),
        "MCC": round(matthews_corrcoef(y_all, preds), 4),
        "F1_macro": round(f1_score(y_all, preds, average="macro"), 4),
        "n_no_hit": int(no_hit.sum()),
    }
    print(f"  AUROC={metrics['AUROC']:.4f}  MCC={metrics['MCC']:.4f}")
    return metrics


def evaluate_holdout(y_lookup, matched_taxids, family_lookup, task_name, holdout_family="Flaviviridae"):
    matched_taxids = np.array(matched_taxids)
    family_all = np.array([family_lookup[t] for t in matched_taxids])
    holdout_mask = family_all == holdout_family
    train_mask = ~holdout_mask
    if holdout_mask.sum() < 5:
        return None

    train_taxids = matched_taxids[train_mask].tolist()
    test_taxids = matched_taxids[holdout_mask].tolist()
    y_test = np.array([y_lookup[t] for t in test_taxids])

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        best_hit = blast_top_hits(test_taxids, train_taxids, tmpdir)

    probs = np.array([y_lookup[best_hit[t]] if t in best_hit else np.nan for t in test_taxids])
    no_hit = np.isnan(probs)
    majority = int(round(np.array([y_lookup[t] for t in train_taxids]).mean()))
    probs[no_hit] = majority

    if len(set(y_test)) < 2:
        return {"AUROC": None, "n_holdout": len(test_taxids), "n_no_hit": int(no_hit.sum())}
    return {
        "AUROC": round(roc_auc_score(y_test, probs), 4),
        "AUPRC": round(average_precision_score(y_test, probs), 4),
        "n_holdout": len(test_taxids),
        "n_no_hit": int(no_hit.sum()),
    }


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
    matched = [t for t in taxid_seq if t in label_lookup.index]
    if dedup_taxids is not None:
        n_before = len(matched)
        matched = [t for t in matched if t in dedup_taxids]
        print(f"  --dedup ({label_tag}): filtered {n_before} -> {len(matched)} taxa")
    family_lookup = {t: label_lookup.loc[t, "virus_family"] for t in matched}

    tasks = [("human_to_human", "H2H")] if not strict else []
    tasks += [("zoonotic", f"Zoonotic_{label_tag}")]

    for label_col, task_name in tasks:
        y_lookup = {t: label_lookup.loc[t, label_col] for t in matched}
        metrics = evaluate_task(y_lookup, matched, task_name)
        metrics["Holdout_Flaviviridae"] = evaluate_holdout(
            y_lookup, matched, family_lookup, f"{task_name}_holdout", "Flaviviridae")
        metrics["Holdout_Adenoviridae"] = evaluate_holdout(
            y_lookup, matched, family_lookup, f"{task_name}_holdout", "Adenoviridae")
        all_results[task_name] = metrics

out_path = OUT_DIR / "blast_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")
print("Done! ✓")
