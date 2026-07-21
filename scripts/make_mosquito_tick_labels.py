"""
Reconstruct the mosquito+tick-only (pre-biting-midge) vector-borne labels
=========================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

The biting-midge expansion (scripts/derive_vector_borne_labels_v2.py) overwrote
data/vector_borne_labels.csv in place, so the original mosquito+tick-only label
set no longer exists on disk. This script deterministically reconstructs it from
the current full label file so the "does adding biting midges change the
conclusions?" ablation can be re-run for all three tiers WITHOUT re-fetching or
clobbering the full (with-midge) labels.

Reconstruction rule (host_types is a comma-separated set of {mosquito, tick,
biting_midge}):
  * taxa whose ONLY vector association is biting_midge  -> dropped (they were
    introduced by the midge expansion and were not in the mosquito+tick dataset)
  * taxa associated with mosquito and/or tick as well as biting_midge -> kept,
    with the biting_midge token stripped from host_types (they already existed in
    the mosquito+tick dataset via their mosquito/tick association)
  * all other columns, including the vector_borne label, are unchanged

Usage:
    conda activate viral-phenotype
    python scripts/make_mosquito_tick_labels.py
Writes:
    data/vector_borne_labels_mosquito_tick.csv
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FULL_PATH = DATA_DIR / "vector_borne_labels.csv"
OUT_PATH  = DATA_DIR / "vector_borne_labels_mosquito_tick.csv"

df = pd.read_csv(FULL_PATH)
for col in ("virus_taxid", "vector_borne", "host_types"):
    if col not in df.columns:
        sys.exit(f"ERROR: expected column '{col}' not in {FULL_PATH.name}; "
                 f"found columns: {list(df.columns)}")

n_full = len(df)
pos_full = int(df["vector_borne"].sum())


def strip_midge(host_types: str):
    """Return host_types with biting_midge removed, or None if it was midge-only."""
    tokens = [t.strip() for t in str(host_types).split(",") if t.strip()]
    kept = [t for t in tokens if t != "biting_midge"]
    return ",".join(sorted(set(kept))) if kept else None


df["_new_host_types"] = df["host_types"].apply(strip_midge)
midge_only = df["_new_host_types"].isna()
mt = df[~midge_only].copy()
mt["host_types"] = mt["_new_host_types"]
mt = mt.drop(columns="_new_host_types")

mt.to_csv(OUT_PATH, index=False)

print(f"Full (with-midge) dataset:  {n_full} taxa  ({pos_full} vector-borne-positive)")
print(f"Dropped as biting-midge-only: {int(midge_only.sum())} taxa")
print(f"Mosquito+tick-only dataset:  {len(mt)} taxa  ({int(mt['vector_borne'].sum())} positive)")
print(f"Saved: {OUT_PATH}")
