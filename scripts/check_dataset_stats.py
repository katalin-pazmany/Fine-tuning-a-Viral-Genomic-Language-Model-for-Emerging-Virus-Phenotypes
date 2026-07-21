"""
Dataset statistics check
========================
Reports how many labelled viruses match with available embeddings.
Run after any embedding update to verify dataset coverage.

Usage:
    python scripts/check_dataset_stats.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

cache = pd.read_csv(BASE_DIR / "outputs/accession_taxid_cache.csv")
acc_to_taxid = dict(zip(cache["accession"], cache["taxid"].astype(int)))

npz = np.load(BASE_DIR / "outputs/refseq_embeddings.npz")
meta = pd.read_csv(BASE_DIR / "outputs/refseq_metadata.csv")
meta["virus_taxid"] = meta["accession"].map(acc_to_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)
have_emb = set(meta["virus_taxid"])

human_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="human")
interactions_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="interactions")

human_hosts = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
non_human   = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
human_df["zoonotic"] = human_df["virus_taxid"].isin(human_hosts & non_human).astype(int)

matched    = set(human_df["virus_taxid"]) & have_emb
matched_df = human_df[human_df["virus_taxid"].isin(matched)]

print("=" * 50)
print("DATASET STATISTICS")
print("=" * 50)
print(f"Total embeddings:              {len(npz['embeddings']):,}")
print(f"Unique virus taxids:           {len(have_emb):,}")
print(f"Labelled viruses (Maya):       {len(human_df):,}")
print(f"Matched (labels + embeddings): {len(matched):,}")
print(f"\nHuman-to-human task:")
print(f"  Positive (h2h=1): {matched_df['human_to_human'].sum():,}")
print(f"  Negative (h2h=0): {(matched_df['human_to_human']==0).sum():,}")
print(f"\nZoonotic spillover task:")
print(f"  Positive (zoo=1): {matched_df['zoonotic'].sum():,}")
print(f"  Negative (zoo=0): {(matched_df['zoonotic']==0).sum():,}")
