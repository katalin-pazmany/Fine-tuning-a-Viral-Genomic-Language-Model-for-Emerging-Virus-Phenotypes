"""
Derive Vector-Borne Phenotype Labels
=====================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Takes the ZOVER mosquito/tick accession -> host-type mapping (produced by
process_zover_data.py) and the reference human-infecting virus taxid list, and derives
a binary vector-borne phenotype label at the virus (taxid) level.

Label logic:
    vector_borne = 1  — virus appears in ZOVER mosquito/tick data AND also
                         appears in the reference human-infecting dataset (crosses
                         into a vertebrate host)
    vector_borne = 0  — virus appears in ZOVER mosquito/tick data but NOT in
                         the reference human-infecting dataset (insect/arachnid-only)

Usage:
    conda activate viral-phenotype
    python scripts/derive_vector_borne_labels.py

Requires (already produced earlier in the pipeline):
    data/zover_accession_host_types.csv        — from process_zover_data.py
    outputs/accession_taxid_cache.csv           — accession -> taxid cache
                                                   (extended here for any new
                                                   ZOVER accessions not yet resolved)
    data/phenotype_labels_with_strict_zoonotic.csv — the reference human-infecting taxid set

Output:
    data/vector_borne_labels.csv
        virus_taxid, host_types, n_accessions, vector_borne
"""

import time
import pandas as pd
from pathlib import Path
from Bio import Entrez

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

HOST_TYPES_PATH = DATA_DIR / "zover_accession_host_types.csv"
CACHE_PATH = OUTPUTS_DIR / "accession_taxid_cache.csv"
HUMAN_LABELS_PATH = DATA_DIR / "phenotype_labels_with_strict_zoonotic.csv"
OUT_PATH = DATA_DIR / "vector_borne_labels.csv"

Entrez.email = "your_email@liverpool.ac.uk"   # <- replace with your email

# ── 1. LOAD HOST-TYPE MAPPING ─────────────────────────────────────────────────
print("Loading ZOVER accession -> host-type mapping...")
host_df = pd.read_csv(HOST_TYPES_PATH)
print(f"  {len(host_df)} unique ZOVER accessions (mosquito/tick)")

# ── 2. RESOLVE ACCESSION -> TAXID (reuse + extend the existing cache) ────────
def load_cache():
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
        return dict(zip(df["accession"], df["taxid"]))
    return {}

def save_cache(cache):
    pd.DataFrame(list(cache.items()), columns=["accession", "taxid"]).to_csv(CACHE_PATH, index=False)

def fetch_taxids_batch(accessions, cache, batch_size=200):
    """Fetch taxids via NCBI esummary (nuccore) — returns TaxId directly."""
    to_fetch = [a for a in accessions if a not in cache]
    print(f"  {len(cache)} accessions already cached, fetching {len(to_fetch)} from NCBI...")
    total_batches = (len(to_fetch) - 1) // batch_size + 1 if to_fetch else 0
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        print(f"    Batch {i // batch_size + 1}/{total_batches} ({len(batch)} accessions)...", end=" ", flush=True)
        resolved = 0
        try:
            handle = Entrez.esearch(db="nuccore", term=" OR ".join(batch), retmax=len(batch))
            search_record = Entrez.read(handle)
            handle.close()
            uids = search_record["IdList"]
            if uids:
                handle2 = Entrez.esummary(db="nuccore", id=",".join(uids))
                summaries = Entrez.read(handle2)
                handle2.close()
                for s in summaries:
                    acc_ver = s.get("AccessionVersion", "")
                    taxid = s.get("TaxId", None)
                    if acc_ver and taxid:
                        cache[acc_ver] = int(taxid)
                        resolved += 1
            print(f"done ({resolved}/{len(batch)} resolved)")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.34)  # NCBI rate limit: 3 req/sec without API key
        save_cache(cache)
    return cache

print("\nResolving accession -> taxid (NCBI lookup with local cache)...")
cache = load_cache()
# accession_taxid_cache.csv is keyed by *versioned* accessions (as embedded);
# zover_accession_host_types.csv is keyed by *stripped* accessions. Build a
# stripped -> taxid lookup from the cache first, then only hit NCBI for the
# ZOVER accessions that are still unresolved.
stripped_to_taxid = {}
for acc_ver, tid in cache.items():
    stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)

host_df["virus_taxid"] = host_df["accession"].map(stripped_to_taxid)
still_missing = host_df.loc[host_df["virus_taxid"].isna(), "accession"].tolist()
print(f"  Resolved from existing cache: {len(host_df) - len(still_missing)}/{len(host_df)}")

if still_missing:
    cache = fetch_taxids_batch(still_missing, cache)
    save_cache(cache)
    for acc_ver, tid in cache.items():
        stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)
    host_df["virus_taxid"] = host_df["accession"].map(stripped_to_taxid)

n_missing = host_df["virus_taxid"].isna().sum()
print(f"  Final resolved: {len(host_df) - n_missing}/{len(host_df)}")
if n_missing > 0:
    print(f"  Dropping {n_missing} accessions with no resolvable taxid")
    host_df = host_df.dropna(subset=["virus_taxid"])
host_df["virus_taxid"] = host_df["virus_taxid"].astype(int)

# ── 3. AGGREGATE HOST TYPES TO TAXID LEVEL ────────────────────────────────────
print("\nAggregating host types per virus taxid...")
def union_host_types(series):
    types = set()
    for s in series:
        types.update(s.split(","))
    return ",".join(sorted(types))

agg = host_df.groupby("virus_taxid").agg(
    host_types=("host_types", union_host_types),
    n_accessions=("accession", "count"),
).reset_index()
print(f"  Unique virus taxids in ZOVER mosquito/tick data: {len(agg)}")

# ── 4. CROSS-REFERENCE WITH MAYA'S HUMAN-INFECTING DATASET ──────────────────
print("\nCross-referencing with the reference human-infecting dataset...")
human_taxids = set(pd.read_csv(HUMAN_LABELS_PATH)["virus_taxid"])
print(f"  reference human-infecting taxids: {len(human_taxids)}")

agg["vector_borne"] = agg["virus_taxid"].isin(human_taxids).astype(int)

# ── 5. SAVE + SUMMARY ─────────────────────────────────────────────────────────
agg.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total taxa: {len(agg)}")
print(f"  vector_borne=1 (crosses into human dataset): {agg['vector_borne'].sum()}")
print(f"  vector_borne=0 (insect/arachnid-only):        {(agg['vector_borne'] == 0).sum()}")
for host_type in ["mosquito", "tick"]:
    mask = agg["host_types"].str.contains(host_type)
    print(f"\n  {host_type}-associated taxa: {mask.sum()}")
    print(f"    vector_borne=1: {(mask & (agg['vector_borne'] == 1)).sum()}")
    print(f"    vector_borne=0: {(mask & (agg['vector_borne'] == 0)).sum()}")
print("\nDone! Next: run pipeline/tier1_vector_borne.py")
