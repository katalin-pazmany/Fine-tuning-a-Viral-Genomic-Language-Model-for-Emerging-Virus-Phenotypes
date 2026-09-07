"""
Download sequences for missing labelled viruses
================================================
Fetches RefSeq genome sequences for the 1,027 labelled viruses
that don't yet have sequences in our dataset.

Strategy:
    1. For each missing taxid, search NCBI nuccore for RefSeq sequences
    2. Download up to 3 sequences per taxid (representative sequences)
    3. Save to data/raw_matched/ alongside existing sequences

Usage:
    conda activate viral-phenotype
    python download_missing_sequences.py
"""

import time
import pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "your_email@example.com"   

BASE_DIR  = Path(__file__).parent
MISSING_CSV = BASE_DIR / "data" / "missing_taxids.csv"
OUT_DIR     = BASE_DIR / "data" / "raw_matched"
CACHE_CSV   = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load missing taxids
missing_df  = pd.read_csv(MISSING_CSV)
missing_taxids = missing_df["virus_taxid"].tolist()
print(f"Missing taxids to fetch: {len(missing_taxids)}")

# Load existing cache so we can update it
cache_df     = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))

MAX_SEQS_PER_TAXID = 3   # keep dataset manageable
new_cache_entries  = {}
downloaded = 0
skipped    = 0
not_found  = 0

for i, taxid in enumerate(missing_taxids):
    # Check if already downloaded
    existing = list(OUT_DIR.glob(f"*.fasta"))
    existing_taxids_downloaded = set()
    for f in existing:
        acc = f.stem
        tid = acc_to_taxid.get(acc)
        if tid:
            existing_taxids_downloaded.add(tid)

    if taxid in existing_taxids_downloaded:
        skipped += 1
        continue

    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(missing_taxids)} | downloaded: {downloaded} | not found: {not_found}")

    try:
        # Search for RefSeq sequences for this taxid
        search_handle = Entrez.esearch(
            db="nuccore",
            term=f"txid{taxid}[Organism] AND refseq[filter] AND complete[title]",
            retmax=MAX_SEQS_PER_TAXID
        )
        search_record = Entrez.read(search_handle)
        search_handle.close()
        uids = search_record["IdList"]

        # If no complete genomes, try without 'complete' filter
        if not uids:
            search_handle = Entrez.esearch(
                db="nuccore",
                term=f"txid{taxid}[Organism] AND refseq[filter]",
                retmax=MAX_SEQS_PER_TAXID
            )
            search_record = Entrez.read(search_handle)
            search_handle.close()
            uids = search_record["IdList"]

        if not uids:
            not_found += 1
            time.sleep(0.34)
            continue

        # Fetch sequences
        fetch_handle = Entrez.efetch(
            db="nuccore",
            id=",".join(uids),
            rettype="fasta",
            retmode="text"
        )
        records = list(SeqIO.parse(fetch_handle, "fasta"))
        fetch_handle.close()

        for record in records:
            out_path = OUT_DIR / f"{record.id}.fasta"
            SeqIO.write(record, out_path, "fasta")
            # Add to cache
            new_cache_entries[record.id] = taxid
            downloaded += 1

    except Exception as e:
        print(f"  Error for taxid {taxid}: {e}")

    time.sleep(0.34)

# Update cache file
if new_cache_entries:
    new_rows = pd.DataFrame(
        list(new_cache_entries.items()),
        columns=["accession", "taxid"]
    )
    updated_cache = pd.concat([cache_df, new_rows], ignore_index=True)
    updated_cache.to_csv(CACHE_CSV, index=False)
    print(f"\nUpdated cache with {len(new_cache_entries)} new entries")

print(f"\n{'='*50}")
print(f"Done!")
print(f"  Downloaded:  {downloaded} sequences")
print(f"  Skipped:     {skipped} (already had)")
print(f"  Not found:   {not_found} taxids had no RefSeq sequences")
total_fastas = len(list(OUT_DIR.glob("*.fasta")))
print(f"  Total FASTA files now: {total_fastas}")
