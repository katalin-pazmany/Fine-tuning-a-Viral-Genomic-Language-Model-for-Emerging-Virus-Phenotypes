"""
Download GenBank sequences for VIRION viruses missing from RefSeq
=================================================================
Searches all of GenBank (not just curated RefSeq) for the 648
VIRION viruses that had no RefSeq sequences available.

Usage:
    conda activate viral-phenotype
    python download_genbank_sequences.py
"""

import time
import pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "k.pazmany@liverpool.ac.uk"

BASE_DIR    = Path(__file__).parent
MISSING_CSV = BASE_DIR / "data" / "virion_still_missing.csv"
OUT_DIR     = BASE_DIR / "data" / "raw_matched"
CACHE_CSV   = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load missing taxids
missing_df     = pd.read_csv(MISSING_CSV)
missing_taxids = missing_df["virus_taxid"].tolist()
print(f"Taxids to search in GenBank: {len(missing_taxids)}")

# Load existing cache
cache_df     = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))
existing_accs = set(f.stem for f in OUT_DIR.glob("*.fasta"))

new_cache_entries = {}
downloaded = 0
not_found  = 0
skipped    = 0

MAX_SEQS_PER_TAXID = 3

for i, taxid in enumerate(missing_taxids):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(missing_taxids)} | downloaded: {downloaded} | not found: {not_found}")

    try:
        # Search GenBank — no refseq filter, but prefer complete genomes
        search_handle = Entrez.esearch(
            db="nuccore",
            term=f"txid{taxid}[Organism] AND complete[title]",
            retmax=MAX_SEQS_PER_TAXID
        )
        record = Entrez.read(search_handle)
        search_handle.close()
        uids = record["IdList"]

        # Fall back to any sequence if no complete genome
        if not uids:
            search_handle = Entrez.esearch(
                db="nuccore",
                term=f"txid{taxid}[Organism]",
                retmax=MAX_SEQS_PER_TAXID
            )
            record = Entrez.read(search_handle)
            search_handle.close()
            uids = record["IdList"]

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
            if record.id in existing_accs:
                skipped += 1
                continue
            out_path = OUT_DIR / f"{record.id}.fasta"
            SeqIO.write(record, out_path, "fasta")
            new_cache_entries[record.id] = taxid
            existing_accs.add(record.id)
            downloaded += 1

    except Exception as e:
        print(f"  Error for taxid {taxid}: {e}")

    time.sleep(0.34)

# Update cache
if new_cache_entries:
    new_rows = pd.DataFrame(
        list(new_cache_entries.items()),
        columns=["accession", "taxid"]
    )
    updated = pd.concat([cache_df, new_rows], ignore_index=True)
    updated.to_csv(CACHE_CSV, index=False)
    print(f"\nUpdated cache with {len(new_cache_entries)} new entries")

print(f"\n{'='*50}")
print(f"Done!")
print(f"  Downloaded:  {downloaded} sequences")
print(f"  Skipped:     {skipped} (already had)")
print(f"  Not found:   {not_found} taxids had no sequences")
print(f"  Total FASTA files now: {len(list(OUT_DIR.glob('*.fasta')))}")
