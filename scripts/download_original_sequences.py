"""
Re-download the original 11,017 RefSeq sequences
Uses accession_taxid_cache.csv to get all accessions
and downloads any not already in raw_matched/
"""

import time
import pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "k.pazmany@liverpool.ac.uk"

BASE_DIR  = Path(__file__).parent.parent
CACHE_CSV = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
OUT_DIR   = BASE_DIR / "data" / "raw_matched"
OUT_DIR.mkdir(exist_ok=True)

cache_df = pd.read_csv(CACHE_CSV)
all_accs = cache_df["accession"].tolist()
existing = set(f.stem for f in OUT_DIR.glob("*.fasta"))

to_download = [a for a in all_accs if a not in existing]
print(f"Total accessions in cache: {len(all_accs)}")
print(f"Already downloaded: {len(existing)}")
print(f"To download: {len(to_download)}")

BATCH_SIZE = 50
downloaded = 0
errors = 0

for i in range(0, len(to_download), BATCH_SIZE):
    batch = to_download[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(to_download) - 1) // BATCH_SIZE + 1
    print(f"Batch {batch_num}/{total_batches}...", end=" ", flush=True)
    try:
        handle = Entrez.efetch(db="nuccore", id=",".join(batch),
                               rettype="fasta", retmode="text")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        for record in records:
            SeqIO.write(record, OUT_DIR / f"{record.id}.fasta", "fasta")
            downloaded += 1
        print(f"saved {len(records)}/{len(batch)}")
    except Exception as e:
        print(f"error: {e}")
        errors += 1
    time.sleep(0.4)

print(f"\nDone! Downloaded: {downloaded}, Errors: {errors}")
print(f"Total FASTA files now: {len(list(OUT_DIR.glob('*.fasta')))}")
