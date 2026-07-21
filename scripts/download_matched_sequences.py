"""
Download matched virus sequences from NCBI
==========================================
Fetches FASTA sequences for the 352 viruses that matched Maya's labels.
Saves one FASTA file per accession in data/raw_matched/

Usage:
    conda activate viral-phenotype
    python download_matched_sequences.py
"""

import time
import pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "your_email@liverpool.ac.uk"   # ← replace with your uni email

BASE_DIR    = Path(__file__).parent
CACHE_CSV   = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
METADATA_CSV = BASE_DIR / "outputs" / "refseq_metadata.csv"
LABELS_XLSX = BASE_DIR / "data" / "human_pathogens.xlsx"
OUT_DIR     = BASE_DIR / "data" / "raw_matched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Find the accessions we actually need ─────────────────────────────────────
print("Finding matched accessions...")
cache_df = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))

meta = pd.read_csv(METADATA_CSV)
meta["virus_taxid"] = meta["accession"].map(acc_to_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)

human_df        = pd.read_excel(LABELS_XLSX, sheet_name="human")
label_taxids    = set(human_df["virus_taxid"])
matched_taxids  = set(meta["virus_taxid"]) & label_taxids
matched_accs    = meta[meta["virus_taxid"].isin(matched_taxids)]["accession"].tolist()

# Skip already downloaded
to_download = [a for a in matched_accs
               if not (OUT_DIR / f"{a}.fasta").exists()]

print(f"  Total matched accessions: {len(matched_accs)}")
print(f"  Already downloaded:       {len(matched_accs) - len(to_download)}")
print(f"  To download:              {len(to_download)}")

# ── Download in batches ───────────────────────────────────────────────────────
BATCH_SIZE = 50
total_batches = (len(to_download) - 1) // BATCH_SIZE + 1 if to_download else 0

for i in range(0, len(to_download), BATCH_SIZE):
    batch = to_download[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} sequences)...", end=" ", flush=True)
    try:
        handle = Entrez.efetch(
            db="nuccore",
            id=",".join(batch),
            rettype="fasta",
            retmode="text"
        )
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()

        saved = 0
        for record in records:
            # Match back to our accession (record.id may include version)
            acc = record.id
            # Try exact match first, then strip version
            if acc not in [a for a in batch]:
                acc = acc.rsplit(".", 1)[0] + "." + acc.rsplit(".", 1)[-1]
            out_path = OUT_DIR / f"{record.id}.fasta"
            SeqIO.write(record, out_path, "fasta")
            saved += 1

        print(f"saved {saved}/{len(batch)}")
    except Exception as e:
        print(f"error: {e}")
    time.sleep(0.4)

# ── Summary ───────────────────────────────────────────────────────────────────
downloaded = list(OUT_DIR.glob("*.fasta"))
print(f"\nDone! {len(downloaded)} FASTA files in data/raw_matched/")
print("Next step: run tier2_partial_unfreeze.py")
