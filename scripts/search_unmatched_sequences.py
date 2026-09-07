import time
import pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "your_email@example.com"

BASE_DIR  = Path(__file__).parent.parent
OUT_DIR   = BASE_DIR / "data" / "raw_matched"
CACHE_CSV = BASE_DIR / "outputs" / "accession_taxid_cache.csv"

unmatched = pd.read_excel(BASE_DIR / "data" / "unmatched_viruses.xlsx")
cache_df  = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))
existing_taxids = set(acc_to_taxid.values())
to_search = unmatched[~unmatched["virus_taxid"].isin(existing_taxids)]
print(f"Viruses to search: {len(to_search)}")

new_cache = {}
downloaded = 0
not_found  = 0

for i, (_, row) in enumerate(to_search.iterrows()):
    taxid = row["virus_taxid"]
    name  = row["virus"]
    if (i+1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(to_search)} | downloaded: {downloaded} | not found: {not_found}")
    try:
        handle = Entrez.esearch(db="nuccore", term=f"txid{taxid}[Organism]", retmax=3)
        rec = Entrez.read(handle); handle.close()
        uids = rec["IdList"]
        if not uids:
            handle = Entrez.esearch(db="nuccore", term=f"{name}[Title] AND complete[Title]", retmax=3)
            rec = Entrez.read(handle); handle.close()
            uids = rec["IdList"]
        if not uids:
            not_found += 1
            time.sleep(0.34)
            continue
        handle = Entrez.efetch(db="nuccore", id=",".join(uids), rettype="fasta", retmode="text")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        for record in records:
            out_path = OUT_DIR / f"{record.id}.fasta"
            if not out_path.exists():
                SeqIO.write(record, out_path, "fasta")
                new_cache[record.id] = taxid
                downloaded += 1
    except Exception as e:
        print(f"  Error for {name}: {e}")
    time.sleep(0.34)

if new_cache:
    new_rows = pd.DataFrame(list(new_cache.items()), columns=["accession","taxid"])
    updated  = pd.concat([cache_df, new_rows], ignore_index=True)
    updated.to_csv(CACHE_CSV, index=False)

print(f"Downloaded: {downloaded} | Not found: {not_found}")
print(f"New cache entries: {len(new_cache)}")
