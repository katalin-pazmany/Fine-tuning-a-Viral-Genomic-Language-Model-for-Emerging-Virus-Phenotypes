"""Add taxids for the 1,545 GenBank sequences not in the cache."""
import time, pandas as pd
from pathlib import Path
from Bio import Entrez
import xml.etree.ElementTree as ET

Entrez.email = "your_email@example.com"

cache = pd.read_csv("outputs/accession_taxid_cache.csv")
cache_accs = set(cache["accession"])
downloaded = set(f.stem for f in Path("data/raw_matched").glob("*.fasta"))
extras = sorted(downloaded - cache_accs)
print(f"Extra accessions to resolve: {len(extras)}")

new_entries = {}
BATCH = 200
for i in range(0, len(extras), BATCH):
    batch = extras[i:i+BATCH]
    print(f"Batch {i//BATCH+1}/{(len(extras)-1)//BATCH+1}...", end=" ", flush=True)
    try:
        handle = Entrez.esummary(db="nuccore", id=",".join(batch))
        summaries = Entrez.read(handle)
        handle.close()
        resolved = 0
        for s in summaries:
            acc = s.get("AccessionVersion","")
            taxid = s.get("TaxId", None)
            if acc and taxid:
                new_entries[acc] = int(taxid)
                resolved += 1
        print(f"resolved {resolved}/{len(batch)}")
    except Exception as e:
        print(f"error: {e}")
    time.sleep(0.34)

if new_entries:
    new_rows = pd.DataFrame(list(new_entries.items()), columns=["accession","taxid"])
    updated = pd.concat([cache, new_rows], ignore_index=True)
    updated.to_csv("outputs/accession_taxid_cache.csv", index=False)
    print(f"\nAdded {len(new_entries)} entries to cache")
