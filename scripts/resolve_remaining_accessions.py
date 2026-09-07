"""Resolve remaining 345 accessions using efetch."""
import time, pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

Entrez.email = "your_email@example.com"

cache = pd.read_csv("outputs/accession_taxid_cache.csv")
cache_accs = set(cache["accession"])
downloaded = set(f.stem for f in Path("data/raw_matched").glob("*.fasta"))
to_resolve = sorted(downloaded - cache_accs)
print(f"To resolve: {len(to_resolve)}")

new_entries = {}
BATCH = 50
for i in range(0, len(to_resolve), BATCH):
    batch = to_resolve[i:i+BATCH]
    print(f"Batch {i//BATCH+1}/{(len(to_resolve)-1)//BATCH+1}...", end=" ", flush=True)
    try:
        handle = Entrez.efetch(db="nuccore", id=",".join(batch),
                               rettype="gb", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        resolved = 0
        for rec in records:
            acc = rec.get("GBSeq_accession-version", "")
            taxid = None
            for feat in rec.get("GBSeq_feature-table", []):
                for qual in feat.get("GBFeature_quals", []):
                    if qual.get("GBQualifier_name") == "db_xref":
                        val = qual.get("GBQualifier_value", "")
                        if val.startswith("taxon:"):
                            taxid = int(val.replace("taxon:", ""))
            if acc and taxid:
                new_entries[acc] = taxid
                resolved += 1
        print(f"resolved {resolved}/{len(batch)}")
    except Exception as e:
        print(f"error: {e}")
    time.sleep(0.4)

if new_entries:
    new_rows = pd.DataFrame(list(new_entries.items()), columns=["accession","taxid"])
    updated = pd.concat([cache, new_rows], ignore_index=True)
    updated.to_csv("outputs/accession_taxid_cache.csv", index=False)
    print(f"\nAdded {len(new_entries)} entries to cache")
    print(f"Still unresolved: {len(to_resolve) - len(new_entries)}")
