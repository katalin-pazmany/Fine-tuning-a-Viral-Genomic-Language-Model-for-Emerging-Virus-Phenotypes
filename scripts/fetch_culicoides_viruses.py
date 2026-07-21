"""
Fetch Biting-Midge (Culicoides/Ceratopogonidae) Associated Viruses
======================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Maya's clarification: vector-borne scope is mosquitoes, ticks, AND biting
midges (Culicoides / family Ceratopogonidae) — the three main vertebrate
vector categories (sandflies excluded for now). ZOVER only covers mosquito
and tick (confirmed directly on its download page — no Culicoides category
exists there, even in the current version), so this sources directly from
NCBI nuccore instead.

A free-text search for "Culicoides" pulls in false positives from mixed
surveillance studies (e.g. Culex/Armigeres mosquito records that happen to
co-occur in the same submission batch). To avoid that, this fetches the
full GenBank record for every candidate and verifies the actual annotated
/host= qualifier contains Culicoides or a Ceratopogonidae genus, rather
than trusting the free-text match.

Usage:
    conda activate viral-phenotype
    python scripts/fetch_culicoides_viruses.py

Output:
    data/culicoides_new_sequences/   — individual FASTA files, verified host
    data/culicoides_host_verified.csv — accession, taxid, host, virus name
"""

import time
import re
from pathlib import Path
from io import StringIO

import pandas as pd
from Bio import Entrez, SeqIO

Entrez.email = "your_email@liverpool.ac.uk"   # <- replace with your email

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_SEQ_DIR = DATA_DIR / "culicoides_new_sequences"
OUT_SEQ_DIR.mkdir(exist_ok=True)
OUT_CSV = DATA_DIR / "culicoides_host_verified.csv"

# Genera within Ceratopogonidae with documented arbovirus association;
# Culicoides is by far the dominant genus (matches Maya's "biting midges").
HOST_KEYWORDS = ["culicoides", "ceratopogonidae"]

SEARCH_TERM = '(Culicoides[All Fields] OR Ceratopogonidae[All Fields]) AND viruses[filter]'

# ── 1. SEARCH NCBI FOR CANDIDATES ─────────────────────────────────────────────
print(f"Searching nuccore: {SEARCH_TERM}")
handle = Entrez.esearch(db="nuccore", term=SEARCH_TERM, retmax=5000)
search_record = Entrez.read(handle)
handle.close()
candidate_ids = search_record["IdList"]
print(f"  {len(candidate_ids)} candidate records found")

# ── 2. FETCH + VERIFY HOST ANNOTATION (batched) ──────────────────────────────
print("\nFetching GenBank records and verifying /host= annotation...")
verified_rows = []
batch_size = 200
for i in range(0, len(candidate_ids), batch_size):
    batch = candidate_ids[i:i + batch_size]
    print(f"  Batch {i // batch_size + 1}/{(len(candidate_ids) - 1) // batch_size + 1} "
          f"({len(batch)} records)...", end=" ", flush=True)
    try:
        handle = Entrez.efetch(db="nuccore", id=",".join(batch), rettype="gb", retmode="text")
        records = list(SeqIO.parse(handle, "genbank"))
        handle.close()
    except Exception as e:
        print(f"error: {e}")
        time.sleep(1)
        continue

    n_verified = 0
    for record in records:
        host = None
        for feature in record.features:
            if feature.type == "source" and "host" in feature.qualifiers:
                host = feature.qualifiers["host"][0]
                break
        if host is None:
            continue
        host_lower = host.lower()
        if not any(kw in host_lower for kw in HOST_KEYWORDS):
            continue

        # Extract taxid from db_xref
        taxid = None
        for feature in record.features:
            if feature.type == "source":
                for xref in feature.qualifiers.get("db_xref", []):
                    if xref.startswith("taxon:"):
                        taxid = int(xref.split(":")[1])
                break

        seq = str(record.seq)
        acc = record.id  # versioned accession, e.g. "PQ536757.1"
        out_path = OUT_SEQ_DIR / f"{acc}.fasta"
        with open(out_path, "w") as f:
            f.write(f">{acc} {record.description}\n{seq}\n")

        verified_rows.append({
            "accession": acc,
            "virus_taxid": taxid,
            "host": host,
            "virus_name": record.annotations.get("organism", ""),
            "seq_length": len(seq),
        })
        n_verified += 1

    print(f"done ({n_verified}/{len(batch)} host-verified)")
    time.sleep(0.34)  # NCBI rate limit: 3 req/sec without API key

verified_df = pd.DataFrame(verified_rows)
verified_df.to_csv(OUT_CSV, index=False)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Candidate records:        {len(candidate_ids)}")
print(f"Host-verified (Culicoides/Ceratopogonidae): {len(verified_df)}")
print(f"Unique virus taxa (by taxid): {verified_df['virus_taxid'].nunique()}")
print(f"\nSaved: {OUT_CSV}")
print(f"Saved: {len(verified_df)} FASTA files to {OUT_SEQ_DIR}")
print("\nNext: resolve/extend accession->taxid cache, then embed new sequences "
      "(same pattern as scripts/embed_new_sequences.py), then extend "
      "scripts/derive_vector_borne_labels.py to include this third host type.")
