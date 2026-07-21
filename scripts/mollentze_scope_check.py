# Mollentze et al. 2021 benchmark: scope check (Ticket 8)
# Downloads their 914-species accession list, resolves taxids, checks
# overlap with our existing embeddings to determine how much new
# sequence fetching/embedding is actually needed.
import time
import pandas as pd
from pathlib import Path
from Bio import Entrez
import numpy as np

Entrez.email = "your_email@liverpool.ac.uk"
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
CACHE_PATH = OUTPUTS_DIR / "accession_taxid_cache.csv"

MOLLENTZE_URL = "https://raw.githubusercontent.com/nardus/zoonotic_rank/main/InternalData/FinalData_Cleaned.csv"

print("Downloading Mollentze et al. dataset...")
mdf = pd.read_csv(MOLLENTZE_URL)
print(f"  {len(mdf)} rows loaded")
mdf = mdf.dropna(subset=["Accessions"])
mdf["accession"] = mdf["Accessions"].astype(str).str.split(";").str[0].str.strip()
print(f"  {len(mdf)} rows with a usable accession")
print(f"  InfectsHumans: {mdf['InfectsHumans'].sum()} True / {(~mdf['InfectsHumans'].astype(bool)).sum()} False")

# ── Resolve accession -> taxid using existing cache (exact + stripped) ──────
cache_df = pd.read_csv(CACHE_PATH)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))
stripped_to_taxid = {}
for acc_ver, tid in acc_to_taxid.items():
    stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)

def resolve(acc):
    return acc_to_taxid.get(acc) or stripped_to_taxid.get(acc.split(".")[0])

mdf["resolved_taxid"] = mdf["accession"].apply(resolve)
n_from_cache = mdf["resolved_taxid"].notna().sum()
print(f"\n  Resolved from existing cache: {n_from_cache}/{len(mdf)}")

# ── Resolve the rest via NCBI ──────────────────────────────────────────────
still_missing = mdf.loc[mdf["resolved_taxid"].isna(), "accession"].tolist()
if still_missing:
    print(f"  Resolving {len(still_missing)} remaining accessions via NCBI...")
    batch_size = 200
    for i in range(0, len(still_missing), batch_size):
        batch = still_missing[i:i + batch_size]
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
                        acc_to_taxid[acc_ver] = int(taxid)
                        stripped_to_taxid.setdefault(acc_ver.split(".")[0], int(taxid))
        except Exception as e:
            print(f"    batch error: {e}")
        time.sleep(0.34)
    mdf["resolved_taxid"] = mdf["accession"].apply(resolve)
    pd.DataFrame(list(acc_to_taxid.items()), columns=["accession", "taxid"]).to_csv(CACHE_PATH, index=False)

n_resolved = mdf["resolved_taxid"].notna().sum()
print(f"  Final resolved: {n_resolved}/{len(mdf)}")

# ── Overlap check against existing embeddings ────────────────────────────────
emb_data = np.load(OUTPUTS_DIR / "refseq_embeddings.npz")
embedded_accs = set(emb_data["accessions"].tolist())
embedded_stripped = set(a.split(".")[0] for a in embedded_accs)

mdf["already_embedded"] = mdf["accession"].apply(
    lambda a: a in embedded_accs or a.split(".")[0] in embedded_stripped)

n_have = mdf["already_embedded"].sum()
n_need = len(mdf) - n_have

out_csv = DATA_DIR / "mollentze_scope_check.csv"
mdf.to_csv(out_csv, index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total Mollentze species (usable accession): {len(mdf)}")
print(f"  Taxid resolved:        {n_resolved}")
print(f"  Already embedded:      {n_have}")
print(f"  Need new embedding:    {n_need}")
print(f"\nSaved: {out_csv}")
print("\nNext: if n_need is small, fetch + embed those directly. If large, "
      "treat as a separate embedding batch (same pattern as embed_new_sequences.py).")
