# One-off: append Culicoides accession->taxid mappings (already resolved
# during fetch_culicoides_viruses.py) into the shared accession_taxid_cache.csv,
# so dedup_sequences.py and the Tier 2/3 vector-borne pipelines (which only
# consult the shared cache, not culicoides_host_verified.csv) can find them.
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CACHE_PATH = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
CULICOIDES_CSV = BASE_DIR / "data" / "culicoides_host_verified.csv"

cache_df = pd.read_csv(CACHE_PATH)
existing = set(cache_df["accession"])
print(f"Existing cache: {len(cache_df)} entries")

cdf = pd.read_csv(CULICOIDES_CSV).dropna(subset=["virus_taxid"])
cdf["virus_taxid"] = cdf["virus_taxid"].astype(int)
new_rows = cdf.loc[~cdf["accession"].isin(existing), ["accession", "virus_taxid"]].rename(
    columns={"virus_taxid": "taxid"})
print(f"New Culicoides accessions to add: {len(new_rows)}")

merged = pd.concat([cache_df, new_rows], ignore_index=True)
merged.to_csv(CACHE_PATH, index=False)
print(f"Saved: {CACHE_PATH} ({len(merged)} total entries)")
