# Derive vector-borne labels including biting midges (Culicoides) as a
# third host-type category, alongside the existing mosquito/tick data.
# Maya's clarification: vector-borne scope = mosquito + tick + biting midge.
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EXISTING_LABELS = DATA_DIR / "vector_borne_labels.csv"
CULICOIDES_CSV = DATA_DIR / "culicoides_host_verified.csv"
HUMAN_LABELS_PATH = DATA_DIR / "phenotype_labels_with_strict_zoonotic.csv"
OUT_PATH = DATA_DIR / "vector_borne_labels.csv"

print("Loading existing mosquito/tick vector-borne labels...")
existing = pd.read_csv(EXISTING_LABELS)
print(f"  {len(existing)} taxa (mosquito/tick)")

print("\nLoading Culicoides (biting midge) verified accessions...")
cdf = pd.read_csv(CULICOIDES_CSV).dropna(subset=["virus_taxid"])
cdf["virus_taxid"] = cdf["virus_taxid"].astype(int)
culicoides_agg = cdf.groupby("virus_taxid").agg(n_accessions_bm=("accession", "count")).reset_index()
culicoides_agg["host_types_bm"] = "biting_midge"
print(f"  {len(culicoides_agg)} unique biting-midge-associated taxa")

# ── Merge: union host_types, sum accession counts, outer-join on taxid ───────
print("\nMerging mosquito/tick + biting-midge datasets...")
merged = existing.set_index("virus_taxid")[["host_types", "n_accessions"]].to_dict("index")
for _, row in culicoides_agg.iterrows():
    tid = row["virus_taxid"]
    if tid in merged:
        existing_types = set(merged[tid]["host_types"].split(","))
        existing_types.add("biting_midge")
        merged[tid]["host_types"] = ",".join(sorted(existing_types))
        merged[tid]["n_accessions"] += row["n_accessions_bm"]
    else:
        merged[tid] = {"host_types": "biting_midge", "n_accessions": row["n_accessions_bm"]}

merged_df = pd.DataFrame([
    {"virus_taxid": tid, "host_types": v["host_types"], "n_accessions": v["n_accessions"]}
    for tid, v in merged.items()
])
print(f"  Combined: {len(merged_df)} unique taxa (was {len(existing)} mosquito/tick-only)")

# ── Cross-reference with the reference human-infecting dataset ──────────────
print("\nCross-referencing with the reference human-infecting dataset...")
human_taxids = set(pd.read_csv(HUMAN_LABELS_PATH)["virus_taxid"])
merged_df["vector_borne"] = merged_df["virus_taxid"].isin(human_taxids).astype(int)

merged_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total taxa: {len(merged_df)}")
print(f"  vector_borne=1: {merged_df['vector_borne'].sum()}")
print(f"  vector_borne=0: {(merged_df['vector_borne'] == 0).sum()}")
for host_type in ["mosquito", "tick", "biting_midge"]:
    mask = merged_df["host_types"].str.contains(host_type)
    print(f"\n  {host_type}-associated taxa: {mask.sum()}")
    print(f"    vector_borne=1: {(mask & (merged_df['vector_borne'] == 1)).sum()}")
    print(f"    vector_borne=0: {(mask & (merged_df['vector_borne'] == 0)).sum()}")
print("\nDone! Next: rerun scripts/dedup_sequences.py --dataset vector_borne, "
      "then Tier 1/2/3 and baselines.")
