"""
Create strict zoonotic labels by removing viruses flagged by Maya
as mislabelled (lab animals, research use, not genuine zoonoses)

Usage:
    python scripts/create_strict_zoonotic.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Viruses Maya flagged as NOT genuinely zoonotic
NOT_ZOONOTIC = {
    "chicken anemia virus",
    "gyrovirus 4",
    "gyrovirus gyv3",
    "torque teno midi virus",
    "torque teno virus",
    "alphapolyomavirus quintihominis",
    "betapolyomavirus hominis",
    "betapolyomavirus macacae",
    "epsilonpolyomavirus bovis",
    "african green monkey polyomavirus",
    "avastrovirus 3",
    "avian nephritis virus 1",
    "avian nephritis virus 2",
    "mamastrovirus 1",
    "mamastrovirus 3",
    "mamastrovirus 4",
    "human astrovirus",
    "human papillomavirus 11",
    "human papillomavirus 16",
    "human papillomavirus 18",
    "human papillomavirus type 6",
    "betapapillomavirus 1",
}

# Load existing labels
human_df        = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="human")
interactions_df = pd.read_excel(BASE_DIR / "data/human_pathogens.xlsx", sheet_name="interactions")

# Relaxed zoonotic (existing)
human_hosts = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
non_human   = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
human_df["zoonotic_relaxed"] = human_df["virus_taxid"].isin(human_hosts & non_human).astype(int)

# Strict zoonotic — remove Maya's flagged viruses
human_df["zoonotic_strict"] = human_df.apply(
    lambda row: 0 if row["virus"].lower() in NOT_ZOONOTIC
    else row["zoonotic_relaxed"], axis=1
)

# Stats
print(f"Total viruses: {len(human_df)}")
print(f"\nRelaxed zoonotic: {human_df['zoonotic_relaxed'].sum()} positive")
print(f"Strict zoonotic:  {human_df['zoonotic_strict'].sum()} positive")
print(f"Difference (removed): {human_df['zoonotic_relaxed'].sum() - human_df['zoonotic_strict'].sum()}")

# Show which ones were changed
changed = human_df[human_df["zoonotic_relaxed"] != human_df["zoonotic_strict"]]
print(f"\nViruses relabelled from zoonotic to non-zoonotic:")
for _, row in changed.iterrows():
    print(f"  - {row['virus']} ({row['virus_family']})")

# Save
human_df[["virus_taxid", "virus", "virus_family", "virus_genus",
          "human_to_human", "zoonotic_relaxed", "zoonotic_strict"]].to_csv(
    BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv", index=False)
print(f"\nSaved: data/phenotype_labels_with_strict_zoonotic.csv")
