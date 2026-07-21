"""
Process VIRION data for expanded phenotype labels
==================================================
Derives zoonotic and human-infecting labels from VIRION edgelist,
then merges with Maya's human_to_human labels.

Output: data/virion_phenotype_labels.csv
    virus_taxid | human_to_human | zoonotic

Usage:
    conda activate viral-phenotype
    python process_virion.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
VIRION_PATH = DATA_DIR / "virion_edgelist.csv"
LABELS_XLSX = DATA_DIR / "human_pathogens.xlsx"
OUT_CSV     = DATA_DIR / "virion_phenotype_labels.csv"

HUMAN_TAXID = 9606

# ── 1. LOAD VIRION ────────────────────────────────────────────────────────────
print("Loading VIRION edgelist...")
virion = pd.read_csv(VIRION_PATH)
print(f"  Total interactions: {len(virion)}")

# ── 2. DERIVE ZOONOTIC LABEL ──────────────────────────────────────────────────
# Zoonotic = virus infects BOTH humans AND non-human hosts
human_virus_taxids     = set(virion[virion["HostTaxID"] == HUMAN_TAXID]["VirusTaxID"])
non_human_virus_taxids = set(virion[virion["HostTaxID"] != HUMAN_TAXID]["VirusTaxID"])
zoonotic_taxids        = human_virus_taxids & non_human_virus_taxids

print(f"  Viruses infecting humans: {len(human_virus_taxids)}")
print(f"  Of which zoonotic (also infect non-humans): {len(zoonotic_taxids)}")

# Build base dataframe of all human-infecting viruses
virion_labels = pd.DataFrame({
    "virus_taxid": sorted(human_virus_taxids),
})
virion_labels["zoonotic"] = virion_labels["virus_taxid"].isin(zoonotic_taxids).astype(int)

# ── 3. MERGE WITH MAYA'S HUMAN_TO_HUMAN LABELS ───────────────────────────────
print("\nLoading Maya's human_to_human labels...")
human_df = pd.read_excel(LABELS_XLSX, sheet_name="human")
print(f"  Viruses with h2h labels: {len(human_df)}")

# Merge — keep all VIRION viruses, add h2h where available
merged = virion_labels.merge(
    human_df[["virus_taxid", "human_to_human"]],
    on="virus_taxid",
    how="left"
)

# Stats
in_both    = merged["human_to_human"].notna().sum()
virion_only = merged["human_to_human"].isna().sum()
print(f"\n  Viruses in both VIRION and Maya's labels: {in_both}")
print(f"  Viruses in VIRION only (no h2h label): {virion_only}")
print(f"  Total human-infecting viruses from VIRION: {len(merged)}")

print(f"\nZoonotic distribution (all VIRION human viruses):")
print(merged["zoonotic"].value_counts())

print(f"\nH2H distribution (where available):")
print(merged["human_to_human"].value_counts())

# ── 4. SAVE ───────────────────────────────────────────────────────────────────
merged.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# ── 5. COMPARE WITH PREVIOUS LABELS ──────────────────────────────────────────
print("\n=== COMPARISON ===")
print(f"Previous dataset: 547 matched viruses (Maya labels + RefSeq)")
print(f"VIRION human-infecting viruses: {len(merged)}")
print(f"VIRION viruses with h2h label: {in_both}")
print(f"VIRION viruses with zoonotic label: {len(merged)}")
print(f"\nPotential new viruses to add: {virion_only} (need h2h labels)")