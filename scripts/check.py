python - << 'EOF'
import pandas as pd
import numpy as np

# Maya's matched viruses (547)
cache_df = pd.read_csv("outputs/accession_taxid_cache.csv")
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))
meta = pd.read_csv("outputs/refseq_metadata.csv")
meta["virus_taxid"] = meta["accession"].map(acc_to_taxid)
meta = meta.dropna(subset=["virus_taxid"])
meta["virus_taxid"] = meta["virus_taxid"].astype(int)
human_df = pd.read_excel("data/human_pathogens.xlsx", sheet_name="human")
maya_matched = set(meta["virus_taxid"]) & set(human_df["virus_taxid"])

# VIRION labels
virion_labels = pd.read_csv("data/virion_phenotype_labels.csv")
virion_with_h2h = virion_labels[virion_labels["human_to_human"].notna()]
virion_taxids = set(virion_with_h2h["virus_taxid"].astype(int))

already_in_547 = virion_taxids & maya_matched
new_from_virion = virion_taxids - maya_matched

print(f"Maya's matched viruses: {len(maya_matched)}")
print(f"VIRION viruses with h2h label: {len(virion_taxids)}")
print(f"Already in our 547: {len(already_in_547)}")
print(f"Genuinely new from VIRION: {len(new_from_virion)}")
print(f"Potential total if merged: {len(maya_matched | virion_taxids)}")
EOF