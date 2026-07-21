"""
IntAct Validation (Ticket 9, adapted from HPIDB which is down)
==================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

HPIDB is unreachable (connection refused on all known URLs). Using IntAct
(EBI) instead — live, actively maintained, virus-host PPI curated, queried
via its PSICQUIC REST endpoint.

Hypothesis: viruses the model confidently predicts as high zoonotic risk
should have a higher documented rate of physical human-protein interactions
than low-risk viruses, as an orthogonal check independent of the training
labels themselves.

Uses the existing per-virus Zoonotic (strict) predictions from
outputs/misclassification/Zoonotic_strict_per_virus.csv — top 10% vs
bottom 10% by predicted probability, IntAct documented-interaction count
compared between the two groups (Mann-Whitney U + proportion with >=1
documented interaction).

Usage:
    conda activate viral-phenotype
    python scripts/intact_validation.py

Output:
    outputs/intact_validation_results.json
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PER_VIRUS_CSV = OUTPUTS_DIR / "misclassification" / "Zoonotic_strict_per_virus.csv"
OUT_PATH = OUTPUTS_DIR / "intact_validation_results.json"

PSICQUIC_URL = "https://www.ebi.ac.uk/Tools/webservices/psicquic/intact/webservices/current/search/query/"
HUMAN_TAXID = 9606


def intact_count(virus_taxid: int) -> int:
    """Count of documented human<->virus interactions in IntAct for this taxon."""
    query = f"taxidA:{HUMAN_TAXID} AND taxidB:{virus_taxid}"
    url = PSICQUIC_URL + urllib.parse.quote(query) + "?format=count"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return int(resp.read().decode().strip())
    except Exception:
        return -1  # query failed, distinguish from a genuine 0


# ── 1. LOAD PER-VIRUS PREDICTIONS, PICK TOP/BOTTOM 10% ───────────────────────
print(f"Loading predictions from {PER_VIRUS_CSV}...")
df = pd.read_csv(PER_VIRUS_CSV)
df = df.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
n = len(df)
decile = max(int(n * 0.10), 5)
top10 = df.iloc[:decile]
bottom10 = df.iloc[-decile:]
print(f"  {n} viruses total, top/bottom decile = {decile} viruses each")
print(f"  Top decile prob range:    {top10['predicted_prob'].min():.3f} - {top10['predicted_prob'].max():.3f}")
print(f"  Bottom decile prob range: {bottom10['predicted_prob'].min():.3f} - {bottom10['predicted_prob'].max():.3f}")

# ── 2. QUERY INTACT FOR EACH VIRUS ────────────────────────────────────────────
def query_group(taxids, label):
    counts = []
    for i, tid in enumerate(taxids):
        c = intact_count(int(tid))
        counts.append(c)
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(taxids)} queried...")
        time.sleep(0.34)
    return np.array(counts)

print("\nQuerying IntAct for top-decile viruses...")
top_counts = query_group(top10["virus_taxid"].tolist(), "top")
print("\nQuerying IntAct for bottom-decile viruses...")
bottom_counts = query_group(bottom10["virus_taxid"].tolist(), "bottom")

# ── 3. COMPARE ─────────────────────────────────────────────────────────────────
top_valid = top_counts[top_counts >= 0]
bottom_valid = bottom_counts[bottom_counts >= 0]
top_has_any = (top_valid > 0).mean()
bottom_has_any = (bottom_valid > 0).mean()

print(f"\n{'='*60}\nRESULTS\n{'='*60}")
print(f"Top decile:    {len(top_valid)} valid queries, {top_has_any:.1%} have >=1 documented interaction, "
      f"mean count={top_valid.mean():.2f}")
print(f"Bottom decile: {len(bottom_valid)} valid queries, {bottom_has_any:.1%} have >=1 documented interaction, "
      f"mean count={bottom_valid.mean():.2f}")

result = {
    "n_top": int(len(top_valid)),
    "n_bottom": int(len(bottom_valid)),
    "top_pct_with_interaction": round(float(top_has_any), 4),
    "bottom_pct_with_interaction": round(float(bottom_has_any), 4),
    "top_mean_count": round(float(top_valid.mean()), 4) if len(top_valid) else None,
    "bottom_mean_count": round(float(bottom_valid.mean()), 4) if len(bottom_valid) else None,
}

if len(top_valid) >= 5 and len(bottom_valid) >= 5:
    stat, pval = mannwhitneyu(top_valid, bottom_valid, alternative="greater")
    result["mannwhitney_u"] = round(float(stat), 4)
    result["p_value"] = round(float(pval), 6)
    print(f"\nMann-Whitney U (top > bottom): U={stat:.2f}, p={pval:.6f}")

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
print("Done! ✓")
