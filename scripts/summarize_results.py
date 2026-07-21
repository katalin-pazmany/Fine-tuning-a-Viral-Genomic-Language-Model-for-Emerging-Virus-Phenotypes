"""
Summarize All Results (full detail)
=====================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Reads every tier's results JSON across all phenotypes and label variants,
and prints one comparison table (plus saves it as CSV) — full detail version:
Tier 1 lists every classifier trained (including the random baseline), not
just the winner, so you can see e.g. how SVM compares even when it wasn't
the single best model for a task. Tier 2/3 only ever produce one model each.

Usage:
    conda activate viral-phenotype
    python scripts/summarize_results.py

Output:
    Printed table
    outputs/results_summary_full.csv
"""

import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

METRICS = ["AUROC", "AUPRC", "MCC", "F1_macro", "Brier"]
TIER1_CLASSIFIER_ORDER = [
    "SVM (RBF)", "Gradient Boosting", "Random Forest",
    "Logistic Regression", "Random (Baseline)",
]

# (phenotype, label_variant, tier, json_path, task_key, tier_label)
# NOTE: human_to_human is the same label column regardless of --strict (only
# the zoonotic label has a relaxed/strict distinction — the supervisor's 22 lab-artefact
# removals), so H2H only appears once here, not duplicated per label variant.
SOURCES = [
    ("Human-to-Human", "-", 1, OUTPUTS_DIR / "results_dedup" / "tier1_results.json", "human_to_human"),
    ("Human-to-Human", "-", 2, OUTPUTS_DIR / "tier2_partial_unfreeze_dedup" / "tier2_partial_unfreeze_results.json", "human_to_human"),
    ("Human-to-Human", "-", 3, OUTPUTS_DIR / "results_dedup" / "tier3_results.json", "human_to_human"),

    ("Zoonotic", "relaxed", 1, OUTPUTS_DIR / "results_dedup" / "tier1_results.json", "zoonotic"),
    ("Zoonotic", "relaxed", 2, OUTPUTS_DIR / "tier2_partial_unfreeze_dedup" / "tier2_partial_unfreeze_results.json", "zoonotic"),
    ("Zoonotic", "relaxed", 3, OUTPUTS_DIR / "results_dedup" / "tier3_results.json", "zoonotic"),
    ("Zoonotic", "strict",  1, OUTPUTS_DIR / "results_strict_dedup" / "tier1_results.json", "zoonotic"),
    ("Zoonotic", "strict",  2, OUTPUTS_DIR / "tier2_partial_unfreeze_strict_dedup" / "tier2_partial_unfreeze_results.json", "zoonotic"),
    ("Zoonotic", "strict",  3, OUTPUTS_DIR / "results_strict_dedup" / "tier3_results.json", "zoonotic"),

    ("Vector-Borne", "-", 1, OUTPUTS_DIR / "results_vector_borne_dedup" / "tier1_results.json", "vector_borne"),
    ("Vector-Borne", "-", 2, OUTPUTS_DIR / "tier2_partial_unfreeze_vector_borne_dedup" / "tier2_partial_unfreeze_results.json", "vector_borne"),
    ("Vector-Borne", "-", 3, OUTPUTS_DIR / "results_vector_borne_dedup" / "tier3_results.json", "vector_borne"),
]

TIER_LABELS = {
    1: "Tier 1 (frozen + shallow)",
    2: "Tier 2 (partial unfreeze)",
    3: "Tier 3 (LoRA)",
}


def load_task_results(json_path: Path, task_key: str):
    """Returns a list of (model_name, metrics_dict, is_best) tuples."""
    if not json_path.exists():
        return None, f"MISSING FILE: {json_path.relative_to(BASE_DIR)}"
    with open(json_path) as f:
        data = json.load(f)
    if task_key not in data:
        return None, f"MISSING TASK '{task_key}' in {json_path.relative_to(BASE_DIR)}"

    task_data = data[task_key]

    # Tier 1: dict of {classifier_name: metrics} — return every classifier,
    # ordered consistently, with the best-by-AUROC flagged.
    if all(isinstance(v, dict) for v in task_data.values()) and \
       any("AUROC" in v for v in task_data.values()):
        candidates = {k: v for k, v in task_data.items() if k != "Random (Baseline)"}
        best_name = max(candidates, key=lambda k: candidates[k]["AUROC"]) if candidates else None
        ordered_names = [n for n in TIER1_CLASSIFIER_ORDER if n in task_data] + \
                         [n for n in task_data if n not in TIER1_CLASSIFIER_ORDER]
        return [(name, task_data[name], name == best_name) for name in ordered_names], None

    # Tier 2/3: flat metrics dict for a single model.
    model_name = "Full unfreeze (layers 6-7)" if "tier2" in str(json_path).lower() else "LoRA (r=8)"
    return [(model_name, task_data, True)], None


rows = []
warnings = []
for phenotype, label_variant, tier, json_path, task_key in SOURCES:
    results, warning = load_task_results(json_path, task_key)
    if warning:
        warnings.append(warning)
        continue
    for model_name, metrics, is_best in results:
        row = {
            "Phenotype": phenotype,
            "Labels": label_variant,
            "Tier": TIER_LABELS[tier],
            "Model": model_name,
            "Best in Tier": "Yes" if is_best else "",
        }
        for m in METRICS:
            row[m] = metrics.get(m)
        rows.append(row)

df = pd.DataFrame(rows)

print("=" * 110)
print("FULL RESULTS COMPARISON — ALL PHENOTYPES, LABEL VARIANTS, TIERS, ALL CLASSIFIERS")
print("=" * 110)
with pd.option_context("display.width", 160, "display.max_columns", None):
    print(df.to_string(index=False))

if warnings:
    print("\n" + "=" * 110)
    print("WARNINGS (skipped rows)")
    print("=" * 110)
    for w in warnings:
        print(f"  - {w}")

out_path = OUTPUTS_DIR / "results_summary_final_dedup.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
