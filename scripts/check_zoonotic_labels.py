"""
Zoonotic label sanity check using Claude
=========================================
Checks all non-zoonotic viruses in Maya's dataset against Claude's
knowledge to flag any obvious misclassifications.

Output: data/zoonotic_llm_check.csv

Usage:
    conda activate viral-phenotype
    python check_zoonotic_labels.py
"""

import json
import os
import sys
import time
import pandas as pd
import anthropic
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
LABELS_XLSX = BASE_DIR / "data" / "human_pathogens.xlsx"
OUT_CSV     = BASE_DIR / "data" / "zoonotic_llm_check.csv"
MODEL       = "claude-sonnet-5"

# ── Fail loudly if the API is not configured ─────────────────────────────────
# The previous run silently swallowed 933 auth failures into zoonotic_llm=False,
# producing a fake "0 flagged" result. Refuse to run without a key so that can't
# happen again.
if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
    sys.exit("ERROR: ANTHROPIC_API_KEY is not set. Refusing to run — a keyless run "
             "would record auth errors as non-zoonotic and look like a clean result. "
             "Set the key in your environment and re-run.")

# ── Load non-zoonotic viruses ─────────────────────────────────────────────────
human_df        = pd.read_excel(LABELS_XLSX, sheet_name="human")
interactions_df = pd.read_excel(LABELS_XLSX, sheet_name="interactions")

human_host_taxids = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
non_human_taxids  = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
human_df["zoonotic"] = human_df["virus_taxid"].isin(human_host_taxids & non_human_taxids).astype(int)

non_zoonotic = human_df[human_df["zoonotic"] == 0][["virus", "virus_taxid", "virus_family", "virus_genus"]].drop_duplicates(subset=["virus_taxid"])
print(f"Non-zoonotic viruses to check: {len(non_zoonotic)}")

# ── Load existing results if any ──────────────────────────────────────────────
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    checked_taxids = set(existing["virus_taxid"].tolist())
    print(f"Already checked: {len(checked_taxids)}, resuming...")
    results = existing.to_dict("records")
else:
    checked_taxids = set()
    results = []

# ── Claude API ────────────────────────────────────────────────────────────────
client = anthropic.Anthropic()

SYSTEM = """You are an expert virologist. Answer ONLY in JSON with these keys:
- zoonotic: boolean (true if virus naturally infects non-human animals AND can cross into humans from animals)
- confidence: "high", "medium", or "low"
- reasoning: one sentence explanation

A virus is zoonotic if it has a known animal reservoir and can naturally spill over into humans.
Viruses that are exclusively human pathogens with no animal reservoir are NOT zoonotic.
Do not include markdown, just raw JSON."""

def check_virus(name, family, genus):
    msg = f'Is "{name}" (family: {family}, genus: {genus}) a zoonotic virus? It is currently labelled as NON-zoonotic in our database. Is this label correct?'
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM,
        messages=[{"role": "user", "content": msg}]
    )
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)
    # zoonotic_llm stays None if the model omits the key, so a parse gap is
    # never silently counted as "non-zoonotic".
    return {
        "zoonotic_llm": parsed.get("zoonotic"),
        "confidence": parsed.get("confidence", "low"),
        "reasoning": parsed.get("reasoning", "")
    }

# ── Run checks ────────────────────────────────────────────────────────────────
to_check = non_zoonotic[~non_zoonotic["virus_taxid"].isin(checked_taxids)]
print(f"Viruses remaining to check: {len(to_check)}")

n_errors = 0
for i, (_, row) in enumerate(to_check.iterrows()):
    try:
        result = check_virus(row["virus"], row["virus_family"], row["virus_genus"])
        status = "error" if result["zoonotic_llm"] is None else "ok"
    except Exception as e:
        result = {"zoonotic_llm": None, "confidence": "error", "reasoning": f"Error: {e}"}
        status = "error"
    if status == "error":
        n_errors += 1
    record = {
        "virus": row["virus"],
        "virus_taxid": row["virus_taxid"],
        "virus_family": row["virus_family"],
        "virus_genus": row["virus_genus"],
        "status": status,
        **result
    }
    results.append(record)

    if result["zoonotic_llm"] and result["confidence"] != "low":
        flag = "⚠️  FLAGGED"
    elif status == "error":
        flag = "✗ ERROR"
    else:
        flag = "✓"
    print(f"  [{i+1}/{len(to_check)}] {row['virus'][:50]:<50} {flag}")

    # Save every 10
    if (i + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    time.sleep(0.3)

# ── Save & summarise ──────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)

flagged = df[(df["zoonotic_llm"] == True) & (df["confidence"] != "low")]
n_err_total = int((df["status"] == "error").sum()) if "status" in df else 0
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total rows:          {len(df)}")
print(f"Successful checks:   {len(df) - n_err_total}")
print(f"Errors (excluded):   {n_err_total}")
if n_err_total and n_err_total >= 0.5 * len(df):
    print("  ⚠️  Majority of calls failed — treat this run as INVALID and re-run.")
print(f"Flagged as possibly zoonotic (high/medium confidence): {len(flagged)}")
for _, row in flagged.iterrows():
    print(f"  - {row['virus']} ({row['virus_family']}) [{row['confidence']}]")
    print(f"    {row['reasoning']}")
print(f"\nSaved: {OUT_CSV}")
