"""
Sequence De-duplication via CD-HIT-EST (Ticket 1)
====================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Randomly splitting into CV folds without deduplication risks leakage:
near-identical strains of the same or closely-related taxa can land in
both train and test folds, inflating cross-validated performance. This
clusters one representative genome per taxon at 90% nucleotide identity
and keeps only the cluster representatives — any taxon whose genome is
>=90% identical to another taxon's genome gets collapsed to one entry
before any train/test split happens.

Run separately per dataset (H2H/Zoonotic and Vector-Borne use different
CV splits for different tasks, so leakage is only a within-dataset
concern — no need to cross-check between them).

Usage:
    conda activate viral-phenotype
    python scripts/dedup_sequences.py --dataset h2h_zoo
    python scripts/dedup_sequences.py --dataset vector_borne

Outputs (written to outputs/dedup/):
    {dataset}_representative.fasta   — one longest-sequence-per-taxon FASTA
    {dataset}.clstr                  — raw CD-HIT cluster output
    {dataset}_dedup_taxids.csv       — virus_taxid, cluster_id, is_representative, cluster_size
    {dataset}_deduplicated_taxids.txt — just the representative taxids, one per line
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_DIR = OUTPUTS_DIR / "dedup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FASTA_DIR = BASE_DIR / "data" / "raw_matched"
ZOVER_DIR = BASE_DIR / "data" / "zover_new_sequences"
CULICOIDES_DIR = BASE_DIR / "data" / "culicoides_new_sequences"
CACHE_CSV = OUTPUTS_DIR / "accession_taxid_cache.csv"

IDENTITY_THRESHOLD = 0.90
WORD_SIZE = 8  # CD-HIT-EST recommended word size for -c 0.90-1.00

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["h2h_zoo", "vector_borne"], required=True)
args = parser.parse_args()


# ── 1. LOAD THE RELEVANT LABEL/TAXID SET ─────────────────────────────────────
if args.dataset == "h2h_zoo":
    human_df = pd.read_csv(BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv")
    label_taxids = set(human_df["virus_taxid"])
    fasta_dirs = [FASTA_DIR]
else:
    vb_df = pd.read_csv(BASE_DIR / "data/vector_borne_labels.csv")
    label_taxids = set(vb_df["virus_taxid"])
    fasta_dirs = [FASTA_DIR, ZOVER_DIR, CULICOIDES_DIR]

print(f"Dataset: {args.dataset}  ({len(label_taxids)} labelled taxa)")

# ── 2. RESOLVE ACCESSION -> TAXID ─────────────────────────────────────────────
cache_df = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))
stripped_to_taxid = {}
for acc_ver, tid in acc_to_taxid.items():
    stripped_to_taxid.setdefault(acc_ver.split(".")[0], tid)

def resolve_taxid(acc: str):
    return acc_to_taxid.get(acc) or stripped_to_taxid.get(acc.split(".")[0])


# ── 3. FIND THE LONGEST SEQUENCE PER TAXID (representative genome) ──────────
print("Scanning FASTA files for the longest sequence per virus taxid...")
taxid_best_seq = {}
seen_accessions = set()
n_scanned = 0
for fasta_dir in fasta_dirs:
    if not fasta_dir.exists():
        continue
    for fasta_path in sorted(fasta_dir.glob("*.fasta")):
        if fasta_path.stem in seen_accessions:
            continue
        seen_accessions.add(fasta_path.stem)
        tid = resolve_taxid(fasta_path.stem)
        if not tid or tid not in label_taxids:
            continue
        n_scanned += 1
        try:
            record = next(SeqIO.parse(fasta_path, "fasta"))
        except StopIteration:
            continue
        seq = str(record.seq).upper()
        if tid not in taxid_best_seq or len(seq) > taxid_best_seq[tid][0]:
            taxid_best_seq[tid] = (len(seq), seq)
        if n_scanned % 2000 == 0:
            print(f"  Scanned {n_scanned} matching FASTA files...")

print(f"  Found representative sequences for {len(taxid_best_seq)}/{len(label_taxids)} labelled taxa")

# ── 4. WRITE COMBINED FASTA FOR CD-HIT ────────────────────────────────────────
fasta_out = OUT_DIR / f"{args.dataset}_representative.fasta"
with open(fasta_out, "w") as f:
    for tid, (length, seq) in taxid_best_seq.items():
        f.write(f">{tid}\n{seq}\n")
print(f"  Saved: {fasta_out}")

# ── 5. RUN CD-HIT-EST ──────────────────────────────────────────────────────────
clstr_out = OUT_DIR / f"{args.dataset}"
print(f"\nRunning cd-hit-est at {IDENTITY_THRESHOLD:.0%} identity...")
cmd = [
    "cd-hit-est",
    "-i", str(fasta_out),
    "-o", str(clstr_out),
    "-c", str(IDENTITY_THRESHOLD),
    "-n", str(WORD_SIZE),
    "-M", "8000",   # memory limit MB
    "-T", "8",      # threads
    "-d", "0",      # full sequence name in output (no truncation)
]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-3000:])
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    raise SystemExit(f"cd-hit-est failed with exit code {result.returncode}")

# ── 6. PARSE .clstr OUTPUT ────────────────────────────────────────────────────
clstr_file = OUT_DIR / f"{args.dataset}.clstr"
print(f"\nParsing {clstr_file}...")
rows = []
current_cluster = None
cluster_members = []

def flush_cluster(cluster_id, members):
    for tid, is_rep in members:
        rows.append({
            "virus_taxid": tid,
            "cluster_id": cluster_id,
            "is_representative": is_rep,
            "cluster_size": len(members),
        })

with open(clstr_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">Cluster"):
            if current_cluster is not None:
                flush_cluster(current_cluster, cluster_members)
            current_cluster = int(line.split()[1])
            cluster_members = []
        else:
            # e.g. "0   12345nt, >1234567... *" or "... at 91.23%"
            is_rep = line.endswith("*")
            tid = int(line.split(">")[1].split("...")[0])
            cluster_members.append((tid, is_rep))
    if current_cluster is not None:
        flush_cluster(current_cluster, cluster_members)

dedup_df = pd.DataFrame(rows)
dedup_csv = OUT_DIR / f"{args.dataset}_dedup_taxids.csv"
dedup_df.to_csv(dedup_csv, index=False)
print(f"  Saved: {dedup_csv}")

n_clusters = dedup_df["cluster_id"].nunique()
n_total = len(dedup_df)
n_dropped = n_total - n_clusters
multi_member_clusters = dedup_df.groupby("cluster_id").size()
multi_member_clusters = multi_member_clusters[multi_member_clusters > 1]

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total taxa clustered:      {n_total}")
print(f"  Unique clusters (kept):    {n_clusters}")
print(f"  Taxa dropped as near-dup:  {n_dropped}  ({n_dropped/n_total:.1%})")
print(f"  Clusters with >1 member:   {len(multi_member_clusters)}")
if len(multi_member_clusters):
    print(f"  Largest cluster size:      {multi_member_clusters.max()}")

representative_taxids = dedup_df[dedup_df["is_representative"]]["virus_taxid"].tolist()
txt_out = OUT_DIR / f"{args.dataset}_deduplicated_taxids.txt"
with open(txt_out, "w") as f:
    for tid in representative_taxids:
        f.write(f"{tid}\n")
print(f"\nSaved: {txt_out}  ({len(representative_taxids)} representative taxa)")
print("Done! ✓")
