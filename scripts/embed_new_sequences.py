"""
Embed new sequences and merge with existing embeddings
======================================================
Runs Vir2vec on the 449 newly downloaded sequences and merges
the resulting embeddings with the existing refseq_embeddings.npz

Usage:
    conda activate viral-phenotype
    python embed_new_sequences.py
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR    = Path(__file__).parent
FASTA_DIR   = BASE_DIR / "data" / "raw_matched"
CACHE_CSV   = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
EXISTING_NPZ = BASE_DIR / "outputs" / "refseq_embeddings.npz"
EXISTING_CSV = BASE_DIR / "outputs" / "refseq_metadata.csv"
OUT_NPZ     = BASE_DIR / "outputs" / "refseq_embeddings.npz"   # overwrite
OUT_CSV     = BASE_DIR / "outputs" / "refseq_metadata.csv"     # overwrite

MODEL_DIR   = os.path.expanduser("~/viral-phenotype-finetuning/models/vir2vec")
MAX_LENGTH  = 512
DEVICE      = torch.device("cpu")   # MPS causes errors with Vir2vec

# ── 1. Find new sequences (not in existing embeddings) ───────────────────────
print("Finding new sequences to embed...")
existing_npz = np.load(EXISTING_NPZ)
existing_accs = set(existing_npz["accessions"].tolist())

new_fastas = []
for fasta_path in sorted(FASTA_DIR.glob("*.fasta")):
    acc = fasta_path.stem
    if acc not in existing_accs:
        new_fastas.append(fasta_path)

print(f"  Existing embeddings: {len(existing_accs)}")
print(f"  New sequences to embed: {len(new_fastas)}")

if not new_fastas:
    print("Nothing new to embed — exiting.")
    exit()

# ── 2. Load model ────────────────────────────────────────────────────────────
print(f"\nLoading Vir2vec from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.float32)
model.to(DEVICE)
model.eval()
print(f"  Model loaded — {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

# ── 3. Embed new sequences ───────────────────────────────────────────────────
def embed_sequence(seq: str) -> np.ndarray:
    """Embed a single sequence, chunking if necessary."""
    seq = seq.upper().replace("U", "T")
    chunks = [seq[i:i+MAX_LENGTH] for i in range(0, len(seq), MAX_LENGTH)]
    chunk_embs = []
    for chunk in chunks:
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH, padding=False)
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(DEVICE),
                output_hidden_states=True
            )
        hidden = out.hidden_states[-1]   # last layer: (1, seq_len, 768)
        pooled = hidden.max(dim=1).values.squeeze(0).cpu().numpy()   # (768,)
        chunk_embs.append(pooled)
    return np.mean(chunk_embs, axis=0)   # mean pool across chunks

new_embeddings = []
new_accessions = []
new_descriptions = []
new_seq_lengths = []
skipped = 0

for i, fasta_path in enumerate(new_fastas):
    acc = fasta_path.stem
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            skipped += 1
            continue
        record = records[0]
        seq = str(record.seq)

        # Skip sequences with too many ambiguous bases
        n_count = seq.upper().count("N")
        if n_count / max(len(seq), 1) > 0.1:
            skipped += 1
            continue

        emb = embed_sequence(seq)
        new_embeddings.append(emb)
        new_accessions.append(acc)
        new_descriptions.append(record.description[:100])
        new_seq_lengths.append(len(seq))

        if (i + 1) % 50 == 0:
            print(f"  Embedded {i+1}/{len(new_fastas)} sequences...")

    except Exception as e:
        print(f"  Error embedding {acc}: {e}")
        skipped += 1

print(f"\n  Embedded: {len(new_embeddings)} sequences")
print(f"  Skipped:  {skipped}")

# ── 4. Merge with existing embeddings ────────────────────────────────────────
print("\nMerging with existing embeddings...")
existing_embs = existing_npz["embeddings"]
existing_accs_arr = existing_npz["accessions"]

new_embs_arr = np.array(new_embeddings, dtype=np.float32)
merged_embs  = np.vstack([existing_embs, new_embs_arr])
merged_accs  = np.concatenate([existing_accs_arr,
                                np.array(new_accessions)])

print(f"  Existing: {existing_embs.shape}")
print(f"  New:      {new_embs_arr.shape}")
print(f"  Merged:   {merged_embs.shape}")

# ── 5. Save merged embeddings ─────────────────────────────────────────────────
np.savez_compressed(OUT_NPZ, embeddings=merged_embs, accessions=merged_accs)
print(f"  Saved: {OUT_NPZ}")

# ── 6. Update metadata CSV ────────────────────────────────────────────────────
existing_meta = pd.read_csv(EXISTING_CSV)
new_meta = pd.DataFrame({
    "accession":   new_accessions,
    "description": new_descriptions,
    "seq_length":  new_seq_lengths,
})
merged_meta = pd.concat([existing_meta, new_meta], ignore_index=True)
merged_meta.to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_CSV}")
print(f"\nDone! Total embeddings: {len(merged_embs)}")
