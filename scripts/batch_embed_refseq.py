"""
batch_embed_refseq.py
=====================
Generates Vir2vec embeddings for all sequences in viral_genomes_raw.fasta
and saves them to a .npz file ready to merge with Maya's labels.

Usage:
    conda activate viral-phenotype
    python scripts/batch_embed_refseq.py

Output:
    outputs/refseq_embeddings.npz  — embeddings matrix + accession IDs
    outputs/refseq_metadata.csv    — accession, description, seq_length
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.expanduser("~/viral-phenotype-finetuning/models/vir2vec")
FASTA_PATH  = os.path.expanduser("~/viral-phenotype-finetuning/data/all_sequences_full.fasta")
OUT_NPZ     = os.path.expanduser("~/viral-phenotype-finetuning/outputs/refseq_embeddings.npz")
OUT_CSV     = os.path.expanduser("~/viral-phenotype-finetuning/outputs/refseq_metadata.csv")
MAX_LENGTH  = 512    # Vir2vec's token context window
BATCH_SIZE  = 8      # sequences per batch — increase if you have RAM to spare
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cpu")
print("💻  Using CPU")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading tokeniser and model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    dtype=torch.float32,
)
model.to(device)
model.eval()
print(f"✅ Model loaded — {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters\n")

# ── Load FASTA ────────────────────────────────────────────────────────────────
print(f"Reading FASTA: {FASTA_PATH}")
records = list(SeqIO.parse(FASTA_PATH, "fasta"))
print(f"✅ {len(records)} sequences found\n")

# ── Embedding function ────────────────────────────────────────────────────────
def get_embedding(sequence: str) -> np.ndarray:
    """Tokenise sequence and max-pool last hidden state → (768,) vector."""
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]          # (1, seq_len, 768)
    embedding = last_hidden.max(dim=1).values[0]     # (768,)
    return embedding.cpu().numpy()

# ── Batch embedding loop ──────────────────────────────────────────────────────
accessions   = []
descriptions = []
seq_lengths  = []
embeddings   = []

print("Generating embeddings...")
t_start = time.time()

for record in tqdm(records, desc="Embedding", unit="seq"):
    accession   = record.id
    description = record.description
    sequence    = str(record.seq).upper()

    # Skip sequences with too many ambiguous bases
    n_count = sequence.count("N")
    if len(sequence) > 0 and n_count / len(sequence) > 0.1:
        print(f"  ⚠️  Skipping {accession} — too many Ns ({n_count/len(sequence):.1%})")
        continue

    emb = get_embedding(sequence)

    accessions.append(accession)
    descriptions.append(description)
    seq_lengths.append(len(sequence))
    embeddings.append(emb)

elapsed = time.time() - t_start
print(f"\n✅ Embedded {len(embeddings)} sequences in {elapsed/60:.1f} minutes")
print(f"   Skipped: {len(records) - len(embeddings)} sequences")

# ── Save embeddings ───────────────────────────────────────────────────────────
emb_matrix = np.vstack(embeddings)   # (n_seqs, 768)
print(f"\nEmbedding matrix shape: {emb_matrix.shape}")

np.savez(
    OUT_NPZ,
    embeddings=emb_matrix,
    accessions=np.array(accessions),
)
print(f"💾  Embeddings saved to: {OUT_NPZ}")

# ── Save metadata CSV ─────────────────────────────────────────────────────────
meta_df = pd.DataFrame({
    "accession":   accessions,
    "description": descriptions,
    "seq_length":  seq_lengths,
})
meta_df.to_csv(OUT_CSV, index=False)
print(f"💾  Metadata saved to:   {OUT_CSV}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
================================================
  Done!
  Sequences embedded : {len(embeddings)}
  Embedding shape    : {emb_matrix.shape}
  Mean embedding     : {emb_matrix.mean():.4f}
  Std embedding      : {emb_matrix.std():.4f}

  Next step:
  When Maya sends labels, run:
    merge_labels.py  (to join embeddings + labels into training CSV)
================================================
""")
