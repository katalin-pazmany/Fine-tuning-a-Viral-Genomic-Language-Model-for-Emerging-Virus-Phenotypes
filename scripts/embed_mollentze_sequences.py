# Fetch + embed the Mollentze et al. accessions not already in our embedding set.
# Same pattern as scripts/embed_new_sequences.py, driven by
# data/mollentze_scope_check.csv (already_embedded == False).
import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from Bio import Entrez, SeqIO
from transformers import AutoTokenizer, AutoModelForCausalLM

Entrez.email = "your_email@liverpool.ac.uk"

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
FASTA_DIR = DATA_DIR / "mollentze_new_sequences"
FASTA_DIR.mkdir(exist_ok=True)

MODEL_DIR = os.path.expanduser("~/viral-phenotype-finetuning/models/vir2vec")
MAX_LENGTH = 512
DEVICE = torch.device("cpu")

# ── 1. LOAD SCOPE CHECK, FIND WHAT NEEDS FETCHING ────────────────────────────
print("Loading scope check...")
mdf = pd.read_csv(DATA_DIR / "mollentze_scope_check.csv")
to_fetch = mdf.loc[~mdf["already_embedded"], "accession"].dropna().unique().tolist()
print(f"  {len(to_fetch)} accessions need fetching + embedding")

# ── 2. FETCH FASTA SEQUENCES FROM NCBI ────────────────────────────────────────
print("\nFetching sequences from NCBI (batched)...")
batch_size = 100
fetched = []
for i in range(0, len(to_fetch), batch_size):
    batch = to_fetch[i:i + batch_size]
    print(f"  Batch {i // batch_size + 1}/{(len(to_fetch) - 1) // batch_size + 1} "
          f"({len(batch)} accessions)...", end=" ", flush=True)
    try:
        handle = Entrez.efetch(db="nuccore", id=",".join(batch), rettype="fasta", retmode="text")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        for record in records:
            out_path = FASTA_DIR / f"{record.id}.fasta"
            SeqIO.write(record, out_path, "fasta")
            fetched.append(record.id)
        print(f"done ({len(records)} fetched)")
    except Exception as e:
        print(f"error: {e}")
    time.sleep(0.34)

print(f"\n  Total fetched: {len(fetched)}/{len(to_fetch)}")

# ── 3. LOAD MODEL ─────────────────────────────────────────────────────────────
print(f"\nLoading Vir2vec from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.float32)
model.to(DEVICE)
model.eval()

def embed_sequence(seq: str) -> np.ndarray:
    seq = seq.upper().replace("U", "T")
    chunks = [seq[i:i + MAX_LENGTH] for i in range(0, len(seq), MAX_LENGTH)]
    chunk_embs = []
    for chunk in chunks:
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                         max_length=MAX_LENGTH, padding=False)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(DEVICE), output_hidden_states=True)
        hidden = out.hidden_states[-1]
        pooled = hidden.max(dim=1).values.squeeze(0).cpu().numpy()
        chunk_embs.append(pooled)
    return np.mean(chunk_embs, axis=0)

# ── 4. EMBED ───────────────────────────────────────────────────────────────────
print("\nEmbedding fetched sequences...")
new_embeddings, new_accessions, new_descriptions, new_seq_lengths = [], [], [], []
for i, acc in enumerate(fetched):
    fasta_path = FASTA_DIR / f"{acc}.fasta"
    try:
        record = next(SeqIO.parse(fasta_path, "fasta"))
        seq = str(record.seq)
        if seq.upper().count("N") / max(len(seq), 1) > 0.1:
            continue
        emb = embed_sequence(seq)
        new_embeddings.append(emb)
        new_accessions.append(acc)
        new_descriptions.append(record.description[:100])
        new_seq_lengths.append(len(seq))
    except Exception as e:
        print(f"  Error embedding {acc}: {e}")
    if (i + 1) % 50 == 0:
        print(f"  Embedded {i+1}/{len(fetched)}...")

print(f"\n  Embedded: {len(new_embeddings)} sequences")

# ── 5. MERGE INTO EXISTING EMBEDDINGS ────────────────────────────────────────
existing_npz = np.load(OUTPUTS_DIR / "refseq_embeddings.npz")
merged_embs = np.vstack([existing_npz["embeddings"], np.array(new_embeddings, dtype=np.float32)])
merged_accs = np.concatenate([existing_npz["accessions"], np.array(new_accessions)])
np.savez_compressed(OUTPUTS_DIR / "refseq_embeddings.npz", embeddings=merged_embs, accessions=merged_accs)

existing_meta = pd.read_csv(OUTPUTS_DIR / "refseq_metadata.csv")
new_meta = pd.DataFrame({"accession": new_accessions, "description": new_descriptions,
                          "seq_length": new_seq_lengths})
pd.concat([existing_meta, new_meta], ignore_index=True).to_csv(OUTPUTS_DIR / "refseq_metadata.csv", index=False)

print(f"\nTotal embeddings now: {len(merged_embs)}")
print("Done! ✓")
