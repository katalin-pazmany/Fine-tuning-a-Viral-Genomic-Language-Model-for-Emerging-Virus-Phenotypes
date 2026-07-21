# Embed the already-fetched, host-verified Culicoides FASTA files
# (data/culicoides_new_sequences/) into the shared embedding set.
# Same pattern as scripts/embed_new_sequences.py.
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FASTA_DIR = BASE_DIR / "data" / "culicoides_new_sequences"
MODEL_DIR = os.path.expanduser("~/viral-phenotype-finetuning/models/vir2vec")
MAX_LENGTH = 512
DEVICE = torch.device("cpu")

print("Finding sequences to embed...")
existing_npz = np.load(OUTPUTS_DIR / "refseq_embeddings.npz")
existing_accs = set(existing_npz["accessions"].tolist())
new_fastas = [p for p in sorted(FASTA_DIR.glob("*.fasta")) if p.stem not in existing_accs]
print(f"  Existing embeddings: {len(existing_accs)}")
print(f"  New sequences to embed: {len(new_fastas)}")
if not new_fastas:
    print("Nothing new to embed — exiting.")
    exit()

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

print("\nEmbedding new sequences...")
new_embeddings, new_accessions, new_descriptions, new_seq_lengths = [], [], [], []
for i, fasta_path in enumerate(new_fastas):
    acc = fasta_path.stem
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
        print(f"  Embedded {i+1}/{len(new_fastas)}...")

print(f"\n  Embedded: {len(new_embeddings)} sequences")

merged_embs = np.vstack([existing_npz["embeddings"], np.array(new_embeddings, dtype=np.float32)])
merged_accs = np.concatenate([existing_npz["accessions"], np.array(new_accessions)])
np.savez_compressed(OUTPUTS_DIR / "refseq_embeddings.npz", embeddings=merged_embs, accessions=merged_accs)

existing_meta = pd.read_csv(OUTPUTS_DIR / "refseq_metadata.csv")
new_meta = pd.DataFrame({"accession": new_accessions, "description": new_descriptions,
                          "seq_length": new_seq_lengths})
pd.concat([existing_meta, new_meta], ignore_index=True).to_csv(OUTPUTS_DIR / "refseq_metadata.csv", index=False)

print(f"\nTotal embeddings now: {len(merged_embs)}")
print("Done! ✓")
