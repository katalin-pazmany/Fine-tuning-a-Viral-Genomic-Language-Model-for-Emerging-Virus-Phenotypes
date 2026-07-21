"""
Tier 2 Full Pipeline: Partial Layer Unfreezing of Vir2vec Backbone
==================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Loads raw FASTA sequences, tokenises them, runs through Vir2vec with
layers 6–7 + norm unfrozen. Mean-pools token hidden states to get a
per-sequence embedding, then mean-pools across segments for multi-segment
viruses. Classifies with a small linear head.

Freeze/unfreeze:
    Frozen:    embed_tokens, layers.0–5
    Trainable: layers.6, layers.7, norm, classifier head

Usage:
    conda activate viral-phenotype
    python tier2_partial_unfreeze.py --task both
    python tier2_partial_unfreeze.py --task h2h --epochs 20
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",       choices=["h2h", "zoo", "both"], default="both")
parser.add_argument("--epochs",     type=int,   default=20)
parser.add_argument("--batch_size", type=int,   default=2)
parser.add_argument("--lr",         type=float, default=2e-5)
parser.add_argument("--max_len",    type=int,   default=2048)
parser.add_argument("--patience",   type=int,   default=5)
parser.add_argument("--strict", action="store_true", help="Use strict zoonotic labels")
parser.add_argument('--dedup', action='store_true',
                     help='Filter to CD-HIT-deduplicated taxa to remove near-duplicate-genome '
                          'leakage across CV folds')
args = parser.parse_args()
LABEL_TYPE = 'strict' if args.strict else 'relaxed'
print(f'Zoonotic label type: {LABEL_TYPE}')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
_outputs_name = ("tier2_partial_unfreeze_strict" if args.strict else "tier2_partial_unfreeze") + \
                ("_dedup" if args.dedup else "")
OUTPUTS_DIR  = BASE_DIR / "outputs" / _outputs_name
FIGURES_DIR  = OUTPUTS_DIR / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_TAXIDS_PATH = BASE_DIR / "outputs" / "dedup" / "h2h_zoo_deduplicated_taxids.txt"

FASTA_DIR    = BASE_DIR / "data" / "raw_matched"
METADATA_CSV = BASE_DIR / "outputs" / "refseq_metadata.csv"
LABELS_XLSX  = BASE_DIR / "data" / "human_pathogens.xlsx"
CACHE_CSV    = BASE_DIR / "outputs" / "accession_taxid_cache.csv"
MODEL_NAME   = "pabloarozarenad/Vir2vec"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 1. LOAD LABELS ────────────────────────────────────────────────────────────
print("\nLoading labels...")
if args.strict:
    strict_df = pd.read_csv(BASE_DIR / "data/phenotype_labels_with_strict_zoonotic.csv")
    human_df  = strict_df.rename(columns={"zoonotic_strict": "zoonotic"})
else:
    human_df        = pd.read_excel(LABELS_XLSX, sheet_name="human")
    interactions_df = pd.read_excel(LABELS_XLSX, sheet_name="interactions")
    
    human_host_taxids = set(interactions_df[interactions_df["host"] == "Homo sapiens"]["virus_taxid"])
    non_human_taxids  = set(interactions_df[interactions_df["host"] != "Homo sapiens"]["virus_taxid"])
    human_df["zoonotic"] = human_df["virus_taxid"].isin(
        human_host_taxids & non_human_taxids).astype(int)

# ── 2. LOAD TAXID CACHE ───────────────────────────────────────────────────────
print("Loading accession→taxid cache...")
cache_df     = pd.read_csv(CACHE_CSV)
acc_to_taxid = dict(zip(cache_df["accession"], cache_df["taxid"].astype(int)))

# ── 3. MATCH FASTA FILES TO LABELS ───────────────────────────────────────────
print("Matching FASTA files to labels...")
label_lookup  = human_df.set_index("virus_taxid")[["human_to_human", "zoonotic"]]
label_taxids  = set(human_df["virus_taxid"])

# Group FASTA files by taxid
taxid_to_fastas = {}
for fasta_path in sorted(FASTA_DIR.glob("*.fasta")):
    acc = fasta_path.stem   # filename without .fasta
    tid = acc_to_taxid.get(acc)
    if tid and tid in label_taxids:
        taxid_to_fastas.setdefault(tid, []).append(fasta_path)

if args.dedup:
    if not DEDUP_TAXIDS_PATH.exists():
        raise SystemExit(f"--dedup requested but {DEDUP_TAXIDS_PATH} not found — "
                          f"run scripts/dedup_sequences.py --dataset h2h_zoo first.")
    with open(DEDUP_TAXIDS_PATH) as f:
        dedup_taxids = set(int(line.strip()) for line in f if line.strip())
    n_before = len(taxid_to_fastas)
    taxid_to_fastas = {t: v for t, v in taxid_to_fastas.items() if t in dedup_taxids}
    print(f"  --dedup: filtered {n_before} -> {len(taxid_to_fastas)} taxa "
          f"({n_before - len(taxid_to_fastas)} near-duplicate taxa removed)")

matched_taxids = sorted(taxid_to_fastas.keys())
print(f"  Matched viruses: {len(matched_taxids)}")

y_h2h = np.array([label_lookup.loc[t, "human_to_human"] for t in matched_taxids])
y_zoo  = np.array([label_lookup.loc[t, "zoonotic"]       for t in matched_taxids])
print(f"  h2h: {y_h2h.sum()} pos / {(y_h2h==0).sum()} neg")
print(f"  zoo: {y_zoo.sum()} pos / {(y_zoo==0).sum()} neg")

# ── 4. TOKENISER ─────────────────────────────────────────────────────────────
print(f"\nLoading tokeniser from {MODEL_NAME}...")
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenise_sequence(seq: str, max_len: int) -> dict:
    """Tokenise a DNA sequence, chunking if longer than max_len."""
    seq = seq.upper().replace("U", "T")   # RNA → DNA
    chunks = [seq[i:i+max_len] for i in range(0, len(seq), max_len)]
    encoded = []
    for chunk in chunks:
        enc = tokeniser(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False
        )
        encoded.append(enc)
    return encoded

# Pre-tokenise all sequences and cache as token tensors
# (so we don't re-tokenise every epoch)
print("Pre-tokenising sequences (this may take a minute)...")
taxid_to_tokens = {}
for i, tid in enumerate(matched_taxids):
    all_encodings = []
    for fasta_path in taxid_to_fastas[tid]:
        records = list(SeqIO.parse(fasta_path, "fasta"))
        for record in records:
            chunks = tokenise_sequence(str(record.seq), args.max_len)
            all_encodings.extend(chunks)
    taxid_to_tokens[tid] = all_encodings
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(matched_taxids)} viruses tokenised...")
print(f"  Done. {len(matched_taxids)} viruses tokenised.")

# ── 5. MODEL ──────────────────────────────────────────────────────────────────
class Vir2vecPartialFinetune(nn.Module):
    """
    Vir2vec backbone with layers 6–7 + norm unfrozen.
    Forward: tokenised input → mean-pool token hidden states → linear classifier.
    """
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        print(f"\nLoading {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name, device_map="cpu")

        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers + final norm
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(p) for p in ["layers.6", "layers.7", "norm"]):
                param.requires_grad = True

        total     = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  Backbone: {total:,} total params, {trainable:,} trainable ({100*trainable/total:.1f}%)")

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, 1)
        )

    def embed_chunks(self, encodings: list) -> torch.Tensor:
        """
        Run a list of tokenised chunks through backbone,
        mean-pool each chunk's tokens, then mean-pool across chunks.
        Returns: (768,) embedding for one virus.
        """
        chunk_embs = []
        for enc in encodings:
            input_ids      = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            with torch.set_grad_enabled(self.training):
                out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )
            # last_hidden_state: (1, seq_len, 768)
            hidden = out.last_hidden_state  # (1, seq_len, 768)
            # Mean pool over non-padding tokens
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, 768)
            chunk_embs.append(pooled.squeeze(0).float().to(DEVICE))   # (768,)

        # Mean pool across chunks/segments
        return torch.stack(chunk_embs).mean(dim=0).to(DEVICE)   # (768,)

    def forward_taxids(self, taxid_batch: list, token_dict: dict) -> torch.Tensor:
        """
        Process a batch of taxids, embed each, classify.
        Returns logits: (batch,)
        """
        embs = []
        for tid in taxid_batch:
            emb = self.embed_chunks(token_dict[tid])
            embs.append(emb)
        embs = torch.stack(embs).to(DEVICE)   # (batch, 768)
        return self.classifier(embs).squeeze(-1)   # (batch,)


# ── 6. TRAINING ───────────────────────────────────────────────────────────────
def train_epoch(model, taxid_batches, y_batches, optimizer, criterion):
    model.train()
    total_loss = 0
    for taxid_batch, y_batch in zip(taxid_batches, y_batches):
        optimizer.zero_grad()
        logits = model.forward_taxids(taxid_batch, taxid_to_tokens).to(DEVICE)
        y_t    = torch.tensor(y_batch, dtype=torch.float32).to(DEVICE)
        loss   = criterion(logits, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(taxid_batches)


@torch.no_grad()
def predict_all(model, taxids):
    model.eval()
    probs = []
    # Process one at a time to avoid OOM on CPU
    for tid in taxids:
        logit = model.forward_taxids([tid], taxid_to_tokens)
        prob  = torch.sigmoid(logit).item()
        probs.append(prob)
    return np.array(probs)


def make_batches(taxids, y, batch_size):
    """Split taxids and labels into batches."""
    batches_x, batches_y = [], []
    for i in range(0, len(taxids), batch_size):
        batches_x.append(taxids[i:i+batch_size])
        batches_y.append(y[i:i+batch_size])
    return batches_x, batches_y


def youden_threshold(y, probs):
    """Optimal decision threshold via Youden's J statistic (max TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y, probs)
    return thresholds[np.argmax(tpr - fpr)]


# ── 7. CROSS-VALIDATED EVALUATION ────────────────────────────────────────────
def run_cv(taxids, y, task_name):
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  Tier 2 Full (layers 6–7 unfrozen)")
    print(f"{'='*60}")

    taxids = np.array(taxids)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(len(y))

    pos_weight = torch.tensor(
        [(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    # Load model once — reuse across folds by resetting weights
    print("  Loading Vir2vec backbone (once for all folds)...")
    base_model = Vir2vecPartialFinetune(MODEL_NAME).to(DEVICE)
    import copy
    base_state = copy.deepcopy(base_model.state_dict())

    for fold, (train_idx, val_idx) in enumerate(cv.split(taxids, y)):
        print(f"\n  Fold {fold+1}/5")
        tr_taxids, val_taxids = taxids[train_idx].tolist(), taxids[val_idx].tolist()
        y_tr, y_val           = y[train_idx], y[val_idx]

        # Reset model weights to base state for each fold
        model = copy.deepcopy(base_model)
        model.load_state_dict(copy.deepcopy(base_state))
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        best_auroc, patience_counter, best_probs = 0, 0, None

        for epoch in range(args.epochs):
            # Shuffle training taxids each epoch
            perm = np.random.permutation(len(tr_taxids))
            sh_taxids = [tr_taxids[i] for i in perm]
            sh_y      = y_tr[perm]

            batches_x, batches_y = make_batches(sh_taxids, sh_y, args.batch_size)
            loss = train_epoch(model, batches_x, batches_y, optimizer, criterion)
            scheduler.step()

            probs_val = predict_all(model, val_taxids)
            try:
                auroc = roc_auc_score(y_val, probs_val)
            except Exception:
                auroc = 0.5

            if auroc > best_auroc:
                best_auroc   = auroc
                best_probs   = probs_val.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d} | loss={loss:.4f} | "
                      f"val AUROC={auroc:.4f} | best={best_auroc:.4f}")

            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        all_probs[val_idx] = best_probs
        print(f"  Fold {fold+1} best AUROC: {best_auroc:.4f}")
        del model
        torch.cuda.empty_cache()

    # Overall OOF metrics. Youden's J threshold instead of a fixed 0.5 —
    # Zoonotic is ~31% positive (more skewed under strict labels), so 0.5
    # understates MCC/F1 the same way it did for Vector-Borne before this fix.
    threshold = youden_threshold(y, all_probs)
    preds = (all_probs >= threshold).astype(int)
    metrics = {
        "AUROC":       round(roc_auc_score(y, all_probs), 4),
        "AUPRC":       round(average_precision_score(y, all_probs), 4),
        "MCC":         round(matthews_corrcoef(y, preds), 4),
        "F1_macro":    round(f1_score(y, preds, average="macro"), 4),
        "F1_weighted": round(f1_score(y, preds, average="weighted"), 4),
        "Brier":       round(brier_score_loss(y, all_probs), 4),
        "Threshold":   round(float(threshold), 4),
    }

    print(f"\n  Overall OOF metrics:")
    for k, v in metrics.items():
        print(f"    {k:<15} {v:.4f}")

    _plot_roc_pr(y, all_probs, task_name)
    _plot_calibration(y, all_probs, task_name)
    _plot_confusion_matrix(y, all_probs, threshold, task_name)

    return metrics


def _plot_roc_pr(y, probs, task_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{task_name} — Tier 2 Full ROC & PR Curves", fontsize=13)
    fpr, tpr, _ = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    axes[0].plot(fpr, tpr, color="darkorange", label=f"Tier 2 Full (AUC={auc:.3f})")
    axes[0].plot([0,1],[0,1],"k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()
    prec, rec, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    axes[1].plot(rec, prec, color="darkorange", label=f"Tier 2 Full (AP={ap:.3f})")
    axes[1].axhline(y.mean(), color="k", linestyle="--", lw=0.8,
                    label=f"Baseline ({y.mean():.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()
    plt.tight_layout()
    fname = FIGURES_DIR / f"tier2full_{task_name.replace(' ','_')}_roc_pr.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")



def _plot_confusion_matrix(y, probs, threshold, task_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{task_name} — Tier 2 Full Confusion Matrix (thresh={threshold:.3f})")
    plt.tight_layout()
    task_safe = task_name.replace(" ", "_")
    fname = FIGURES_DIR / f"{task_safe}_confusion_matrix.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")

def _plot_calibration(y, probs, task_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(f"{task_name} — Tier 2 Full Calibration")
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", color="darkorange", label="Tier 2 Full")
    ax.plot([0,1],[0,1],"k--", lw=0.8, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(); plt.tight_layout()
    fname = FIGURES_DIR / f"tier2full_{task_name.replace(' ','_')}_calibration.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")


# ── 8. RUN ────────────────────────────────────────────────────────────────────
all_results = {}

if args.task in ("h2h", "both"):
    m = run_cv(matched_taxids, y_h2h, "Human-to-Human")
    all_results["human_to_human"] = m

if args.task in ("zoo", "both"):
    m = run_cv(matched_taxids, y_zoo, "Zoonotic Spillover")
    all_results["zoonotic"] = m

# ── 9. SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TIER 2 FULL SUMMARY")
print("="*60)
for task, m in all_results.items():
    print(f"\n{task}")
    for k, v in m.items():
        print(f"  {k:<15} {v:.4f}")

out_path = OUTPUTS_DIR / "tier2_partial_unfreeze_results.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved: {out_path}")
print("Done! ✓")
