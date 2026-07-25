"""
Companion figure: Vir2vec interpolates but does not extrapolate
===============================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

A slope chart of Vir2vec's *own* AUROC in two regimes, per phenotype:
  - Cross-validation (interpolation): standard 5-fold CV over all viruses, so
    train and test share the same viral diversity.
  - Unseen viruses (extrapolation): train on viruses Vir2vec saw in pre-training,
    test on viruses it never saw.
Each phenotype line falls from left to right. The gap is the price of extrapolation
to the emerging-virus setting -- steep for human-to-human, gentle for zoonotic.

Uses the same per-phenotype classifier as the significance figure so the right-hand
endpoints match. Reads outputs/seen_unseen_{results,significance}.json.
Writes outputs/figures_results/interpolation_extrapolation.png/.pdf.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
FIG = OUT / "figures_results"
FIG.mkdir(parents=True, exist_ok=True)

try:
    font_manager.fontManager.addfont("/System/Library/Fonts/Avenir Next.ttc")
    plt.rcParams["font.family"] = "Avenir Next"
except Exception:
    pass

PHENO = [("human_to_human", "Human-to-human", "#0072B2"),
         ("zoonotic_relaxed", "Zoonotic (relaxed)", "#D55E00"),
         ("zoonotic_strict", "Zoonotic (strict)", "#009E73")]
CHANCE = "#999999"


def main():
    sig = json.load(open(OUT / "seen_unseen_significance.json"))
    res = json.load(open(OUT / "seen_unseen_results.json"))

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    x0, x1 = 0, 1
    rows = []
    for key, name, col in PHENO:
        model = sig[key]["vir2vec_model"]
        cell = res[key][f"Vir2vec / {model}"]
        rows.append((name, col, cell["cv_all"]["AUROC"],
                     cell["train_seen_test_unseen"]["AUROC"]))

    for name, col, a_in, a_ex in rows:
        ax.plot([x0, x1], [a_in, a_ex], "-", color=col, lw=2.4, zorder=3,
                marker="o", markersize=9, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(x0 - 0.03, a_in, f"{a_in:.2f}", ha="right", va="center",
                fontsize=10, color=col)
        ax.text(x1 + 0.03, a_ex, f"{a_ex:.2f}", ha="left", va="center",
                fontsize=10, color=col, fontweight="bold")

    # declutter the right-hand phenotype labels: enforce a minimum vertical gap
    order = sorted(range(len(rows)), key=lambda i: rows[i][3])
    gap, ly = 0.032, None
    label_y = {}
    for i in order:
        y = rows[i][3]
        if ly is not None and y - ly < gap:
            y = ly + gap
        label_y[i] = ly = y
    for i, (name, col, a_in, a_ex) in enumerate(rows):
        ax.annotate(f"{name}  (−{a_in - a_ex:.2f})",
                    xy=(x1 + 0.045, a_ex), xytext=(x1 + 0.16, label_y[i]),
                    ha="left", va="center", fontsize=9.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    shrinkA=0, shrinkB=2) if abs(label_y[i] - a_ex) > 0.012 else None)

    ax.axhline(0.5, ls="--", lw=1, color=CHANCE, zorder=1)
    ax.text(x0, 0.505, "chance", fontsize=8, color=CHANCE, va="bottom")

    ax.set_xlim(-0.28, 1.9)
    ax.set_ylim(0.45, 0.9)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Cross-validation\n(interpolation)", "Unseen viruses\n(extrapolation)"],
                       fontsize=11)
    ax.set_ylabel("Vir2vec AUROC", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Vir2vec interpolates within known viral diversity,\nbut does not extrapolate to unseen viruses",
                 fontsize=12.5, fontweight="bold", pad=12)
    fig.text(0.5, -0.02,
             "Same classifier per phenotype as the significance figure.  "
             "Left: 5-fold CV over all viruses.  Right: train on seen, test on unseen.",
             ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"interpolation_extrapolation.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"saved {FIG/'interpolation_extrapolation.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
