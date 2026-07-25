"""
Findings figure: extrapolation to unseen viruses (the dissertation's headline)
=============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

For each phenotype, plots AUROC in the *prospective* setting -- train on viruses
Vir2vec saw during pre-training, test on viruses it never saw -- for Vir2vec and
three composition baselines, with 95 percent bootstrap confidence intervals and
paired-bootstrap significance vs Vir2vec. The message: on genuinely novel viruses
Vir2vec has no advantage, and for human-to-human transmission it is significantly
beaten by codon usage and k-mers. Interpolation, not extrapolation.

Reads outputs/seen_unseen_significance.json (+ _results.json for the seen->unseen
drop annotation). Writes outputs/figures_results/seen_unseen_findings.png/.pdf.
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

# use Avenir Next to match the other figures, fall back gracefully
try:
    for w in ("Regular", "Demi Bold"):
        font_manager.fontManager.addfont("/System/Library/Fonts/Avenir Next.ttc")
    plt.rcParams["font.family"] = "Avenir Next"
except Exception:
    pass

VIR_C = "#0072B2"      # Vir2vec (highlight)
BASE_C = "#B0B0B0"     # baselines (muted)
WIN_C = "#D55E00"      # baseline that significantly beats Vir2vec
CHANCE = "#999999"

PHENO = [("human_to_human", "Human-to-human"),
         ("zoonotic_relaxed", "Zoonotic (relaxed)"),
         ("zoonotic_strict", "Zoonotic (strict)")]
ORDER = ["Vir2vec", "k-mer(k=4)", "Codon-usage", "GC+Length"]
LABEL = {"Vir2vec": "Vir2vec", "k-mer(k=4)": "k-mer (k=4)",
         "Codon-usage": "Codon usage", "GC+Length": "GC + length"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    sig = json.load(open(OUT / "seen_unseen_significance.json"))
    res = json.load(open(OUT / "seen_unseen_results.json"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True)
    for ax, (key, title) in zip(axes, PHENO):
        s = sig[key]
        vir_name = next(k for k in s["methods"] if k.startswith("Vir2vec"))
        # method -> (auroc, lo, hi)
        vals = {}
        vals["Vir2vec"] = s["methods"][vir_name]
        for m in ORDER[1:]:
            vals[m] = s["methods"][m]

        xs = range(len(ORDER))
        for x, m in zip(xs, ORDER):
            a = vals[m]["AUROC"]; lo, hi = vals[m]["CI95"]
            is_win = (m != "Vir2vec" and m in s["vs_vir2vec"]
                      and s["vs_vir2vec"][m]["delta_AUROC"] > 0
                      and s["vs_vir2vec"][m]["p_value"] < 0.05)
            c = VIR_C if m == "Vir2vec" else (WIN_C if is_win else BASE_C)
            ax.bar(x, a, color=c, width=0.72, zorder=3,
                   edgecolor="white", linewidth=0.5)
            ax.errorbar(x, a, yerr=[[a - lo], [hi - a]], fmt="none",
                        ecolor="#333333", elinewidth=1.3, capsize=4, zorder=4)
            ax.text(x, a + (hi - a) + 0.012, f"{a:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#333333")
            if is_win:
                st = stars(s["vs_vir2vec"][m]["p_value"])
                ax.text(x, hi + 0.055, st, ha="center", va="bottom",
                        fontsize=14, color=WIN_C, fontweight="bold")

        ax.axhline(0.5, ls="--", lw=1, color=CHANCE, zorder=1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=25, ha="right", fontsize=9.5)
        ax.set_ylim(0.4, 0.92)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("AUROC on unseen viruses", fontsize=11)
    axes[0].text(-0.5, 0.51, "chance", fontsize=8, color=CHANCE, va="bottom")

    fig.suptitle("Predicting phenotypes for viruses Vir2vec never saw in pre-training",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "Train on seen viruses, test on unseen viruses (n = 615, same set under three labellings).  "
             "Bars = AUROC, whiskers = 95% bootstrap CI.  "
             "Orange = baseline significantly beats Vir2vec (paired bootstrap:  * p<0.05,  ** p<0.01,  *** p<0.001).",
             ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"seen_unseen_findings.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"saved {FIG/'seen_unseen_findings.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
