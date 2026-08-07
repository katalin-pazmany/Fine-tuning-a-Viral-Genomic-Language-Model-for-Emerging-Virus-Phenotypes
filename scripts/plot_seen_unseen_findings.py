"""
Findings figure: extrapolation to unseen viruses (the dissertation's headline)
=============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

For each phenotype, plots AUROC in the *prospective* setting -- train on viruses
Vir2vec saw during pre-training, test on viruses it never saw -- for Vir2vec and
three composition baselines, with 95 percent bootstrap confidence intervals and
paired-bootstrap significance vs Vir2vec.

Renders TWO themes: a white version (name.png/.pdf) for the dissertation, and a
dark-blue version (name_dark.png) that matches the galaxy figures for slides.
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
plt.rcParams["font.family"] = "Arial"

PHENO = [("human_to_human", "Human-to-human"),
         ("zoonotic_relaxed", "Zoonotic (relaxed)"),
         ("zoonotic_strict", "Zoonotic (strict)")]
ORDER = ["Vir2vec", "k-mer(k=4)", "Codon-usage", "GC+Length"]
LABEL = {"Vir2vec": "Vir2vec", "k-mer(k=4)": "k-mer (k=4)",
         "Codon-usage": "Codon usage", "GC+Length": "GC + length"}

# theme -> colours ("" = white for the thesis, "_dark" = navy for slides)
THEMES = {
    "": dict(bg="white", fg="#333333", sub="#555555", vir="#0072B2", base="#B0B0B0",
             win="#D55E00", err="#333333", chance="#999999", title="black", exts=("png", "pdf")),
    "_dark": dict(bg="#0A0E28", fg="#E8EAF2", sub="#AAB0C4", vir="#4DA6E8", base="#B9BEC9",
                  win="#FF8A3D", err="#C8CCD8", chance="#8890A4", title="#B0C4F5", exts=("png",)),
}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def render(sig, suffix, th):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True)
    fig.patch.set_facecolor(th["bg"])
    for ax, (key, title) in zip(axes, PHENO):
        ax.set_facecolor(th["bg"])
        s = sig[key]
        vir_name = next(k for k in s["methods"] if k.startswith("Vir2vec"))
        vals = {"Vir2vec": s["methods"][vir_name]}
        for m in ORDER[1:]:
            vals[m] = s["methods"][m]
        xs = range(len(ORDER))
        for x, m in zip(xs, ORDER):
            a = vals[m]["AUROC"]; lo, hi = vals[m]["CI95"]
            is_win = (m != "Vir2vec" and m in s["vs_vir2vec"]
                      and s["vs_vir2vec"][m]["delta_AUROC"] > 0
                      and s["vs_vir2vec"][m]["p_value"] < 0.05)
            c = th["vir"] if m == "Vir2vec" else (th["win"] if is_win else th["base"])
            ax.bar(x, a, color=c, width=0.72, zorder=3, edgecolor=th["bg"], linewidth=0.5)
            ax.errorbar(x, a, yerr=[[a - lo], [hi - a]], fmt="none",
                        ecolor=th["err"], elinewidth=1.3, capsize=4, zorder=4)
            ax.text(x, a + (hi - a) + 0.012, f"{a:.2f}", ha="center", va="bottom",
                    fontsize=9, color=th["fg"])
            if is_win:
                ax.text(x, hi + 0.055, stars(s["vs_vir2vec"][m]["p_value"]), ha="center",
                        va="bottom", fontsize=14, color=th["win"], fontweight="bold")
        ax.axhline(0.5, ls="--", lw=1, color=th["chance"], zorder=1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color=th["fg"])
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=25, ha="right", fontsize=9.5, color=th["fg"])
        ax.set_ylim(0.4, 0.92)
        ax.tick_params(colors=th["fg"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(th["fg"])
    axes[0].set_ylabel("AUROC on unseen viruses", fontsize=11, color=th["fg"])
    axes[0].text(-0.5, 0.51, "chance", fontsize=8, color=th["chance"], va="bottom")
    fig.suptitle("Predicting phenotypes for viruses Vir2vec never saw in pre-training",
                 fontsize=14, fontweight="bold", y=1.02, color=th["title"])
    fig.text(0.5, -0.06,
             "Train on seen viruses, test on unseen viruses (n = 615, same set under three labellings).  "
             "Bars = AUROC, whiskers = 95% bootstrap CI.  "
             "Orange = baseline significantly beats Vir2vec (paired bootstrap:  * p<0.05,  ** p<0.01,  *** p<0.001).",
             ha="center", fontsize=8.5, color=th["sub"])
    fig.tight_layout()
    for ext in th["exts"]:
        fig.savefig(FIG / f"seen_unseen_findings{suffix}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)


def main():
    sig = json.load(open(OUT / "seen_unseen_significance.json"))
    for suffix, th in THEMES.items():
        render(sig, suffix, th)
    print("saved seen_unseen_findings.png/.pdf (white) + seen_unseen_findings_dark.png (navy)")


if __name__ == "__main__":
    main()
