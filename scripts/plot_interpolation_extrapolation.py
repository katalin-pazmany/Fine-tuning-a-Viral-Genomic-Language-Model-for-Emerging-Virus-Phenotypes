"""
Companion figure: Vir2vec interpolates but does not extrapolate
===============================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Slope chart of Vir2vec's own AUROC under cross-validation (interpolation) vs the
train-seen/test-unseen prospective test (extrapolation), per phenotype.

Renders a white version (name.png/.pdf) for the dissertation and a dark-blue version
(name_dark.png) matching the galaxy figures for slides.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"
FIG = OUT / "figures_results"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "Arial"

PHENO = [("human_to_human", "Human-to-human"),
         ("zoonotic_relaxed", "Zoonotic (relaxed)"),
         ("zoonotic_strict", "Zoonotic (strict)")]
THEMES = {
    "": dict(bg="white", fg="#333333", sub="#555555", chance="#999999", title="black",
             cols=["#0072B2", "#D55E00", "#009E73"], exts=("png", "pdf")),
    "_dark": dict(bg="#0A0E28", fg="#E8EAF2", sub="#AAB0C4", chance="#8890A4", title="#B0C4F5",
                  cols=["#4DA6E8", "#FF8A3D", "#2FD48F"], exts=("png",)),
}


def render(sig, res, suffix, th):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    fig.patch.set_facecolor(th["bg"]); ax.set_facecolor(th["bg"])
    x0, x1 = 0, 1
    rows = []
    for (key, name), col in zip(PHENO, th["cols"]):
        model = sig[key]["vir2vec_model"]
        cell = res[key][f"Vir2vec / {model}"]
        rows.append((name, col, cell["cv_all"]["AUROC"], cell["train_seen_test_unseen"]["AUROC"]))

    for name, col, a_in, a_ex in rows:
        ax.plot([x0, x1], [a_in, a_ex], "-", color=col, lw=2.4, zorder=3,
                marker="o", markersize=9, markeredgecolor=th["bg"], markeredgewidth=1.2)
        ax.text(x0 - 0.03, a_in, f"{a_in:.2f}", ha="right", va="center", fontsize=10, color=col)
        ax.text(x1 + 0.03, a_ex, f"{a_ex:.2f}", ha="left", va="center", fontsize=10, color=col, fontweight="bold")

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
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=2)
                    if abs(label_y[i] - a_ex) > 0.012 else None)

    ax.axhline(0.5, ls="--", lw=1, color=th["chance"], zorder=1)
    ax.text(x0, 0.505, "chance", fontsize=8, color=th["chance"], va="bottom")
    ax.set_xlim(-0.28, 1.9); ax.set_ylim(0.45, 0.9)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Cross-validation\n(interpolation)", "Unseen viruses\n(extrapolation)"],
                       fontsize=11, color=th["fg"])
    ax.set_ylabel("Vir2vec AUROC", fontsize=11, color=th["fg"])
    ax.tick_params(colors=th["fg"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(th["fg"])
    ax.set_title("Vir2vec interpolates within known viral diversity,\nbut does not extrapolate to unseen viruses",
                 fontsize=12.5, fontweight="bold", pad=12, color=th["title"])
    fig.text(0.5, -0.02,
             "Same classifier per phenotype as the significance figure.  "
             "Left: 5-fold CV over all viruses.  Right: train on seen, test on unseen.",
             ha="center", fontsize=8.5, color=th["sub"])
    fig.tight_layout()
    for ext in th["exts"]:
        fig.savefig(FIG / f"interpolation_extrapolation{suffix}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)


def main():
    sig = json.load(open(OUT / "seen_unseen_significance.json"))
    res = json.load(open(OUT / "seen_unseen_results.json"))
    for suffix, th in THEMES.items():
        render(sig, res, suffix, th)
    print("saved interpolation_extrapolation.png/.pdf (white) + _dark.png (navy)")


if __name__ == "__main__":
    main()
