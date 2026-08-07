"""
Vector-borne ablation figure: does adding biting midges change the conclusion?
==============================================================================
Dissertation: Fine-tuning a Viral Genomic Language Model for Emerging Virus Phenotypes
Author: Katalin Pazmany

Best model per tier for vector-borne-to-humans transmission, full (mosquito + tick +
midge) vs mosquito-and-tick-only. Numbers from dissertation Table 'tab:vb-ablation'.

Renders a white version (name.png/.pdf) for the dissertation and a dark-blue version
(name_dark.png) matching the galaxy figures for slides.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
FIG = BASE / "outputs" / "figures_results"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "Arial"

TIERS = ["Tier 1\n(frozen + shallow)", "Tier 2\n(partial unfreeze)", "Tier 3\n(LoRA)"]
FULL = [0.804, 0.583, 0.735]      # mosquito + tick + midge
MT = [0.829, 0.654, 0.680]        # mosquito + tick only

THEMES = {
    "": dict(bg="white", fg="#333333", sub="#555555", chance="#999999", title="black",
             full="#0072B2", mt="#E69F00", exts=("png", "pdf")),
    "_dark": dict(bg="#0A0E28", fg="#E8EAF2", sub="#AAB0C4", chance="#8890A4", title="#B0C4F5",
                  full="#4DA6E8", mt="#FFC24D", exts=("png",)),
}


def render(suffix, th):
    x = range(len(TIERS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.patch.set_facecolor(th["bg"]); ax.set_facecolor(th["bg"])
    b1 = ax.bar([i - w/2 for i in x], FULL, w, color=th["full"], zorder=3,
                edgecolor=th["bg"], linewidth=0.6, label="Full (mosquito + tick + midge)")
    b2 = ax.bar([i + w/2 for i in x], MT, w, color=th["mt"], zorder=3,
                edgecolor=th["bg"], linewidth=0.6, label="Mosquito + tick only")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.008, f"{r.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9, color=th["fg"])
    ax.axhline(0.5, ls="--", lw=1, color=th["chance"], zorder=1)
    ax.text(-0.55, 0.505, "chance", fontsize=8, color=th["chance"], va="bottom")
    ax.annotate("best tier on both", xy=(0, 0.829), xytext=(0.15, 0.90),
                fontsize=9, color=th["full"], ha="left",
                arrowprops=dict(arrowstyle="->", color=th["full"], lw=1))
    ax.set_xticks(list(x)); ax.set_xticklabels(TIERS, fontsize=10, color=th["fg"])
    ax.set_ylim(0.4, 0.95); ax.set_ylabel("AUROC", fontsize=11, color=th["fg"])
    ax.tick_params(colors=th["fg"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(th["fg"])
    leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.0 if suffix else 0.9)
    for t in leg.get_texts():
        t.set_color(th["fg"])
    ax.set_title("Adding biting midges does not change the conclusion:\n"
                 "frozen embeddings (Tier 1) remain the best approach",
                 fontsize=12.5, fontweight="bold", pad=10, color=th["title"])
    fig.text(0.5, -0.02, "Vector-borne-to-humans transmission. Best model per tier; "
             "Tier 1 wins and the tier ordering (1 > 3 > 2) holds on both datasets.",
             ha="center", fontsize=8.5, color=th["sub"])
    fig.tight_layout()
    for ext in th["exts"]:
        fig.savefig(FIG / f"vector_borne_ablation{suffix}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)


def main():
    for suffix, th in THEMES.items():
        render(suffix, th)
    print("saved vector_borne_ablation.png/.pdf (white) + _dark.png (navy)")


if __name__ == "__main__":
    main()
