"""
3D Plotly Network Visualisation — Futuristic Edition
======================================================
Creates two interactive 3D scatter plots:
    1. Human-to-Human Transmissibility risk
    2. Zoonotic Spillover risk

Usage:
    conda activate viral-phenotype
    python scripts/plot_3d_network.py

Output:
    outputs/results/figures_3d/h2h_3d_network.html
    outputs/results/figures_3d/zoo_3d_network.html
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "outputs" / "results" / "figures_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
pseudo_df = pd.read_csv(
    BASE_DIR / "outputs" / "results" / "semi_supervised" / "pseudo_labels.csv"
)

h2h_df = pd.read_excel(BASE_DIR / "data" / "complete_h2h_labels.xlsx")
zoo_df = pd.read_excel(BASE_DIR / "data" / "complete_zoonotic_labels.xlsx")
df     = h2h_df.merge(zoo_df[["virus_taxid","zoonotic"]], on="virus_taxid")

pseudo_df = pseudo_df.merge(
    df[["virus_taxid","virus","virus_family"]].drop_duplicates("virus_taxid"),
    on="virus_taxid", how="left"
)
pseudo_df["virus_family"] = pseudo_df["virus_family"].fillna("Unknown")
pseudo_df["virus"]        = pseudo_df["virus"].fillna(pseudo_df["virus_taxid"].astype(str))

# Normalise UMAP to [-10, 10] range for better spread
for col in ["umap_x", "umap_y"]:
    mn, mx = pseudo_df[col].min(), pseudo_df[col].max()
    pseudo_df[col] = (pseudo_df[col] - mn) / (mx - mn) * 20 - 10

labelled   = pseudo_df[pseudo_df["is_labelled"] == True]
unlabelled = pseudo_df[pseudo_df["is_labelled"] == False]
print(f"  Labelled: {len(labelled)}, Unlabelled: {len(unlabelled)}")

# ── 2. FUTURISTIC COLORSCALES ─────────────────────────────────────────────────
# H2H: deep blue -> cyan -> yellow -> red
H2H_COLORS = [
    [0.0,  "rgb(40,80,255)"],
    [0.25, "rgb(80,180,255)"],
    [0.5,  "rgb(100,240,240)"],
    [0.75, "rgb(255,220,50)"],
    [1.0,  "rgb(255,50,50)"],
]

# Zoo: deep purple -> teal -> lime -> hot green
ZOO_COLORS = [
    [0.0,  "rgb(80,40,180)"],
    [0.25, "rgb(40,160,220)"],
    [0.5,  "rgb(40,220,180)"],
    [0.75, "rgb(120,255,120)"],
    [1.0,  "rgb(0,255,80)"],
]

# ── 3. BUILD FIGURE ───────────────────────────────────────────────────────────
def make_3d_figure(df_lab, df_unl, z_col, title, colorscale, z_label):

    hover_unl = [
        f"<b>{row['virus']}</b><br>"
        f"Family: {row['virus_family']}<br>"
        f"H2H risk: {row['h2h_prob']:.1%}<br>"
        f"Zoonotic risk: {row['zoo_prob']:.1%}"
        for _, row in df_unl.iterrows()
    ]
    hover_lab = [
        f"<b>{row['virus']}</b><br>"
        f"Family: {row['virus_family']}<br>"
        f"H2H risk: {row['h2h_prob']:.1%}<br>"
        f"Zoonotic risk: {row['zoo_prob']:.1%}<br>"
        f"<i>✓ Ground truth label</i>"
        for _, row in df_lab.iterrows()
    ]

    # Unlabelled — small, semi-transparent
    trace_unl = go.Scatter3d(
        x=df_unl["umap_x"], y=df_unl["umap_y"], z=df_unl["umap_z"],
        mode="markers",
        name="Pseudo-labelled (~10k viruses)",
        hovertext=hover_unl, hoverinfo="text",
        marker=dict(
            size=4,
            color=df_unl[z_col],
            colorscale=colorscale,
            cmin=0, cmax=1,
            opacity=0.7,
            showscale=False,
            line=dict(width=0),
        )
    )

    # Labelled — larger, fully opaque, NO outline
    trace_lab = go.Scatter3d(
        x=df_lab["umap_x"], y=df_lab["umap_y"], z=df_lab["umap_z"],
        mode="markers",
        name="Ground truth labelled (1,373 viruses)",
        hovertext=hover_lab, hoverinfo="text",
        marker=dict(
            size=4,
            color=df_lab[z_col],
            colorscale=colorscale,
            cmin=0, cmax=1,
            opacity=0.95,
            showscale=True,
            colorbar=dict(
                title=dict(text=z_label, font=dict(size=11, color="rgba(180,220,255,0.8)")),
                thickness=12, len=0.5, x=1.02,
                tickfont=dict(color="rgba(180,220,255,0.7)", size=10),
                bgcolor="rgba(6,10,18,0.8)",
                bordercolor="rgba(100,200,255,0.2)",
            ),
            line=dict(width=0),  # NO outline
        )
    )

    fig = go.Figure(data=[trace_unl, trace_lab])

    axis_style = dict(
        backgroundcolor="rgb(6,10,18)",
        gridcolor="rgba(100,200,255,0.08)",
        showbackground=True,
        zerolinecolor="rgba(100,200,255,0.15)",
        tickfont=dict(color="rgba(150,200,255,0.5)", size=9),
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="rgba(180,220,255,0.9)", family="monospace"),
            x=0.5, y=0.97
        ),
        scene=dict(
            xaxis=dict(title="UMAP dimension 1", **axis_style),
            yaxis=dict(title="UMAP dimension 2", **axis_style),
            zaxis=dict(title="UMAP dimension 3", **axis_style),
            bgcolor="rgb(6,10,18)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            aspectmode="cube",
        ),
        paper_bgcolor="rgb(6,10,18)",
        font=dict(color="rgba(180,220,255,0.8)", family="monospace"),
        legend=dict(
            x=0.01, y=0.95,
            bgcolor="rgba(6,10,18,0.85)",
            bordercolor="rgba(100,200,255,0.2)",
            borderwidth=1,
            font=dict(size=10, color="rgba(180,220,255,0.7)"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=750,
    )
    return fig

fig_h2h = make_3d_figure(
    labelled, unlabelled,
    z_col="h2h_prob",
    title="Human-to-Human Transmissibility Risk · Vir2vec Embedding Space",
    colorscale=H2H_COLORS,
    z_label="H2H predicted risk"
)

fig_zoo = make_3d_figure(
    labelled, unlabelled,
    z_col="zoo_prob",
    title="Zoonotic Spillover Risk · Vir2vec Embedding Space",
    colorscale=ZOO_COLORS,
    z_label="Zoonotic predicted risk"
)

# ── 4. SAVE ───────────────────────────────────────────────────────────────────
fig_h2h.write_html(str(OUT_DIR / "h2h_3d_network.html"), include_plotlyjs="cdn")
fig_zoo.write_html(str(OUT_DIR / "zoo_3d_network.html"), include_plotlyjs="cdn")

print(f"Saved to: {OUT_DIR}")
print("Done! ✓")
