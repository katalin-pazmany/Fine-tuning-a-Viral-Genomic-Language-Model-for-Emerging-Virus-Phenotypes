# outputs/

Generated artefacts from the analysis pipeline. Large regenerable files
(embeddings, raw sequences, model weights) are intentionally not tracked — see
the repository `.gitignore`.

## Result data (the record)
- `*_results.json`, `results_summary*.csv` — per-tier and summary metrics.
- `seen_unseen_results.json`, `seen_unseen_significance.json` — the seen/unseen
  out-of-distribution analysis and its bootstrap significance.
- `baselines_*/` — composition and BLAST baseline results.
- `misclassification_dedup/` — per-virus and per-group error tables.
- `mollentze_benchmark/`, `intact_validation_results.json` — external validation.
- `vir2vec_seen_labels.csv`, `labels_matched.csv`, `accession_taxid_cache.csv`,
  `refseq_metadata.csv` — labels and lookup tables.
- `results*/`, `tier*/` — per-run result JSONs for each configuration
  (dedup vs non-dedup, strict vs relaxed zoonotic, vector-borne variants).

## Figures
- `figures_results/` — the headline results figures used in the dissertation
  (seen/unseen bar chart, interpolation–extrapolation slope, vector-borne ablation).
- `figures_umap/` — embedding-space maps (galaxy network, UMAP panels).

## Networks
- `*.gexf` — Gephi network files of the Vir2vec embedding space.
- `results/figures_3d/*.html` — interactive 3D embedding views.

Per-run diagnostic plots (ROC/PR curves, calibration, confusion matrices) are
regenerable from the result JSONs by re-running the pipeline scripts, so they are
not kept in the repository.
