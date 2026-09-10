# Figure Reproduction

This directory contains the fixed source data and plotting code for the final
quantitative figures. The method name is consistently reported as FWT-MF-PIRN.
The plotting script reads the CSV files without changing their values.

## Files

- `data/main_SOTA_*.csv`: run-level and summary method comparison.
- `data/pairwise_statistics.csv` and `data/convergence_runs.csv`: statistical and convergence records.
- `data/complete_ablation_runs.csv` and `data/factorial_interaction_runs.csv`: ablation data.
- `data/analytical_generalization_runs.csv`, `data/material_topology_OOD_runs.csv`, and
  `data/robustness_runs.csv`: transfer and robustness records.
- `data/PIRN_*.csv`: residual-network holdout and sample-size results.
- `data/Frank_WT_hierarchy_runs.csv`, `data/mesh_convergence.csv`, and
  `data/energy_performance_cases.csv`: mechanics-hierarchy results.
- `data/shape_performance_front.csv`, `data/same_shape_performance_pairs.csv`, and
  `data/candidate_14D_designs.csv`: shape-performance design records.
- `data/material_curves.csv`, `data/device_response.csv`, and `data/cycle_retention.csv`:
  material and device validation records.
- `data/Table9_Frank_WT_hierarchy.csv` and `data/Table11_device_validation.csv`:
  fixed summaries used directly by Figures 3 and 8.
- `scripts/plot_all.py`: reads the fixed data and exports Figures 3-8.

## Run

```bash
python plotting/scripts/plot_all.py
```

Figures are written to `plotting/figures` as PDF, SVG, PNG, and TIFF files.
