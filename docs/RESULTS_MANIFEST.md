# Results manifest

| File | Feeds | Paper item |
|---|---|---|
| results/d1namo/split_severity_ablation.csv | Table 2 (D1NAMO rows), Fig 1a | split-severity |
| results/physiocgm/split_severity_ablation.csv | Table 2 (PhysioCGM rows), Fig 1b | split-severity |
| results/*/pooled_metrics.json | Table 4 point estimates; Sec "Cross-Subject" | nested LOSO |
| results/*/ci_*.json | Table 4 confidence intervals | cluster bootstrap |
| results/*/event_predictions.npz | Figs 6, 7; all Clarke zones; prediction ranges | event-level predictions |
| results/*/baseline_ladder.csv | Fig 3 | baseline ladder |
| results/*/label_permutation.csv | Fig 2 | permutation control |
| results/*/per_subject.csv | Fig 4; per-subject correlations; inner-CV range | per-subject |
| results/*/by_glycemic_range.csv | Table 5, Fig 8 | glycaemic strata |
| results/*/feature_selection_frequency.csv | Fig 5; feature-stability section | RFECV stability |
| results/physiocgm_lag/lag_sweep.csv | Table 3 | 7-arm lag sweep |
| results/physiocgm_lag/common_support_7arm.json | Table 3 footnote (32,524 events) | common support |
| results/d1namo/lag_sweep.csv | Fig 9; D1NAMO common support = 0 | lag identifiability |
| results/transfer/transfer_metrics.json | Table 6, Figs 10-11 | cross-cohort transfer |
| results/transfer/transfer_perm_*.csv | Table 6 permutation floors | transfer controls |
| results/transfer/transfer_*_by_range.csv | transfer stratified claims | transfer strata |
