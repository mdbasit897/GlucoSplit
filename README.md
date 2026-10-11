# Subject-Level Validation Overturns Reported Accuracy in ECG-Based Glucose Estimation

Code, results and figures for a leakage-controlled re-analysis of ECG-based
blood-glucose estimation on two independent ambulatory cohorts.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

## What this repository shows

Non-invasive glucose estimation from the ECG has repeatedly been reported to
reach clinically acceptable accuracy. We show that the reported accuracy is a
property of how the models are evaluated, not of a physiological signal.

Holding the model, the feature space and the data fixed and varying only the
data-partitioning scheme, predictive R-squared falls from +0.25 to -0.08 on
D1NAMO and from +0.39 to -0.06 on PhysioCGM. Under leakage-free nested
leave-one-subject-out evaluation neither cohort yields a recoverable signal: the
model does not exceed a within-subject label-permutation control, and it is
beaten by a predictor that emits each held-out participant's own mean glucose
and uses no ECG at all.

Two further results generalise beyond this task:

- **Leakage runs through two independent channels.** Window replication, which
  grouping by glucose event removes, and subject identity, which it does not.
  Event-level grouping - the safeguard the field has adopted - closes only one.
- **Clarke Error Grid Zone A+B carries no evidence of model quality.** It was
  86.5% and 93.8% here, and a constant predictor matched both to one decimal
  place.

## Verify the paper without downloading any data

Every number in the manuscript derives from the small result files in
`results/`. You do not need the raw datasets, the 75 GB of intermediates, or any
model training to check them.

```bash
git clone https://github.com/<USER>/ecg-glucose-leakage-audit.git
cd ecg-glucose-leakage-audit
pip install -r requirements.txt
python src/verify_paper_numbers.py
```

Runs in under ten seconds. It recomputes every table and every numeric claim in
the paper from the stored event-level predictions and prints a PASS/FAIL line
per item, with the manuscript value and the recomputed value side by side.

## Repository layout

```
src/        analysis pipeline (phases 1-5) + the verification script
results/    every metric file the paper reports, per cohort
figures/    all figures, as generated and as used in the manuscript
docs/       data access, reproduction, results manifest, pipeline notes
legacy/     the SUPERSEDED v1 scripts, retained for transparency
paper/      the manuscript PDF
```

## Pipeline

| Phase | Script | Does |
|---|---|---|
| 1 | `src/d1namo/phase1_preprocess.py` | D1NAMO: filter, window, align ECG to glucose |
| 1 | `src/physiocgm/phase1_physiocgm.py` | PhysioCGM: same contract, reading-driven alignment |
| 2 | `src/phase2_evaluate.py` | Split-severity ablation, nested LOSO, baseline ladder, permutation control, lag sweep |
| 3 | `src/phase3_figures.py` | Figures 1-9 |
| 4 | `src/phase4_transfer.py` | Bidirectional frozen-model cross-cohort transfer |
| 4 | `src/phase4_figures.py` | Figures 10-11 |
| 5 | `src/phase5_bootstrap_ci.py` | Subject-cluster bootstrap confidence intervals |

Full reproduction instructions, including runtimes and memory requirements, are
in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Data

Neither dataset is redistributed here. Both are public, under their own terms:

- **D1NAMO** - Dubosson et al., *Informatics in Medicine Unlocked* 13:92-100
  (2018). https://doi.org/10.5281/zenodo.5651217
- **PhysioCGM** - Quamer et al., *Scientific Data* 12:1822 (2025).
  https://doi.org/10.6084/m9.figshare.28136294

See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## Reproducibility notes

- All stochastic components are seeded (seed 42). Random number use is confined
  to the label-permutation control, window subsampling, and bootstrap
  resampling; no data are generated or simulated anywhere in this repository.
- Intermediate `.pkl` files (about 75 GB) are deliberately not tracked. They are
  regenerable from the raw datasets by the phase 1 scripts.
- Known caveats in the alignment implementation, and what they do and do not
  affect, are documented in [`docs/PIPELINE.md`](docs/PIPELINE.md).

## License

MIT - see [LICENSE](LICENSE). The datasets carry their own licences.

## Contact

Md Basit Azam - mdbasit@tezu.ernet.in
Department of Computer Science & Engineering, Tezpur University, Assam, India
