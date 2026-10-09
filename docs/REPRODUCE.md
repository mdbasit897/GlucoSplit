# Reproducing the results

## Tier 1 — verify every number (10 seconds, no data needed)

    pip install -r requirements.txt
    python src/verify_paper_numbers.py

Recomputes every table and inline claim from the files in results/.

## Tier 2 — re-run the full pipeline (requires both datasets)

Hardware used: 16 cores, 125 GB RAM. Peak memory 31 GB (lag sweep).
Disk: about 75 GB of intermediates.

    # Phase 1 — preprocessing and alignment (2-5 h per cohort per tau)
    python src/d1namo/phase1_preprocess.py   --tau 5 --window-width 10 \
        --base-path <D1NAMO_ROOT>
    python src/physiocgm/phase1_physiocgm.py --tau 5 --window-width 10 \
        --windows-per-reading 4 --base-path <PHYSIOCGM_ROOT>

    # Phase 2 — evaluation (ablation minutes; nested LOSO about 30 h on PhysioCGM)
    python src/phase2_evaluate.py --data preprocessed_data_taup5.pkl --mode ablation
    python src/phase2_evaluate.py --data preprocessed_data_taup5.pkl --mode loso
    python src/phase2_evaluate.py --data preprocessed_data_taup5.pkl --mode permutation

    # Lag sweep (PhysioCGM; about 1 h per arm with --no-rfecv)
    python src/phase2_evaluate.py --mode lagsweep \
        --lag-glob 'physiocgm_tau*.pkl' --no-rfecv --n-iter 5

    # Phase 4 — cross-cohort transfer (about 1 h)
    python src/phase4_transfer.py --d1namo <...> --physiocgm <...>

    # Phase 5 — bootstrap CIs (seconds)
    python src/phase5_bootstrap_ci.py --npz results/d1namo/event_predictions.npz \
        --label D1NAMO --n-boot 10000

    # Figures
    python src/phase3_figures.py --results-dir results/d1namo --out-dir figures/d1namo
    python src/phase3_figures.py --results-dir results/physiocgm \
        --out-dir figures/physiocgm --cohort-labels 'Cohort c1,Cohort c2'

Note the --cohort-labels flag for PhysioCGM. Without it the per-subject figure
is labelled T1D/Healthy, which is incorrect for that cohort.

All stochastic components are seeded (42); reruns reproduce the published
numbers exactly on the pinned package versions.
