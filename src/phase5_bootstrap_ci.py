#!/usr/bin/env python3
"""
PHASE 5 - cluster-bootstrap confidence intervals for the pooled event-level
metrics reported in the primary nested-LOSO evaluation (tau = +5 min).

This does NOT re-run any model. It resamples the event-level predictions that
phase2_evaluate.py already saved to `event_predictions.npz`, so it runs in
seconds. The resampling unit is the SUBJECT (the independent cluster), matching
the leave-one-subject-out design: whole subjects are drawn with replacement,
all of a drawn subject's events travel together, and the pooled metric is
recomputed on each bootstrap replicate.

Why subject-level (cluster) bootstrap rather than event-level:
    Events within a subject are not independent (they share the subject's ECG
    morphology and glucose level). Resampling events individually would
    understate the CI. Resampling subjects respects the dependence structure and
    is the honest interval for a LOSO estimate.

Input contract:
    event_predictions.npz with arrays:
        event_id  (n_events,) str   e.g. "c1s01|r123" or "008_diabetes|g4"
        y_true    (n_events,) float reference glucose (mg/dL)
        y_pred    (n_events,) float model prediction  (mg/dL)
    Subject is parsed as the prefix of event_id before the first '|', which is
    how both phase1_preprocess.py (D1NAMO) and phase1_physiocgm.py build it.

Usage
-----
    python phase5_bootstrap_ci.py --npz results_d1namo/event_predictions.npz \
        --label D1NAMO --n-boot 10000
    python phase5_bootstrap_ci.py --npz results_physiocgm/event_predictions.npz \
        --label PhysioCGM --n-boot 10000

Add --subject-sep '|' to change the delimiter, or --subject-from-npz if you
saved a `subject` array explicitly.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RNG = 42


# ---- metrics (identical definitions to phase2_evaluate.py) ------------------

def _mard(y_true, y_pred):
    ok = y_true > 0
    return float(np.mean(np.abs(y_pred[ok] - y_true[ok]) / y_true[ok]) * 100) if ok.any() else np.nan


def _metrics(y_true, y_pred):
    return {
        'r2':   float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        'mard': _mard(y_true, y_pred),
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ---- cluster bootstrap ------------------------------------------------------

def cluster_bootstrap(y_true, y_pred, subjects, n_boot=10000, alpha=0.05, seed=RNG):
    """
    Resample SUBJECTS with replacement; recompute pooled event-level metrics on
    each replicate. Returns the point estimate (on the observed data) and the
    percentile CI over replicates, per metric.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    # index lists per subject, so a drawn subject contributes all its events
    idx_by_subj = {s: np.flatnonzero(subjects == s) for s in uniq}

    point = _metrics(y_true, y_pred)
    keys = list(point.keys())
    draws = {k: np.empty(n_boot) for k in keys}

    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_subj[s] for s in chosen])
        m = _metrics(y_true[idx], y_pred[idx])
        for k in keys:
            draws[k][b] = m[k]

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k in keys:
        d = draws[k][np.isfinite(draws[k])]
        out[k] = {
            'point': round(point[k], 4),
            'ci_low': round(float(np.percentile(d, lo_q)), 4),
            'ci_high': round(float(np.percentile(d, hi_q)), 4),
            'boot_mean': round(float(np.mean(d)), 4),
            'boot_median': round(float(np.median(d)), 4),
            'n_valid_replicates': int(d.size),
        }
    return out, point


# ---- io ---------------------------------------------------------------------

def load_events(npz_path, subject_sep='|', subject_from_npz=False):
    d = np.load(npz_path, allow_pickle=True)
    y_true = np.asarray(d['y_true'], dtype=float)
    y_pred = np.asarray(d['y_pred'], dtype=float)

    if subject_from_npz and 'subject' in d:
        subjects = np.asarray(d['subject']).astype(str)
    else:
        if 'event_id' not in d:
            raise KeyError(
                f"{npz_path} has no 'event_id'; pass --subject-from-npz if you "
                f"saved a 'subject' array. Keys present: {list(d.keys())}")
        eid = np.asarray(d['event_id']).astype(str)
        subjects = np.array([e.split(subject_sep, 1)[0] for e in eid])

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok], subjects[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='path to event_predictions.npz')
    ap.add_argument('--label', default='cohort')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--alpha', type=float, default=0.05, help='0.05 -> 95%% CI')
    ap.add_argument('--subject-sep', default='|')
    ap.add_argument('--subject-from-npz', action='store_true')
    ap.add_argument('--out', default=None, help='optional json output path')
    args = ap.parse_args()

    y_true, y_pred, subjects = load_events(
        args.npz, args.subject_sep, args.subject_from_npz)
    n_ev, n_sub = len(y_true), len(np.unique(subjects))
    print(f"\n{args.label}: {n_ev} events across {n_sub} subjects "
          f"({args.n_boot:,} bootstrap replicates, "
          f"{int(100*(1-args.alpha))}% percentile CI)")
    if n_sub < 3:
        print("  WARNING: <3 subjects; a subject-cluster bootstrap is degenerate here.")

    ci, point = cluster_bootstrap(
        y_true, y_pred, subjects, n_boot=args.n_boot, alpha=args.alpha)

    print(f"\n  {'metric':<6} {'point':>9} {'95% CI':>22}")
    print("  " + "-" * 40)
    for k in ('r2', 'mard', 'mae', 'rmse'):
        c = ci[k]
        unit = '%' if k == 'mard' else ''
        print(f"  {k:<6} {c['point']:>9.3f} "
              f"[{c['ci_low']:>8.3f}, {c['ci_high']:>8.3f}]{unit}")

    result = {'label': args.label, 'n_events': n_ev, 'n_subjects': n_sub,
              'n_boot': args.n_boot, 'ci': ci}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == '__main__':
    main()
