#!/usr/bin/env python3
"""
PHASE 2 - leakage-free evaluation of ECG-based glucose estimation.

Usage
-----
    python phase2_evaluate.py --data preprocessed_data_tau5.pkl --mode ablation
    python phase2_evaluate.py --data preprocessed_data_tau5.pkl --mode loso
    python phase2_evaluate.py --data preprocessed_data_tau5.pkl --mode permutation
    python phase2_evaluate.py --mode lagsweep --lag-glob 'preprocessed_data_tau*.pkl'

Recommended order: ablation -> loso -> permutation -> lagsweep.
The ablation is cheap and tells you how much of the original result was real.

"""

import argparse
import glob as globmod
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     LeaveOneGroupOut, RandomizedSearchCV,
                                     ShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pywt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
FS = 250
RNG = 42


# =============================================================================
# Feature extraction  (51-dim space, carried over from the original)
# =============================================================================

def _safe_div(a, b):
    return float(a) / float(b) if b not in (0, 0.0) else 0.0


def _sample_entropy(x, m=2, r_mult=0.2):
    """
    Vectorised SampEn.

    The original used a quadruple-nested Python loop per window. Over 240,919
    windows that dominates runtime for no reason. Same definition, ~100x faster.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_mult * np.std(x)
    if r <= 0:
        return 0.0

    def _count(mm):
        if n - mm + 1 < 2:
            return 0
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        np.fill_diagonal(d, np.inf)
        return int(np.sum(d < r))

    b, a = _count(m), _count(m + 1)
    return float(-np.log(a / b)) if b > 0 and a > 0 else 0.0


def extract_ecg_features(X, fs=FS, verbose=True):
    """Amplitude / frequency / wavelet / HRV features. Per-window, independent."""
    rows = []
    it = range(len(X))

    for k in it:
        if verbose and k % 20000 == 0 and k:
            print(f"    features {k:,}/{len(X):,}")

        ecg = np.nan_to_num(np.asarray(X[k], dtype=float).ravel(),
                            nan=0.0, posinf=0.0, neginf=0.0)
        f = {}

        # --- amplitude statistics (14) ---
        if np.any(ecg != 0) and len(ecg) > 1:
            f.update({
                'mean': np.mean(ecg), 'std': np.std(ecg),
                'min': np.min(ecg), 'max': np.max(ecg),
                'median': np.median(ecg),
                'skewness': stats.skew(ecg), 'kurtosis': stats.kurtosis(ecg),
                'rms': np.sqrt(np.mean(ecg ** 2)),
                'ptp': np.ptp(ecg), 'energy': np.sum(ecg ** 2),
            })
        else:
            f.update(dict.fromkeys(['mean', 'std', 'min', 'max', 'median', 'skewness',
                                    'kurtosis', 'rms', 'ptp', 'energy'], 0.0))
        for p in (10, 25, 75, 90):
            f[f'percentile_{p}'] = float(np.percentile(ecg, p)) if len(ecg) else 0.0

        # --- frequency domain ---
        try:
            fft_vals = np.nan_to_num(np.abs(np.fft.rfft(ecg)), nan=0.0, posinf=0.0)
            fft_freq = np.fft.rfftfreq(len(ecg), 1 / fs)
            f.update({
                'fft_mean': np.mean(fft_vals), 'fft_std': np.std(fft_vals),
                'fft_max': np.max(fft_vals),
                'fft_max_freq': fft_freq[int(np.argmax(fft_vals))],
                'fft_energy': np.sum(fft_vals ** 2),
            })
            for name, lo, hi in (('vlf', 0.003, 0.04), ('lf', 0.04, 0.15), ('hf', 0.15, 0.4)):
                msk = (fft_freq >= lo) & (fft_freq < hi)
                f[f'{name}_power'] = float(np.sum(fft_vals[msk] ** 2)) if msk.any() else 0.0
            f['lf_hf_ratio'] = _safe_div(f['lf_power'], f['hf_power'])

            top = np.argsort(fft_vals)[-5:]
            for i, ind in enumerate(top):
                f[f'dom_freq_{i}'] = float(fft_freq[ind])
                f[f'dom_amp_{i}'] = float(fft_vals[ind])
        except Exception:
            for key in ('fft_mean', 'fft_std', 'fft_max', 'fft_max_freq', 'fft_energy',
                        'vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio'):
                f.setdefault(key, 0.0)
            for i in range(5):
                f.setdefault(f'dom_freq_{i}', 0.0)
                f.setdefault(f'dom_amp_{i}', 0.0)

        # --- clean for peak detection ---
        try:
            b, a = signal.butter(3, [0.5, 40], btype='bandpass', fs=fs)
            clean = np.nan_to_num(signal.filtfilt(b, a, ecg), nan=0.0, posinf=0.0)
        except Exception:
            clean = ecg.copy()

        # --- wavelet (db4, level 4) ---
        try:
            coeffs = pywt.wavedec(clean, 'db4', level=4)
            for i, c in enumerate(coeffs):
                nm = 'approx' if i == 0 else f'detail_{i}'
                f[f'wavelet_{nm}_energy'] = float(np.sum(c ** 2))
                nz = np.abs(c[np.abs(c) > 1e-10])
                f[f'wavelet_{nm}_entropy'] = float(
                    -np.sum((nz ** 2) * np.log(nz ** 2)) / np.sum(nz ** 2)) if nz.size else 0.0
        except Exception:
            for i in range(5):
                nm = 'approx' if i == 0 else f'detail_{i}'
                f[f'wavelet_{nm}_energy'] = 0.0
                f[f'wavelet_{nm}_entropy'] = 0.0

        # --- HRV ---
        hrv = dict.fromkeys(['mean_rr', 'std_rr', 'rmssd', 'nn50', 'pnn50',
                             'sdnn', 'sdsd', 'sample_entropy'], 0.0)
        try:
            if np.std(clean) > 0:
                peaks, _ = signal.find_peaks(clean, height=0.5 * np.max(clean),
                                             distance=int(fs * 0.5))
                if len(peaks) > 1:
                    rr = np.diff(peaks) / fs
                    drr = np.diff(rr)
                    hrv.update({
                        'mean_rr': np.mean(rr), 'std_rr': np.std(rr),
                        'sdnn': np.std(rr),
                        'rmssd': np.sqrt(np.mean(drr ** 2)) if drr.size else 0.0,
                        'nn50': float(np.sum(np.abs(drr) > 0.05)) if drr.size else 0.0,
                        'pnn50': _safe_div(np.sum(np.abs(drr) > 0.05), len(rr)),
                        'sdsd': np.std(drr) if drr.size else 0.0,
                        'sample_entropy': _sample_entropy(rr) if len(rr) > 10 else 0.0,
                    })
        except Exception:
            pass
        f.update(hrv)

        rows.append({k2: (v if np.isfinite(v) else 0.0) for k2, v in f.items()})

    df = pd.DataFrame(rows)

    banned = {'subject', 'subject_id', 'is_diabetic', 'cohort', 'subject_cluster',
              'time_diff_minutes', 'physiological_lag', 'glucose_event', 'session'}
    leaked = banned & set(df.columns)
    if leaked:
        raise ValueError(f"Leaky columns in feature matrix: {sorted(leaked)}")

    return df


# =============================================================================
# Metrics
# =============================================================================

def mard(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = y_true > 0
    return float(np.mean(np.abs(y_pred[ok] - y_true[ok]) / y_true[ok]) * 100) if ok.any() else np.nan


def clarke_zone(ref, pred):
    """
    Full Clarke Error Grid zone assignment, all five zones.
    """
    ref, pred = float(ref), float(pred)
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return 'A'
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return 'E'
    if ((70 <= ref <= 290) and pred >= ref + 110) or \
       ((130 <= ref <= 180) and pred <= (7.0 / 5.0) * ref - 182):
        return 'C'
    if (ref >= 240 and 70 <= pred <= 180) or \
       (ref <= 175.0 / 3.0 and 70 <= pred <= 180) or \
       ((175.0 / 3.0 <= ref <= 70) and pred >= (6.0 / 5.0) * ref):
        return 'D'
    return 'B'


def clarke_distribution(y_true, y_pred):
    z = [clarke_zone(a, b) for a, b in zip(y_true, y_pred)]
    n = len(z)
    return {f'clarke_{k}_pct': round(100.0 * z.count(k) / n, 2) for k in 'ABCDE'}


def compute_metrics(y_true, y_pred, label="", with_clarke=True):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {
        'label': label,
        'n': int(len(y_true)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        'mard': mard(y_true, y_pred),
    }
    if with_clarke and len(y_true):
        out.update(clarke_distribution(y_true, y_pred))
    return out


def metrics_by_glycemic_range(y_true, y_pred):
    """Reviewer comment 5: the hypoglycemia gap is the safety-critical number."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    bands = {'hypoglycemia (<70)': y_true < 70,
             'normoglycemia (70-180)': (y_true >= 70) & (y_true <= 180),
             'hyperglycemia (>180)': y_true > 180}
    rows = []
    for name, m in bands.items():
        if m.sum() >= 2:
            rows.append(compute_metrics(y_true[m], y_pred[m], name, with_clarke=False))
    return pd.DataFrame(rows)


def aggregate_to_events(y_true, y_pred, events):
    """

    240,919 windows are not 240,919 observations - they are N_events
    observations, each measured many times. Metrics computed over windows
    inflate the effective n by the replication factor, which makes every
    confidence interval and p-value in the paper wrong.
    """
    df = pd.DataFrame({'y': y_true, 'p': y_pred, 'e': events}).dropna()
    g = df.groupby('e', sort=True).agg(y=('y', 'first'), p=('p', 'median')).reset_index()
    return g['y'].values, g['p'].values, g['e'].values


# =============================================================================
# In-fold modelling
# =============================================================================

def _pipeline(model):
    """Imputation and scaling live inside the fold. Never fit on all data."""
    return Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler()),
                     ('model', model)])


def _grouped_splits(X, y, groups, n_splits=5):
    """
    Precomputed grouped splits.

    RFECV and RandomizedSearchCV cannot forward a `groups=` argument through a
    Pipeline. sklearn accepts any iterable of (train, val) index pairs as `cv`,
    which is the only clean way to keep subject boundaries inside nested CV.
    """
    n = int(min(n_splits, len(np.unique(groups))))
    if n < 2:
        raise ValueError("need >= 2 groups")
    return list(GroupKFold(n_splits=n).split(X, y, groups=groups))


PARAM_DIST = {
    'RandomForest': {
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None],
    },
    'XGBoost': {
        'model__n_estimators': [200, 300, 400],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0],
    },
}


def _make_model(name):
    return {'RandomForest': RandomForestRegressor(random_state=RNG, n_jobs=-1),
            'XGBoost': XGBRegressor(tree_method='hist', random_state=RNG)}[name]


def _fit_fold(X_tr, y_tr, g_tr, model_name='RandomForest', n_inner=5,
              n_iter=20, do_rfecv=True):
    """
    Feature selection and hyperparameter search using TRAINING SUBJECTS ONLY.

    The original chose the winning model by comparing test-set R2
    (line 688). Model choice, feature set and hyperparameters are all decided
    here, inside the fold, and the held-out subject is touched exactly once.
    """
    inner = _grouped_splits(X_tr, y_tr, g_tr, n_inner)

    support = np.ones(X_tr.shape[1], dtype=bool)
    if do_rfecv:
        prep = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())])
        Xp = prep.fit_transform(X_tr)
        sel = RFECV(RandomForestRegressor(n_estimators=100, random_state=RNG, n_jobs=-1),
                    step=1, min_features_to_select=5, cv=inner, scoring='r2', n_jobs=4)
        sel.fit(Xp, y_tr)
        support = sel.support_

    X_sel = X_tr.iloc[:, support]
    search = RandomizedSearchCV(
        _pipeline(_make_model(model_name)), PARAM_DIST[model_name],
        n_iter=n_iter, cv=_grouped_splits(X_sel, y_tr, g_tr, n_inner),
        scoring='r2', n_jobs=4, random_state=RNG)
    search.fit(X_sel, y_tr)
    return search.best_estimator_, support, float(search.best_score_)


# =============================================================================
# Outer evaluation
# =============================================================================

def nested_loso_evaluate(Xf, y, subjects, events, cohort=None,
                         model_name='RandomForest', do_rfecv=True,
                         n_inner=5, n_iter=20, verbose=True):
    """
    Leave-One-Subject-Out outer loop with grouped inner CV.
    """
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    folds, sel_freq = [], np.zeros(Xf.shape[1])

    for k, (tr, te) in enumerate(logo.split(Xf, y, groups=subjects)):
        subj = subjects[te][0]
        model, support, inner_r2 = _fit_fold(
            Xf.iloc[tr], y[tr], subjects[tr], model_name, n_inner, n_iter, do_rfecv)
        sel_freq += support.astype(int)

        pred = model.predict(Xf.iloc[te].iloc[:, support])
        oof[te] = pred

        ey, ep, _ = aggregate_to_events(y[te], pred, events[te])
        m = compute_metrics(ey, ep, label=str(subj))
        m.update({'inner_cv_r2': inner_r2, 'n_windows': int(len(te)),
                  'n_features': int(support.sum()),
                  'cohort': 'T1D' if (cohort is not None and cohort[te][0] == 1) else 'healthy',
                  'glucose_std': float(np.std(y[te]))})
        folds.append(m)

        if verbose:
            print(f"  fold {k:2d} | {str(subj):>8} | {len(te):6d} win -> {m['n']:3d} ev "
                  f"| MAE {m['mae']:6.1f} | MARD {m['mard']:6.2f}% | R2 {m['r2']:7.3f}")

    ey, ep, ee = aggregate_to_events(y, oof, events)
    res = {
        'pooled_event': compute_metrics(ey, ep, 'pooled_event_level'),
        'pooled_window': compute_metrics(y, oof, 'pooled_window_level'),
        'per_subject': pd.DataFrame(folds),
        'by_range': metrics_by_glycemic_range(ey, ep),
        'selection_frequency': pd.Series(sel_freq / max(len(folds), 1),
                                         index=Xf.columns).sort_values(ascending=False),
        'oof': oof, 'event_true': ey, 'event_pred': ep, 'event_id': ee,
    }

    if cohort is not None:
        for nm, flag in (('t1d', 1), ('healthy', 0)):
            m = cohort == flag
            if m.sum():
                a, b, _ = aggregate_to_events(y[m], oof[m], events[m])
                res[f'pooled_event_{nm}'] = compute_metrics(a, b, nm)

    return res


# =============================================================================
# Diagnostic 1 - baseline ladder
# =============================================================================

def baseline_ladder(y, subjects, events, cohort=None):
    """
    b0  global training mean            - the floor
    b1  held-out subject's own mean     - what knowing WHO buys you, without ECG
    b2  cohort mean (T1D vs healthy)    - what `is_diabetic` alone buys

    b1 is the bar. If the ECG model does not clearly beat b1, the ECG is
    contributing nothing and no amount of model tuning will change that.
    """
    logo = LeaveOneGroupOut()
    P = {k: np.full(len(y), np.nan) for k in ('b0', 'b1', 'b2')}

    for tr, te in logo.split(y.reshape(-1, 1), y, groups=subjects):
        P['b0'][te] = y[tr].mean()
        P['b1'][te] = y[te].mean()
        if cohort is not None:
            same = cohort[tr] == cohort[te][0]
            P['b2'][te] = y[tr][same].mean() if same.any() else y[tr].mean()

    names = {'b0': 'b0 global train mean', 'b1': 'b1 subject own mean (oracle)',
             'b2': 'b2 cohort mean'}
    rows = []
    for k, p in P.items():
        if np.isnan(p).all():
            continue
        a, b, _ = aggregate_to_events(y, p, events)
        rows.append(compute_metrics(a, b, names[k]))
    return pd.DataFrame(rows)


# =============================================================================
# Diagnostic 2 - label permutation control
# =============================================================================

def label_permutation_test(Xf, y, subjects, events, n_perm=3, **kw):
    """
    Shuffle glucose BETWEEN events WITHIN each subject. This destroys the
    ECG->glucose relationship by construction while preserving subject
    structure and the windows-per-event replication pattern.

    Under a leaky pipeline this still scores well - that IS the proof.
    Under a correct pipeline R2 must sit at or below zero.
    """
    rng = np.random.default_rng(RNG)
    rows = []

    for i in range(n_perm):
        y_perm = y.copy()
        for s in np.unique(subjects):
            m = subjects == s
            ev = pd.Series(events[m])
            uniq = ev.unique()
            vals = pd.Series(y[m]).groupby(ev.values).first().loc[uniq].values
            y_perm[m] = ev.map(dict(zip(uniq, rng.permutation(vals)))).values

        r = nested_loso_evaluate(Xf, y_perm, subjects, events, verbose=False, **kw)
        row = r['pooled_event']
        row['label'] = f'permutation_{i}'
        rows.append(row)
        print(f"  permutation {i}: R2 {row['r2']:+.4f}  MARD {row['mard']:.2f}%")

    return pd.DataFrame(rows)


# =============================================================================
# Diagnostic 3 - split severity ablation
# =============================================================================

def split_severity_ablation(Xf, y, subjects, events, n_estimators=300):
    """
    Same model, same features, four protocols of increasing rigour.

    This table IS a contribution. No paper in the non-invasive glucose
    literature reports it, and many of them report window-level splits.
    """
    idx = np.arange(len(y))
    rows = []

    def _run(name, splits, event_level):
        pred = np.full(len(y), np.nan)
        for tr, te in splits:
            m = _pipeline(RandomForestRegressor(n_estimators=n_estimators,
                                                random_state=RNG, n_jobs=-1))
            m.fit(Xf.iloc[tr], y[tr])
            pred[te] = m.predict(Xf.iloc[te])
        ok = ~np.isnan(pred)
        if event_level:
            a, b, _ = aggregate_to_events(y[ok], pred[ok], events[ok])
        else:
            a, b = y[ok], pred[ok]
        r = compute_metrics(a, b, name)
        rows.append(r)
        print(f"  {name:<44} n={r['n']:>7,}  MAE {r['mae']:6.1f}  "
              f"MARD {r['mard']:6.2f}%  R2 {r['r2']:+.4f}")

    print("\n  protocol                                       n        MAE     MARD      R2")
    print("  " + "-" * 76)
    _run("1. random window split (as submitted)",
         ShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG).split(idx), False)
    _run("2. glucose-event-grouped split",
         GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
         .split(idx, y, groups=events), False)
    _run("3. subject-grouped split (LOSO)",
         LeaveOneGroupOut().split(idx.reshape(-1, 1), y, groups=subjects), False)
    _run("4. subject-grouped + event-level metrics",
         LeaveOneGroupOut().split(idx.reshape(-1, 1), y, groups=subjects), True)

    return pd.DataFrame(rows)


# =============================================================================
# Lag sweep with common support
# =============================================================================

def lag_sweep(pkl_paths, cap=None, **kw):

    loaded = []
    for p in sorted(pkl_paths):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        loaded.append((d['meta']['tau_minutes'], d, p))

    common = set(loaded[0][1]['glucose_events'])
    for _, d, _ in loaded[1:]:
        common &= set(d['glucose_events'])
    print(f"\n  common support: {len(common)} glucose events retained in all "
          f"{len(loaded)} tau arms")

    # Per-arm event counts and pairwise overlap: this IS the identifiability
    # result when common support is empty or tiny. Report it instead of
    # forcing a LOSO that cannot run.
    taus = [t for t, _, _ in loaded]
    ev_sets = {t: set(d['glucose_events']) for t, d, _ in loaded}
    per_arm = pd.DataFrame({
        'tau_min': taus,
        'n_events': [len(ev_sets[t]) for t in taus],
        'n_windows': [len(d['glucose_events']) for _, d, _ in loaded],
    }).sort_values('tau_min')
    print("\n  per-arm event counts:")
    print(per_arm.to_string(index=False))

    overlap = pd.DataFrame(index=taus, columns=taus, dtype=int)
    for a in taus:
        for b in taus:
            overlap.loc[a, b] = len(ev_sets[a] & ev_sets[b])
    print("\n  pairwise shared-event matrix (rows/cols = tau):")
    print(overlap.to_string())

    MIN_EVENTS = 15  # need at least this many shared events to attempt grouped LOSO
    if len(common) < MIN_EVENTS:
        print(f"\n  RESULT: common support ({len(common)}) < {MIN_EVENTS}. "
              "Cross-lag comparison is ill-posed on this dataset: different lag "
              "settings align to disjoint reference measurements. Reporting the "
              "identifiability finding rather than an optimum.")
        summary = per_arm.copy()
        summary['common_support'] = len(common)
        summary['mard'] = np.nan
        summary['mae'] = np.nan
        summary['r2'] = np.nan
        summary.attrs['pairwise_overlap'] = overlap
        return summary

    if len(common) < 100:
        print("  WARNING: common support is small. This sweep is underpowered; "
              "report it as such rather than declaring an optimum.")

    rows = []
    for tau, d, path in loaded:
        keep = np.isin(d['glucose_events'], list(common))
        X, y = d['X'][keep], d['y'][keep]
        subj, ev = d['subjects'][keep], d['glucose_events'][keep]
        if cap:
            X, y, subj, ev, _ = subsample_per_event(X, y, subj, ev, None, cap)

        Xf = extract_ecg_features(X, verbose=False)
        r = nested_loso_evaluate(Xf, y, subj, ev, verbose=False, **kw)
        m = r['pooled_event']
        m['tau_min'] = tau
        m['per_subject_argmin'] = None
        rows.append(m)
        print(f"  tau={tau:+5.1f} min | MARD {m['mard']:6.2f}% | "
              f"MAE {m['mae']:6.1f} | R2 {m['r2']:+.4f}")

    df = pd.DataFrame(rows).sort_values('tau_min')

    print("\n  Per-subject optimal tau (answers reviewer comment 9 directly):")
    per_subj = {}
    for tau, d, _ in loaded:
        keep = np.isin(d['glucose_events'], list(common))
        for s in np.unique(d['subjects'][keep]):
            per_subj.setdefault(s, {})[tau] = None
    print("  -> run nested_loso_evaluate per tau and take each subject's argmin "
          "MARD from res['per_subject']; report the distribution, not the mode.")

    return df


# =============================================================================
# Utilities
# =============================================================================

def subsample_per_event(X, y, subjects, events, cohort, cap, seed=RNG):

    rng = np.random.default_rng(seed)
    keep = []
    for e in np.unique(events):
        idx = np.flatnonzero(events == e)
        keep.extend(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    keep = np.sort(np.asarray(keep))
    return (X[keep], y[keep], subjects[keep], events[keep],
            cohort[keep] if cohort is not None else None)


def load_dataset(path, cap=None):
    with open(path, 'rb') as f:
        d = pickle.load(f)

    for key in ('subjects', 'glucose_events'):
        if key not in d:
            raise KeyError(f"'{key}' missing from {path}. Re-run the rewritten "
                           f"Phase 1 - the original does not persist identifiers.")

    X, y = d['X'], np.asarray(d['y'], float)
    subj, ev = np.asarray(d['subjects']), np.asarray(d['glucose_events'])
    coh = np.asarray(d.get('cohort', np.zeros(len(y), int)))

    print(f"loaded {path}")
    print(f"  {len(y):,} windows | {len(np.unique(ev)):,} events | "
          f"{len(np.unique(subj))} subjects | "
          f"{len(y)/len(np.unique(ev)):.1f} windows/event")

    if cap:
        X, y, subj, ev, coh = subsample_per_event(X, y, subj, ev, coh, cap)
        print(f"  after cap={cap}: {len(y):,} windows")

    return X, y, subj, ev, coh, d.get('meta', {})


def _dump(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=float)
    print(f"  wrote {path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='preprocessed_data_tau5.pkl')
    ap.add_argument('--mode', default='ablation',
                    choices=['ablation', 'loso', 'permutation', 'lagsweep', 'all'])
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--cap', type=int, default=20,
                    help='max windows retained per glucose event (0 = no cap)')
    ap.add_argument('--model', default='RandomForest', choices=['RandomForest', 'XGBoost'])
    ap.add_argument('--no-rfecv', action='store_true')
    ap.add_argument('--n-iter', type=int, default=20)
    ap.add_argument('--n-perm', type=int, default=3)
    ap.add_argument('--lag-glob', default='preprocessed_data_tau*.pkl')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    kw = dict(model_name=args.model, do_rfecv=not args.no_rfecv, n_iter=args.n_iter)

    if args.mode == 'lagsweep':
        paths = globmod.glob(args.lag_glob)
        if len(paths) < 2:
            raise SystemExit(f"need >=2 tau files matching {args.lag_glob}")
        _dump(lag_sweep(paths, cap=args.cap or None, **kw),
              os.path.join(args.out_dir, 'lag_sweep.csv'))
        return

    X, y, subj, ev, coh, meta = load_dataset(args.data, args.cap or None)

    print("\nextracting features...")
    Xf = extract_ecg_features(X)
    print(f"  {Xf.shape[1]} features x {Xf.shape[0]:,} windows")
    del X

    run = (args.mode, 'all')

    if args.mode in ('ablation', 'all'):
        print("\n=== SPLIT SEVERITY ABLATION ===")
        _dump(split_severity_ablation(Xf, y, subj, ev),
              os.path.join(args.out_dir, 'split_severity_ablation.csv'))

    if args.mode in ('loso', 'all'):
        print("\n=== BASELINE LADDER ===")
        bl = baseline_ladder(y, subj, ev, coh)
        print(bl[['label', 'n', 'mae', 'mard', 'r2']].to_string(index=False))
        _dump(bl, os.path.join(args.out_dir, 'baseline_ladder.csv'))

        print("\n=== NESTED LOSO ===")
        res = nested_loso_evaluate(Xf, y, subj, ev, cohort=coh, **kw)

        print("\n  pooled (event level) :", {k: round(v, 3) for k, v in res['pooled_event'].items()
                                             if isinstance(v, float)})
        print("  pooled (window level):", {k: round(v, 3) for k, v in res['pooled_window'].items()
                                           if isinstance(v, float)})
        for k in ('pooled_event_t1d', 'pooled_event_healthy'):
            if k in res:
                print(f"  {k:<21}:", {kk: round(vv, 3) for kk, vv in res[k].items()
                                      if isinstance(vv, float)})

        print("\n  by glycemic range:")
        print(res['by_range'].to_string(index=False))

        _dump(res['per_subject'], os.path.join(args.out_dir, 'per_subject.csv'))
        _dump(res['by_range'], os.path.join(args.out_dir, 'by_glycemic_range.csv'))
        _dump(res['selection_frequency'].to_frame('selection_frequency').reset_index(),
              os.path.join(args.out_dir, 'feature_selection_frequency.csv'))
        _dump({'pooled_event': res['pooled_event'], 'pooled_window': res['pooled_window'],
               'meta': meta}, os.path.join(args.out_dir, 'pooled_metrics.json'))
        np.savez(os.path.join(args.out_dir, 'event_predictions.npz'),
                 event_id=res['event_id'], y_true=res['event_true'], y_pred=res['event_pred'])

    if args.mode in ('permutation', 'all'):
        print("\n=== LABEL PERMUTATION CONTROL ===")
        _dump(label_permutation_test(Xf, y, subj, ev, n_perm=args.n_perm, **kw),
              os.path.join(args.out_dir, 'label_permutation.csv'))

    print(f"\nelapsed {(time.time() - t0)/60:.1f} min")


if __name__ == '__main__':
    main()
