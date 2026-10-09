#!/usr/bin/env python3
"""
PHASE 1 for PhysioCGM - ECG/CGM preprocessing and alignment.

Emits the SAME pickle contract as the D1NAMO phase1_preprocess.py, so the
existing phase2_evaluate.py and phase3_figures.py run on it unchanged:
    X, y, subjects, sessions, glucose_events, cohort, meta

Key differences from D1NAMO, all handled here:
  - ECG column is 'EcgWaveform'; timestamp is a real wall-clock column
    ('%d/%m/%Y %H:%M:%S.%f'), not filename-encoded.
  - Glucose is a Dexcom Clarity export: keep only Event Type == 'EGV',
    glucose already in mg/dL (NO conversion factor).
  - CGM is dense (~5 min). This is a leakage minefield: ~10^4-10^5 ECG windows
    exist per CGM reading. The CGM reading is the event unit; all its windows
    stay together (never split across train/test). We also CAP windows per
    reading, or feature extraction is intractable.
  - Subject IDs: cXsYY folders. Cohort from the leading c1/c2 group.

Usage
-----
    python phase1_physiocgm.py --tau 5 --window-width 10 \
        --windows-per-reading 4 --base-path /workspace/datasets/PhysioCGM
"""

import argparse
import glob
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

FS = 250                       # confirmed: ~4 ms sample step
WINDOW_SECONDS = 30
STEP_SECONDS = 15
ECG_COL = 'EcgWaveform'
ECG_TIME_FMT = '%d/%m/%Y %H:%M:%S.%f'   # day-first, e.g. 25/06/2022 10:12:02.441


# =============================================================================
# Discovery
# =============================================================================

def discover_subjects(base_path):
    subs = sorted(d for d in os.listdir(base_path)
                  if d.endswith('_raw') and os.path.isdir(os.path.join(base_path, d)))
    out = []
    for d in subs:
        sid = d.replace('_raw', '')            # c1s01
        zephyr = os.path.join(base_path, d, 'zephyr')
        cgm = os.path.join(base_path, d, 'cgm.csv')
        if os.path.isdir(zephyr) and os.path.exists(cgm):
            cohort = 1 if sid.startswith('c1') else 0   # c1 vs c2 groups
            out.append((sid, d, cohort))
    print(f"  {len(out)} subjects: {[s for s, _, _ in out]}")
    return out


# =============================================================================
# CGM parsing (Dexcom Clarity export)
# =============================================================================

def load_cgm(cgm_path):
    """Keep only EGV rows; glucose already mg/dL. Returns sorted df[timestamp, glucose]."""
    df = pd.read_csv(cgm_path, low_memory=False)

    ev_col = next((c for c in df.columns if c.strip().lower() == 'event type'), None)
    ts_col = next((c for c in df.columns if c.strip().lower().startswith('timestamp')), None)
    g_col = next((c for c in df.columns if 'glucose value' in c.strip().lower()), None)
    if not all([ev_col, ts_col, g_col]):
        raise ValueError(f"Unexpected CGM columns in {cgm_path}: {list(df.columns)}")

    egv = df[df[ev_col].astype(str).str.strip() == 'EGV'].copy()
    egv['timestamp'] = pd.to_datetime(egv[ts_col], errors='coerce')
    egv['glucose'] = pd.to_numeric(egv[g_col], errors='coerce')  # already mg/dL
    egv = egv.dropna(subset=['timestamp', 'glucose']).sort_values('timestamp')
    egv = egv.reset_index(drop=True)
    egv['reading_idx'] = egv.index          # stable per-reading id
    return egv[['timestamp', 'glucose', 'reading_idx']]


# =============================================================================
# ECG preprocessing (identical filter chain to D1NAMO; per-recording z-norm)
# =============================================================================

def preprocess_ecg_signal(x, fs=FS, use_wavelet=True):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if use_wavelet:
        try:
            import pywt
            coeffs = pywt.wavedec(x, 'db6', level=8)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i]) * 0.1, mode='soft')
            xd = pywt.waverec(coeffs, 'db6')
            x = xd[:len(x)] if len(xd) > len(x) else np.pad(xd, (0, len(x) - len(xd)))
        except Exception:
            pass
    nyq = 0.5 * fs
    b, a = signal.butter(3, [0.5 / nyq, 40 / nyq], btype='band')
    x = signal.filtfilt(b, a, x)
    b2, a2 = signal.butter(2, 0.8 / nyq, btype='high')
    x = signal.filtfilt(b2, a2, x)
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 0 else x - np.mean(x)


def _read_ecg_session(path):
    """Return (raw_values float array, timestamps DatetimeIndex) for one ECG csv."""
    df = pd.read_csv(path)
    col = ECG_COL if ECG_COL in df.columns else df.columns[-1]
    tcol = 'Time' if 'Time' in df.columns else df.columns[0]
    ts = pd.to_datetime(df[tcol], format=ECG_TIME_FMT, errors='coerce')
    vals = pd.to_numeric(df[col], errors='coerce').values
    ok = ts.notna().values & np.isfinite(vals)
    return vals[ok], ts[ok]


# =============================================================================
# Alignment  -  CGM reading is the event unit; windows per reading are capped
# =============================================================================

def align_session(raw_vals, ts, cgm_df, subject_id, session_id,
                  tau=5.0, window_width=10.0, windows_per_reading=4,
                  window_seconds=WINDOW_SECONDS, fs=FS, min_std=0.01, rng=None):
    """
    For each CGM reading whose (t - tau) falls inside this ECG session, extract up
    to `windows_per_reading` ECG windows ending at/around that time, all tagged
    with the SAME glucose_event id. This makes the reading the independent unit
    and prevents autocorrelated windows from splitting across train/test.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    win_n = int(window_seconds * fs)
    if len(raw_vals) < win_n:
        return rows

    t0, t1 = ts.iloc[0], ts.iloc[-1]
    half = pd.Timedelta(minutes=window_width / 2.0)
    centre_shift = pd.Timedelta(minutes=tau)

    # candidate CGM readings whose ECG-anchor time lies in this session
    for _, g in cgm_df.iterrows():
        # ECG window should PRECEDE glucose by tau (ECG now -> glucose in tau min)
        anchor = g['timestamp'] - centre_shift          # ideal ECG time
        lo, hi = anchor - half, anchor + half
        if hi < t0 or lo > t1:
            continue

        # find ECG indices within [lo, hi]
        mask = (ts >= lo) & (ts <= hi)
        idx = np.flatnonzero(mask.values)
        if len(idx) < win_n:
            continue

        # sample up to N non-overlapping window start positions within the mask
        starts = np.arange(idx[0], idx[-1] - win_n + 1, win_n)
        if len(starts) == 0:
            continue
        if len(starts) > windows_per_reading:
            starts = rng.choice(starts, windows_per_reading, replace=False)

        event_id = f"{subject_id}|r{int(g['reading_idx'])}"
        for j in starts:
            seg = raw_vals[j:j + win_n]
            if len(seg) < win_n or np.std(seg) <= min_std:
                continue
            proc = preprocess_ecg_signal(seg)
            if np.std(proc) <= min_std:
                continue
            rows.append({
                'ecg_segment': proc.astype(np.float32),
                'glucose_value': float(g['glucose']),
                'timestamp': ts.iloc[j],
                'subject': subject_id,
                'session': session_id,
                'glucose_event': event_id,
                'reading_time': g['timestamp'],
            })
    return rows


# =============================================================================
# Per-subject
# =============================================================================

def process_subject(base_path, folder, sid, cohort, tau, window_width,
                    windows_per_reading, max_sessions=None, rng=None):
    cgm = load_cgm(os.path.join(base_path, folder, 'cgm.csv'))
    zephyr = os.path.join(base_path, folder, 'zephyr')
    sessions = sorted(glob.glob(os.path.join(zephyr, '*', '*_ECG.csv')))
    if max_sessions:
        sessions = sessions[:max_sessions]

    all_rows = []
    for spath in sessions:
        try:
            vals, ts = _read_ecg_session(spath)
            if len(vals) < WINDOW_SECONDS * FS:
                continue
            session_id = f"{sid}|{os.path.basename(os.path.dirname(spath))}"
            all_rows.extend(align_session(
                vals, ts, cgm, sid, session_id, tau=tau,
                window_width=window_width, windows_per_reading=windows_per_reading,
                rng=rng))
        except Exception as e:
            print(f"    {sid}/{os.path.basename(spath)}: {e}")

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    df['cohort'] = cohort
    return df


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-path', default='/workspace/datasets/PhysioCGM')
    ap.add_argument('--tau', type=float, default=5.0)
    ap.add_argument('--window-width', type=float, default=10.0)
    ap.add_argument('--windows-per-reading', type=int, default=4,
                    help='cap ECG windows per CGM reading (controls leakage AND runtime)')
    ap.add_argument('--max-sessions', type=int, default=0,
                    help='limit ECG sessions per subject for a fast trial run (0=all)')
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    t0 = datetime.now()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    subs = discover_subjects(args.base_path)
    if not subs:
        raise SystemExit("No subjects found.")

    X, y, subjects, sessions, events, cohort = [], [], [], [], [], []
    for sid, folder, coh in tqdm(subs, desc="subjects"):
        df = process_subject(args.base_path, folder, sid, coh, args.tau,
                              args.window_width, args.windows_per_reading,
                              max_sessions=args.max_sessions or None, rng=rng)
        if df is None:
            print(f"  {sid}: no aligned windows")
            continue
        X.extend(df['ecg_segment'].tolist())
        y.extend(df['glucose_value'].tolist())
        subjects.extend(df['subject'].tolist())
        sessions.extend(df['session'].tolist())
        events.extend(df['glucose_event'].tolist())
        cohort.extend(df['cohort'].tolist())
        print(f"  {sid}: {len(df)} windows / {df['glucose_event'].nunique()} readings")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    subjects = np.asarray(subjects); sessions = np.asarray(sessions)
    events = np.asarray(events); cohort = np.asarray(cohort, dtype=int)

    n_ev = len(np.unique(events))
    print(f"\n{len(y):,} windows | {n_ev:,} CGM-reading events | "
          f"{len(np.unique(subjects))} subjects | "
          f"{len(y)/max(n_ev,1):.1f} windows/event")

    tag = f"tau{'m' if args.tau < 0 else 'p'}{abs(int(args.tau))}"
    out = os.path.join(args.out_dir, f'physiocgm_{tag}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({
            'X': X, 'y': y, 'subjects': subjects, 'sessions': sessions,
            'glucose_events': events, 'cohort': cohort,
            'meta': {
                'dataset': 'PhysioCGM', 'tau_minutes': args.tau,
                'window_width_minutes': args.window_width,
                'windows_per_reading': args.windows_per_reading,
                'window_seconds': WINDOW_SECONDS, 'fs': FS,
                'glucose_units': 'mg/dL', 'conversion_factor': 1.0,
                'n_windows': int(len(y)), 'n_events': int(n_ev),
                'n_subjects': int(len(np.unique(subjects))),
                'created': t0.isoformat(),
            },
        }, f, protocol=4)
    print(f"wrote {out}  (elapsed {datetime.now() - t0})")


if __name__ == '__main__':
    main()
