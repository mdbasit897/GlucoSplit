#!/usr/bin/env python3
"""
PHASE 1 - D1NAMO ECG/glucose preprocessing and alignment.

Produces raw ECG windows, glucose labels, and - critically - the grouping
identifiers that make honest evaluation possible downstream.

Usage
-----
    python phase1_preprocess.py --tau 5 --window-width 10
    python phase1_preprocess.py --tau -5 --window-width 10   # falsification arm

Output
------
    preprocessed_data_tau{TAU}.pkl  containing:
        X               (n_windows, 7500) float32   raw preprocessed ECG windows
        y               (n_windows,)      float64   glucose, mg/dL
        subjects        (n_windows,)      str       LOSO grouping variable
        sessions        (n_windows,)      str       ECG recording ID
        glucose_events  (n_windows,)      str       INDEPENDENT UNIT OF ANALYSIS
        cohort          (n_windows,)      int       1 = T1D, 0 = healthy (metadata only)
        meta            dict                        provenance and config
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

FS = 250                      # Zephyr BioHarness 3 ECG sampling rate
WINDOW_SECONDS = 30           # HRV analysis guideline minimum
STEP_SECONDS = 15             # 50% overlap
MGDL_PER_MMOL = 18.0182       # single canonical constant


# =============================================================================
# Dataset discovery
# =============================================================================

def _paths(base_path, is_diabetic):
    if is_diabetic:
        return (os.path.join(base_path, 'diabetes_subset_ecg_data/diabetes_subset_ecg_data'),
                os.path.join(base_path, 'diabetes_subset_pictures-glucose-food-insulin/'
                                        'diabetes_subset_pictures-glucose-food-insulin'))
    return (os.path.join(base_path, 'healthy_subset_ecg_data/healthy_subset_ecg_data'),
            os.path.join(base_path, 'healthy_subset_pictures-glucose-food/'
                                    'healthy_subset_pictures-glucose-food'))


def discover_subjects(base_path, is_diabetic=True):
    """Subjects possessing BOTH ECG sensor data and a glucose.csv."""
    ecg_path, glucose_path = _paths(base_path, is_diabetic)

    if not (os.path.exists(ecg_path) and os.path.exists(glucose_path)):
        print(f"  missing path: {ecg_path if not os.path.exists(ecg_path) else glucose_path}")
        return []

    ecg_subjects = {d for d in os.listdir(ecg_path)
                    if os.path.isdir(os.path.join(ecg_path, d))
                    and os.path.exists(os.path.join(ecg_path, d, 'sensor_data'))}
    glu_subjects = {d for d in os.listdir(glucose_path)
                    if os.path.isdir(os.path.join(glucose_path, d))
                    and os.path.exists(os.path.join(glucose_path, d, 'glucose.csv'))}

    valid = sorted(ecg_subjects & glu_subjects)
    label = 'T1D' if is_diabetic else 'healthy'
    print(f"  {label}: {len(valid)} subjects with both modalities -> {valid}")
    return valid


# =============================================================================
# Signal preprocessing
# =============================================================================

def preprocess_ecg_signal(ecg_signal, fs=FS, use_wavelet=True):
    """
    Wavelet denoise -> bandpass 0.5-40 Hz -> baseline removal -> z-normalise.

    NOTE ON NORMALISATION: z-scoring is applied per RECORDING, not per
    window. This is defensible (a recording never straddles a subject boundary,
    so no train/test statistics cross) but it does mean each window's scaling
    uses information from later in the same recording. This must be stated
    explicitly in the manuscript's preprocessing subsection. If a reviewer
    objects, switch to per-window normalisation by moving these four lines into
    the windowing loop - the cost is losing recording-level amplitude context.
    """
    x = np.nan_to_num(np.asarray(ecg_signal, dtype=float))

    if use_wavelet:
        try:
            import pywt
            coeffs = pywt.wavedec(x, 'db6', level=8)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i]) * 0.1, mode='soft')
            x_dn = pywt.waverec(coeffs, 'db6')
            x = x_dn[:len(x)] if len(x_dn) > len(x) else np.pad(x_dn, (0, len(x) - len(x_dn)))
        except Exception as e:
            print(f"  wavelet denoising skipped: {e}")

    nyq = 0.5 * fs
    b, a = signal.butter(3, [0.5 / nyq, 40 / nyq], btype='band')
    x = signal.filtfilt(b, a, x)

    b2, a2 = signal.butter(2, 0.8 / nyq, btype='high')
    x = signal.filtfilt(b2, a2, x)

    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 0 else x - np.mean(x)


# =============================================================================
# Temporal alignment
# =============================================================================

def align_ecg_with_glucose(ecg_segments, timestamps, glucose_df, subject_id,
                           tau=5.0, window_width=10.0, max_abs_lag=30.0,
                           window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS,
                           min_std=0.01, fs=FS):
    rows = []
    win_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)
    half_w = window_width / 2.0

    for ecg, ts_str in zip(ecg_segments, timestamps):
        try:
            parts = ts_str.split('-')
            t_ecg = pd.to_datetime(f"{parts[0].replace('_', '-')} {parts[1].replace('_', ':')}")
        except Exception:
            continue

        session_id = f"{subject_id}|{ts_str}"
        centre = t_ecg + pd.Timedelta(minutes=tau)
        lo = centre - pd.Timedelta(minutes=half_w)
        hi = centre + pd.Timedelta(minutes=half_w)

        mask = (glucose_df['timestamp'] >= lo) & (glucose_df['timestamp'] <= hi)
        if mask.sum() == 0:
            continue

        cand = glucose_df[mask]
        g_idx = (cand['timestamp'] - centre).abs().idxmin()
        g_row = glucose_df.loc[g_idx]

        signed_lag = (g_row['timestamp'] - t_ecg).total_seconds() / 60.0
        if abs(signed_lag) > max_abs_lag:
            continue

        event_id = f"{subject_id}|g{int(g_idx)}"

        if len(ecg) >= win_samples:
            starts = range(0, len(ecg) - win_samples + 1, step_samples)
        else:
            starts = [None]

        for w, j in enumerate(starts):
            if j is None:
                seg = np.pad(ecg, (0, win_samples - len(ecg)), 'constant')
                t_win = t_ecg
            else:
                seg = ecg[j:j + win_samples]
                t_win = t_ecg + pd.Timedelta(seconds=j / fs)

            if np.std(seg) <= min_std:
                continue

            rows.append({
                'ecg_segment': seg.astype(np.float32),
                'glucose_value': float(g_row['glucose']),
                'timestamp': t_win,
                'subject': subject_id,
                'session': session_id,
                'glucose_event': event_id,
                'window_index': w,
                'signed_lag_minutes': signed_lag,
            })

    return pd.DataFrame(rows) if rows else None


# =============================================================================
# Per-subject processing
# =============================================================================

def _load_glucose(path):
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)

    if date_col and time_col:
        df['timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
    else:
        col = date_col or time_col or df.columns[0]
        df['timestamp'] = pd.to_datetime(df[col])

    gcol = next((c for c in df.columns if 'glucose' in c.lower()),
                df.columns[1] if len(df.columns) > 1 else df.columns[0])

    df['glucose'] = pd.to_numeric(df[gcol], errors='coerce') * MGDL_PER_MMOL
    df = df.dropna(subset=['glucose', 'timestamp']).sort_values('timestamp')
    return df.reset_index(drop=True)


def preprocess_ecg_data(base_path, subject_ids, is_diabetic, tau, window_width,
                        prefix, debug=False):
    ecg_root, glu_root = _paths(base_path, is_diabetic)
    out = {}

    for sid in tqdm(subject_ids, desc=f"{'T1D' if is_diabetic else 'healthy'} subjects"):
        tagged = f"{prefix}{sid}"
        sensor_dir = os.path.join(ecg_root, sid, 'sensor_data')
        glucose_csv = os.path.join(glu_root, sid, 'glucose.csv')

        if not os.path.exists(sensor_dir) or not os.path.exists(glucose_csv):
            continue

        try:
            glucose_df = _load_glucose(glucose_csv)
        except Exception as e:
            print(f"  {tagged}: glucose load failed - {e}")
            continue

        segments, stamps = [], []
        for sess in sorted(os.listdir(sensor_dir)):
            sess_dir = os.path.join(sensor_dir, sess)
            if not os.path.isdir(sess_dir):
                continue
            files = glob.glob(os.path.join(sess_dir, '*_ECG.csv'))
            if not files:
                continue
            try:
                raw_df = pd.read_csv(files[0])
                col = next((c for c in raw_df.columns
                            if any(k in c.lower() for k in ('ecg', 'waveform', 'value'))),
                           raw_df.columns[min(1, len(raw_df.columns) - 1)])
                segments.append(preprocess_ecg_signal(raw_df[col].values))
                stamps.append(os.path.basename(files[0]).split('_ECG.csv')[0])
            except Exception as e:
                print(f"  {tagged}/{sess}: ECG read failed - {e}")

        if not segments:
            continue

        aligned = align_ecg_with_glucose(segments, stamps, glucose_df, tagged,
                                         tau=tau, window_width=window_width)
        if aligned is None or len(aligned) == 0:
            print(f"  {tagged}: no windows survived alignment at tau={tau}")
            continue

        out[tagged] = {'aligned_data': aligned, 'is_diabetic': is_diabetic,
                       'n_glucose_readings': len(glucose_df)}

        if debug:
            n_ev = aligned['glucose_event'].nunique()
            print(f"  {tagged}: {len(aligned)} windows / {n_ev} events "
                  f"({len(aligned) / n_ev:.0f} per event)")

    return out


# =============================================================================
# Combination and dataset characterisation
# =============================================================================

def combine_all_subject_data(all_subjects_data):

    X, y, subjects, sessions, events, cohort = [], [], [], [], [], []

    for sid, sd in all_subjects_data.items():
        ad = sd['aligned_data']
        X.extend(ad['ecg_segment'].tolist())
        y.extend(ad['glucose_value'].tolist())
        subjects.extend(ad['subject'].tolist())
        sessions.extend(ad['session'].tolist())
        events.extend(ad['glucose_event'].tolist())
        cohort.extend([1 if sd['is_diabetic'] else 0] * len(ad))

    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=float),
            np.asarray(subjects), np.asarray(sessions),
            np.asarray(events), np.asarray(cohort, dtype=int))


def characterise(y, subjects, events, cohort):

    n_windows = len(y)
    n_events = len(np.unique(events))
    n_subjects = len(np.unique(subjects))

    per_subject = []
    for s in np.unique(subjects):
        m = subjects == s
        ev = np.unique(events[m])
        per_subject.append({
            'subject': s,
            'cohort': 'T1D' if cohort[m][0] == 1 else 'healthy',
            'n_windows': int(m.sum()),
            'n_events': int(len(ev)),
            'windows_per_event': float(m.sum() / len(ev)),
            'glucose_mean': float(np.mean(y[m])),
            'glucose_std': float(np.std(y[m])),
            'glucose_min': float(np.min(y[m])),
            'glucose_max': float(np.max(y[m])),
        })

    df = pd.DataFrame(per_subject).sort_values('subject')

    print("\n" + "=" * 78)
    print("REPLICATION STRUCTURE  (this is the table reviewers will want)")
    print("=" * 78)
    print(df.to_string(index=False))
    print("-" * 78)
    print(f"windows            : {n_windows:,}")
    print(f"glucose events     : {n_events:,}   <- TRUE SAMPLE SIZE")
    print(f"subjects           : {n_subjects}")
    print(f"replication factor : {n_windows / n_events:.1f} windows per event")
    print(f"\nReporting metrics over {n_windows:,} windows instead of {n_events:,}")
    print(f"events inflates the effective n by {n_windows / n_events:.0f}x.")
    print("=" * 78 + "\n")

    return df


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-path', default='/workspace/datasets/D1NAMO/versions/1')
    ap.add_argument('--tau', type=float, default=5.0,
                    help='lag window CENTRE in minutes; negative permitted')
    ap.add_argument('--window-width', type=float, default=10.0,
                    help='lag window WIDTH in minutes; hold constant across a sweep')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    t0 = datetime.now()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nPhase 1 | tau = {args.tau:+.1f} min | width = {args.window_width:.1f} min")
    print(f"lag window = [{args.tau - args.window_width/2:+.1f}, "
          f"{args.tau + args.window_width/2:+.1f}] minutes relative to ECG\n")

    diabetic = discover_subjects(args.base_path, is_diabetic=True)
    healthy = discover_subjects(args.base_path, is_diabetic=False)
    if not diabetic and not healthy:
        raise SystemExit("No subjects found. Check --base-path.")

    all_data = {}
    all_data.update(preprocess_ecg_data(args.base_path, diabetic, True,
                                        args.tau, args.window_width, 'D_', args.debug))
    all_data.update(preprocess_ecg_data(args.base_path, healthy, False,
                                        args.tau, args.window_width, 'H_', args.debug))

    if not all_data:
        raise SystemExit(f"No windows survived alignment at tau={args.tau}.")

    X, y, subjects, sessions, events, cohort = combine_all_subject_data(all_data)
    summary = characterise(y, subjects, events, cohort)

    tag = f"tau{int(args.tau):+d}".replace('+', 'p').replace('-', 'm')
    pkl_path = os.path.join(args.out_dir, f'preprocessed_data_{tag}.pkl')

    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'X': X, 'y': y,
            'subjects': subjects,
            'sessions': sessions,
            'glucose_events': events,
            'cohort': cohort,
            'meta': {
                'tau_minutes': args.tau,
                'window_width_minutes': args.window_width,
                'window_seconds': WINDOW_SECONDS,
                'step_seconds': STEP_SECONDS,
                'fs': FS,
                'glucose_units': 'mg/dL',
                'conversion_factor': MGDL_PER_MMOL,
                'n_windows': int(len(y)),
                'n_events': int(len(np.unique(events))),
                'n_subjects': int(len(np.unique(subjects))),
                'created': t0.isoformat(),
            },
        }, f, protocol=4)

    summary.to_csv(os.path.join(args.out_dir, f'replication_structure_{tag}.csv'), index=False)
    with open(os.path.join(args.out_dir, f'phase1_meta_{tag}.json'), 'w') as f:
        json.dump({'per_subject': summary.to_dict('records')}, f, indent=2)

    print(f"wrote {pkl_path}")
    print(f"elapsed {datetime.now() - t0}")


if __name__ == '__main__':
    main()
