#!/usr/bin/env python3
"""
Compute ONLY the tau = +5 arm of the PhysioCGM lag sweep.

Why this exists
---------------
The six arms {-10,-5,0,+10,+15,+20} are already in lag_sweep.csv, scored on the
32,525-event support common to those six. Re-running all seven to add tau=+5 is
wasted compute: the seven-arm intersection is 32,524 events, one fewer, which is
immaterial at reporting precision.

This script therefore:
  PASS 1  loads each tau pickle ONE AT A TIME, keeps only the glucose-event ids,
          and computes the seven-arm intersection.  Peak RAM ~5 GB, not ~31 GB.
          The id set is cached, so a re-run skips this pass.
  PASS 2  loads ONLY physiocgm_taup5.pkl, restricts to that intersection, and
          runs the same nested LOSO used for the other six arms.
  PASS 3  appends the tau=+5 row to lag_sweep.csv, sorted by tau.

Settings are matched to the existing six arms: --cap 20, no RFECV, n_iter 5.

Usage
-----
    cd /workspace/projects/ECG-Glucose-Prediction_v2_revision/src/PhysioCGM
    python3 single_arm_taup5.py
"""
import argparse, gc, glob, json, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_evaluate import (extract_ecg_features, nested_loso_evaluate,
                             subsample_per_event)


def common_support(lag_glob, cache):
    if os.path.exists(cache):
        ids = set(json.load(open(cache)))
        print(f"  cached common support: {len(ids):,} events  ({cache})")
        return ids
    paths = sorted(glob.glob(lag_glob))
    if len(paths) < 2:
        raise SystemExit(f"need >=2 tau files matching {lag_glob}; found {len(paths)}")
    print(f"  computing common support over {len(paths)} arms (one pickle at a time)")
    ids = None
    for p in paths:
        t0 = time.time()
        with open(p, 'rb') as f:
            d = pickle.load(f)
        ev = set(np.asarray(d['glucose_events']).tolist())
        ids = ev if ids is None else (ids & ev)
        print(f"    {os.path.basename(p):26s} {len(ev):>7,} events | "
              f"running intersection {len(ids):>7,} | {time.time()-t0:5.1f}s")
        del d, ev
        gc.collect()
    json.dump(sorted(ids), open(cache, 'w'))
    print(f"  common support: {len(ids):,} events  -> cached to {cache}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='PhysioCGM/physiocgm_taup5.pkl')
    ap.add_argument('--lag-glob', default='PhysioCGM/physiocgm_tau*.pkl')
    ap.add_argument('--csv', default='PhysioCGM/results/lag_sweep.csv')
    ap.add_argument('--cache', default='PhysioCGM/results/common_support_7arm.json')
    ap.add_argument('--cap', type=int, default=20)
    ap.add_argument('--n-iter', type=int, default=5)
    ap.add_argument('--tau', type=float, default=5.0)
    args = ap.parse_args()
    t0 = time.time()

    print("PASS 1 - common support")
    keep_ids = common_support(args.lag_glob, args.cache)

    print(f"\nPASS 2 - evaluating tau = {args.tau:+g} min")
    with open(args.target, 'rb') as f:
        d = pickle.load(f)
    ev_all = np.asarray(d['glucose_events'])
    keep = np.isin(ev_all, list(keep_ids))
    X, y = d['X'][keep], np.asarray(d['y'], float)[keep]
    subj, ev = np.asarray(d['subjects'])[keep], ev_all[keep]
    del d, ev_all
    gc.collect()
    print(f"  restricted to {len(np.unique(ev)):,} events / {len(y):,} windows")

    if args.cap:
        X, y, subj, ev, _ = subsample_per_event(X, y, subj, ev, None, args.cap)
        print(f"  after cap={args.cap}: {len(y):,} windows")

    print("  extracting features...")
    Xf = extract_ecg_features(X, verbose=True)
    del X
    gc.collect()

    print("  nested LOSO (no RFECV, matching the other six arms)...")
    res = nested_loso_evaluate(Xf, y, subj, ev, do_rfecv=False,
                               n_iter=args.n_iter, verbose=True)
    row = res['pooled_event']
    row['tau_min'] = args.tau
    row['per_subject_argmin'] = None
    print(f"\n  tau={args.tau:+5.1f} min | MARD {row['mard']:6.2f}% | "
          f"MAE {row['mae']:6.1f} | R2 {row['r2']:+.4f}")

    print("\nPASS 3 - merging into lag_sweep.csv")
    new = pd.DataFrame([row])
    if os.path.exists(args.csv):
        old = pd.read_csv(args.csv)
        old = old[old['tau_min'] != args.tau]          # replace if already present
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out = out.sort_values('tau_min').reset_index(drop=True)
    out.to_csv(args.csv, index=False)
    print(out[['tau_min', 'n', 'mae', 'r2', 'mard']].to_string(index=False))
    print(f"\n  wrote {args.csv}")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
