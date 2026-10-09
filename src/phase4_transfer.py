#!/usr/bin/env python3

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from phase2_evaluate import (extract_ecg_features, compute_metrics, mard,
                             aggregate_to_events, metrics_by_glycemic_range,
                             _pipeline, subsample_per_event)

RNG = 42


def _load(path, cap):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    X, y = d['X'], np.asarray(d['y'], float)
    subj = np.asarray(d['subjects'])
    ev = np.asarray(d['glucose_events'])
    coh = np.asarray(d.get('cohort', np.zeros(len(y), int)))
    name = d.get('meta', {}).get('dataset', os.path.basename(path))
    if cap:
        X, y, subj, ev, coh = subsample_per_event(X, y, subj, ev, coh, cap)
    return dict(X=X, y=y, subjects=subj, events=ev, cohort=coh, name=name)


def _featurize(ds, verbose=True):
    if verbose:
        print(f"  featurizing {ds['name']}: {len(ds['y']):,} windows")
    ds['Xf'] = extract_ecg_features(ds['X'], verbose=verbose)
    del ds['X']
    return ds


def transfer_once(src, tgt, do_rfecv_features=None, verbose=True):

    Xs, ys = src['Xf'], src['y']
    Xt, yt = tgt['Xf'], tgt['y']

    # align columns defensively (same extractor -> should already match)
    cols = [c for c in Xs.columns if c in Xt.columns]
    Xs, Xt = Xs[cols], Xt[cols]

    model = _pipeline(RandomForestRegressor(n_estimators=300, random_state=RNG,
                                            n_jobs=-1))
    model.fit(Xs, ys)                      # SOURCE ONLY
    pred = model.predict(Xt)              # TARGET, no refitting

    # event-level metrics on the target
    ey, ep, _ = aggregate_to_events(yt, pred, tgt['events'])
    m = compute_metrics(ey, ep, label=f"{src['name']}->{tgt['name']}")
    m['n_target_events'] = int(len(ey))
    m['n_features'] = len(cols)

    # baseline: predict SOURCE mean on every target event
    src_mean = float(np.mean(ys))
    by, bp, _ = aggregate_to_events(yt, np.full(len(yt), src_mean), tgt['events'])
    baseline = compute_metrics(by, bp, label=f"predict_source_mean({src_mean:.0f})")

    by_range = metrics_by_glycemic_range(ey, ep)

    if verbose:
        print(f"\n  {src['name']} -> {tgt['name']}")
        print(f"    transfer : R2 {m['r2']:+.3f} | MARD {m['mard']:.1f}% | "
              f"MAE {m['mae']:.1f} | n_events {m['n_target_events']}")
        print(f"    baseline : R2 {baseline['r2']:+.3f} | MARD {baseline['mard']:.1f}% "
              f"(predict source mean)")

    return {'transfer': m, 'baseline': baseline, 'by_range': by_range,
            'event_true': ey, 'event_pred': ep}


def permutation_transfer(src, tgt, n_perm=3, verbose=True):

    rng = np.random.default_rng(RNG)
    Xs, ys = src['Xf'], src['y']
    model = _pipeline(RandomForestRegressor(n_estimators=300, random_state=RNG,
                                            n_jobs=-1))
    cols = [c for c in Xs.columns if c in tgt['Xf'].columns]
    model.fit(Xs[cols], ys)
    Xt = tgt['Xf'][cols]

    rows = []
    for i in range(n_perm):
        yperm = tgt['y'].copy()
        for s in np.unique(tgt['subjects']):
            msk = tgt['subjects'] == s
            ev = pd.Series(tgt['events'][msk])
            uniq = ev.unique()
            vals = pd.Series(tgt['y'][msk]).groupby(ev.values).first().loc[uniq].values
            yperm[msk] = ev.map(dict(zip(uniq, rng.permutation(vals)))).values
        pred = model.predict(Xt)
        ey, ep, _ = aggregate_to_events(yperm, pred, tgt['events'])
        r = compute_metrics(ey, ep, label=f"perm_{i}")
        rows.append(r)
        if verbose:
            print(f"    perm {i}: R2 {r['r2']:+.3f} MARD {r['mard']:.1f}%")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d1namo', default='preprocessed_data_taup5.pkl')
    ap.add_argument('--physiocgm', default='PhysioCGM/physiocgm_taup5.pkl')
    ap.add_argument('--cap', type=int, default=20,
                    help='windows per event cap (keep same as within-dataset runs)')
    ap.add_argument('--n-perm', type=int, default=3)
    ap.add_argument('--out-dir', default='transfer_results')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("loading + featurizing both datasets...")
    d1 = _featurize(_load(args.d1namo, args.cap or None))
    pc = _featurize(_load(args.physiocgm, args.cap or None))

    print("\n=== CROSS-DATASET TRANSFER ===")
    a = transfer_once(d1, pc)     # D1NAMO -> PhysioCGM
    b = transfer_once(pc, d1)     # PhysioCGM -> D1NAMO

    print("\n=== PERMUTATION CONTROL (target labels shuffled) ===")
    print(f"  {d1['name']} -> {pc['name']}:")
    perm_a = permutation_transfer(d1, pc, args.n_perm)
    print(f"  {pc['name']} -> {d1['name']}:")
    perm_b = permutation_transfer(pc, d1, args.n_perm)

    # ---- persist ----
    summary = pd.DataFrame([
        {**a['transfer'], 'direction': f"{d1['name']}->{pc['name']}", 'kind': 'transfer'},
        {**a['baseline'], 'direction': f"{d1['name']}->{pc['name']}", 'kind': 'baseline_source_mean'},
        {**b['transfer'], 'direction': f"{pc['name']}->{d1['name']}", 'kind': 'transfer'},
        {**b['baseline'], 'direction': f"{pc['name']}->{d1['name']}", 'kind': 'baseline_source_mean'},
    ])
    summary.to_csv(os.path.join(args.out_dir, 'transfer_summary.csv'), index=False)
    a['by_range'].to_csv(os.path.join(args.out_dir, 'transfer_d1namo_to_physiocgm_by_range.csv'), index=False)
    b['by_range'].to_csv(os.path.join(args.out_dir, 'transfer_physiocgm_to_d1namo_by_range.csv'), index=False)
    perm_a.to_csv(os.path.join(args.out_dir, 'transfer_perm_d1namo_to_physiocgm.csv'), index=False)
    perm_b.to_csv(os.path.join(args.out_dir, 'transfer_perm_physiocgm_to_d1namo.csv'), index=False)

    for tag, res in [('d1namo_to_physiocgm', a), ('physiocgm_to_d1namo', b)]:
        np.savez(os.path.join(args.out_dir, f'transfer_{tag}_predictions.npz'),
                 y_true=res['event_true'], y_pred=res['event_pred'])

    with open(os.path.join(args.out_dir, 'transfer_metrics.json'), 'w') as f:
        json.dump({'d1namo_to_physiocgm': {'transfer': a['transfer'], 'baseline': a['baseline']},
                   'physiocgm_to_d1namo': {'transfer': b['transfer'], 'baseline': b['baseline']}},
                  f, indent=2, default=float)

    print(f"\nwrote results to {args.out_dir}/")
    print("\nSUMMARY")
    print(summary[['direction', 'kind', 'n', 'mae', 'mard', 'r2']].to_string(index=False))


if __name__ == '__main__':
    main()
