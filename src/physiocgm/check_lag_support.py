#!/usr/bin/env python3

import argparse
import gc
import glob as globmod
import pickle

import numpy as np
import pandas as pd

MIN_EVENTS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lag-glob', default='PhysioCGM/physiocgm_tau*.pkl')
    ap.add_argument('--out', default='results/lag_support_precheck.csv')
    args = ap.parse_args()

    paths = sorted(globmod.glob(args.lag_glob))
    if len(paths) < 2:
        raise SystemExit(f"need >=2 tau files matching {args.lag_glob}; found {len(paths)}")
    print(f"found {len(paths)} tau files\n")

    arms = {}
    for p in paths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        tau = float(d['meta']['tau_minutes'])
        ev = np.asarray(d['glucose_events'])
        arms[tau] = set(ev)
        print(f"  tau={tau:+6.1f} min | {len(ev):>8,} windows | "
              f"{len(arms[tau]):>7,} unique events | {p}")
        del d, ev
        gc.collect()

    taus = sorted(arms)
    common = set.intersection(*(arms[t] for t in taus))

    print("\npairwise shared-event matrix (rows/cols = tau, min):")
    M = pd.DataFrame(
        [[len(arms[a] & arms[b]) for b in taus] for a in taus],
        index=taus, columns=taus, dtype=int)
    print(M.to_string())

    print(f"\ncommon support across all {len(taus)} arms = {len(common):,} events")
    if len(common) < MIN_EVENTS:
        print(f"\n  -> BELOW the {MIN_EVENTS}-event threshold. Cross-lag comparison is "
              "ill-posed on PhysioCGM as well: lag settings align to disjoint\n"
              "     reference sets. Report the identifiability finding; the full "
              "LOSO sweep is unnecessary (phase2 would exit early anyway).")
    else:
        frac = 100.0 * len(common) / max(len(arms[t]) for t in taus)
        print(f"\n  -> WELL-POSED ({frac:.1f}% of the largest arm is shared by all arms). "
              "The lag sweep is meaningful here.\n"
              "     Proceed to Step 3, but expect a long run: 7 arms x nested LOSO.")

    M.to_csv(args.out.replace('.csv', '_matrix.csv'))
    pd.DataFrame({'tau_min': taus,
                  'n_events': [len(arms[t]) for t in taus],
                  'common_support': len(common)}).to_csv(args.out, index=False)
    print(f"\nwrote {args.out} and {args.out.replace('.csv', '_matrix.csv')}")


if __name__ == '__main__':
    main()
