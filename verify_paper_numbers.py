#!/usr/bin/env python3
"""
verify_paper_numbers.py
=======================

Usage
-----
    python verify_paper_numbers.py
    python verify_paper_numbers.py --results-dir results --verbose

Exit status is 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

# =============================================================================
# Manuscript values. Transcribed from the paper; the single source of truth for
# what this script asserts. Edit here if the manuscript changes.
# =============================================================================

PAPER = {
    # Table 1 - cohort characteristics
    "t1.events":            {"D1NAMO": 52,      "PhysioCGM": 33757},
    "t1.windows_aligned":   {"D1NAMO": 138747,  "PhysioCGM": 135902},
    "t1.windows_model":     {"D1NAMO": 1021,    "PhysioCGM": 134729},
    "t1.glucose_mean":      {"D1NAMO": 124.8,   "PhysioCGM": 129.2},
    "t1.glucose_sd":        {"D1NAMO": 54.4,    "PhysioCGM": 45.1},
    "t1.glucose_median":    {"D1NAMO": 107.2,   "PhysioCGM": 119.0},
    "t1.glucose_min":       {"D1NAMO": 39.6,    "PhysioCGM": 40.0},
    "t1.glucose_max":       {"D1NAMO": 304.5,   "PhysioCGM": 392.0},
    "t1.n_hypo":            {"D1NAMO": 4,       "PhysioCGM": 1145},
    "t1.n_normo":           {"D1NAMO": 41,      "PhysioCGM": 28378},
    "t1.n_hyper":           {"D1NAMO": 7,       "PhysioCGM": 4234},
    "t1.pct_hypo":          {"D1NAMO": 7.7,     "PhysioCGM": 3.4},
    "t1.pct_normo":         {"D1NAMO": 78.8,    "PhysioCGM": 84.1},
    "t1.pct_hyper":         {"D1NAMO": 13.5,    "PhysioCGM": 12.5},

    # Table 2 - split-severity ablation. Order: random / event / LOSO / LOSO+event
    "t2.n":    {"D1NAMO": [205, 220, 1021, 52],
                "PhysioCGM": [26946, 26928, 134729, 33757]},
    "t2.mae":  {"D1NAMO": [32.5, 51.8, 42.9, 42.0],
                "PhysioCGM": [25.6, 25.5, 36.2, 35.5]},
    "t2.rmse": {"D1NAMO": [47.5, 71.0, 56.4, 55.9],
                "PhysioCGM": [35.3, 35.0, 47.2, 46.5]},
    "t2.r2":   {"D1NAMO": [0.248, -0.132, -0.086, -0.076],
                "PhysioCGM": [0.392, 0.391, -0.097, -0.061]},
    "t2.mard": {"D1NAMO": [38.6, 57.8, 42.1, 41.3],
                "PhysioCGM": [21.6, 21.3, 30.3, 29.7]},
    "t2.A":    {"D1NAMO": [57.1, 35.5, 36.1, 40.4],
                "PhysioCGM": [60.4, 60.6, 42.2, 42.7]},
    "t2.B":    {"D1NAMO": [30.7, 46.4, 50.0, 44.2],
                "PhysioCGM": [34.6, 34.9, 51.6, 51.2]},
    "t2.D":    {"D1NAMO": [12.2, 18.2, 13.7, 13.5],
                "PhysioCGM": [4.9, 4.5, 6.1, 6.1]},

    # Table 3 - PhysioCGM lag sweep, tau = -10 -5 0 +5 +10 +15 +20
    "t3.tau":  [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
    "t3.n":    [32525, 32525, 32525, 32524, 32525, 32525, 32525],
    "t3.mae":  [34.71, 34.75, 34.70, 34.69, 34.75, 34.78, 34.77],
    "t3.r2":   [-0.025, -0.025, -0.026, -0.020, -0.025, -0.026, -0.022],
    "t3.mard": [28.82, 28.87, 28.75, 28.85, 28.80, 28.86, 28.97],
    "t3.spread_mae": 0.09, "t3.spread_r2": 0.006, "t3.spread_mard": 0.22,
    "t3.perm_spread_r2": 0.022, "t3.perm_spread_mard": 0.33,
    "t3.common_support": 32524,

    # Table 4 - primary leakage-free result with cluster-bootstrap CIs
    "t4.r2":   {"D1NAMO": (-0.021, -0.240, 0.054),  "PhysioCGM": (-0.040, -0.148, -0.001)},
    "t4.mard": {"D1NAMO": (40.1, 27.2, 52.4),       "PhysioCGM": (29.0, 26.6, 31.3)},
    "t4.mae":  {"D1NAMO": (41.0, 29.6, 50.9),       "PhysioCGM": (35.0, 32.1, 38.4)},
    "t4.rmse": {"D1NAMO": (54.4, 38.0, 66.1),       "PhysioCGM": (46.0, 40.7, 51.6)},

    # Table 5 - glycaemic strata. Order: hypo / normo / hyper
    "t5.n":    {"D1NAMO": [4, 41, 7],       "PhysioCGM": [1145, 28378, 4234]},
    "t5.mae":  {"D1NAMO": [76.3, 26.6, 105.0], "PhysioCGM": [67.2, 25.7, 88.9]},
    "t5.r2":   {"D1NAMO": [-123.0, -0.60, -6.16], "PhysioCGM": [-87.3, -0.25, -6.04]},
    "t5.mard": {"D1NAMO": [183.6, 25.5, 43.5], "PhysioCGM": [114.7, 23.9, 39.5]},

    # Table 6 - cross-cohort transfer
    "t6.transfer_n":    {"d1namo_to_physiocgm": 33757, "physiocgm_to_d1namo": 52},
    "t6.transfer_mae":  {"d1namo_to_physiocgm": 42.6,  "physiocgm_to_d1namo": 52.4},
    "t6.transfer_r2":   {"d1namo_to_physiocgm": -0.61, "physiocgm_to_d1namo": -0.31},
    "t6.transfer_mard": {"d1namo_to_physiocgm": 31.3,  "physiocgm_to_d1namo": 58.0},
    "t6.baseline_mae":  {"d1namo_to_physiocgm": 34.0,  "physiocgm_to_d1namo": 41.6},
    "t6.baseline_r2":   {"d1namo_to_physiocgm": -0.006,"physiocgm_to_d1namo": -0.006},
    "t6.baseline_mard": {"d1namo_to_physiocgm": 27.8,  "physiocgm_to_d1namo": 41.5},
    "t6.perm_r2":       {"d1namo_to_physiocgm": -0.61, "physiocgm_to_d1namo": -0.35},
    "t6.perm_mard":     {"d1namo_to_physiocgm": 31.5,  "physiocgm_to_d1namo": 58.8},

    # Inline claims
    "in.b0_mae":        {"D1NAMO": 41.6,  "PhysioCGM": 35.2},
    "in.b1_r2":         {"D1NAMO": 0.29,  "PhysioCGM": 0.15},
    "in.b2_r2_d1namo":  0.04,
    "in.perm_r2_lo":    {"D1NAMO": -0.03, "PhysioCGM": -0.08},
    "in.perm_r2_hi":    {"D1NAMO": -0.004,"PhysioCGM": -0.06},
    "in.perm_dmard":    {"D1NAMO": 2.0,   "PhysioCGM": -0.5},
    "in.gap_to_b1":     {"D1NAMO": 10.2,  "PhysioCGM": 2.8},
    "in.inner_cv_min":  {"D1NAMO": -1.02, "PhysioCGM": -0.36},
    "in.inner_cv_max":  {"D1NAMO": -0.10, "PhysioCGM": -0.14},
    "in.inner_cv_med":  {"D1NAMO": -0.29, "PhysioCGM": -0.17},
    "in.folds_negative": 30,
    "in.clarke_A":      {"D1NAMO": 36.5,  "PhysioCGM": 42.9},
    "in.clarke_D":      {"D1NAMO": 13.5,  "PhysioCGM": 6.2},
    "in.clarke_AB_model":    {"D1NAMO": 86.54, "PhysioCGM": 93.84},
    "in.clarke_AB_constant": {"D1NAMO": 86.54, "PhysioCGM": 93.82},
    "in.zoneD_ci":      {"D1NAMO": (3.3, 23.7), "PhysioCGM": (5.0, 7.3)},
    "in.pred_min":      {"D1NAMO": 75,   "PhysioCGM": 92},
    "in.pred_max":      {"D1NAMO": 165,  "PhysioCGM": 201},
    "in.window_r2":     {"D1NAMO": -0.045, "PhysioCGM": -0.051},
    "in.window_mard":   {"D1NAMO": 40.37,  "PhysioCGM": 29.18},
    "in.ci_inflation":  {"D1NAMO": 4.4,    "PhysioCGM": 2.0},
    "in.corr_mae_sd":   {"D1NAMO": 0.82, "PhysioCGM": 0.91},
    "in.corr_pooled":   0.77,
    "in.spearman_pooled": 0.72,
    "in.mard_min_d1namo": 6.5,
    "in.mard_max_d1namo": 97.4,
    "in.mard_med_d1namo": 24.9,
    "in.subjects_positive_r2": {"D1NAMO": 0, "PhysioCGM": 1},
    "in.nfeat_min":     {"D1NAMO": 5,  "PhysioCGM": 10},
    "in.nfeat_max":     {"D1NAMO": 49, "PhysioCGM": 47},
    "in.nfeat_med":     {"D1NAMO": 13, "PhysioCGM": 20},
    "in.allfold_feats": {"D1NAMO": 3,  "PhysioCGM": 9},
    "in.hrv_median_freq": {"D1NAMO": 0.35, "PhysioCGM": 0.40},
    "in.d1namo_lag_support": 0,
    "in.d1namo_lag_arms_min": 38,
    "in.d1namo_lag_arms_max": 54,
    "in.transfer_hyper_mard": 51.9,
    "in.transfer_hyper_r2":  -10.5,
    "in.transfer_normo_mard": 26.4,
    "in.rev_hypo_mard": 252.3,
    "in.rev_hypo_r2":  -231.0,
    "in.rev_normo_mard": 43.8,
}

HRV_FEATURES = ["rmssd", "sdnn", "pnn50", "nn50", "sdsd", "std_rr",
                "sample_entropy", "lf_hf_ratio", "lf_power", "hf_power", "vlf_power"]

COHORTS = {"D1NAMO": "d1namo", "PhysioCGM": "physiocgm"}


def mae(y, p):   return float(np.mean(np.abs(y - p)))
def rmse(y, p):  return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
def mard(y, p):
    ok = y > 0
    return float(np.mean(np.abs(p[ok] - y[ok]) / y[ok]) * 100)


def clarke_zone(ref, pred):
    """Clarke 1987 zone assignment. Branch order is significant."""
    r, q = float(ref), float(pred)
    if (r <= 70 and q <= 70) or (0.8 * r <= q <= 1.2 * r):
        return "A"
    if (r >= 180 and q <= 70) or (r <= 70 and q >= 180):
        return "E"
    if ((70 <= r <= 290) and q >= r + 110) or \
       ((130 <= r <= 180) and q <= (7.0 / 5.0) * r - 182):
        return "C"
    if (r >= 240 and 70 <= q <= 180) or \
       (r <= 175.0 / 3.0 and 70 <= q <= 180) or \
       ((175.0 / 3.0 <= r <= 70) and q >= (6.0 / 5.0) * r):
        return "D"
    return "B"


def cluster_bootstrap(values, subjects, stat, n_boot=10000, seed=42):
    """Percentile CI resampling whole subjects, matching the LOSO design."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    idx = {s: np.flatnonzero(subjects == s) for s in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in pick])
        try:
            v = stat(sel)
        except Exception:
            continue
        if np.isfinite(v):
            out.append(v)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# =============================================================================
# Reporting
# =============================================================================

class Report:
    def __init__(self, verbose=False):
        self.rows, self.verbose = [], verbose
        self.section = None

    def head(self, title):
        self.section = title
        print(f"\n{title}")
        print("-" * len(title))

    def check(self, item, paper, computed, tol):
        try:
            ok = abs(float(paper) - float(computed)) <= tol
        except (TypeError, ValueError):
            ok = (paper == computed)
        self.rows.append(ok)
        if not ok or self.verbose:
            mark = "PASS" if ok else "FAIL"
            pv = f"{paper:.4g}" if isinstance(paper, (int, float)) else str(paper)
            cv = f"{computed:.4g}" if isinstance(computed, (int, float)) else str(computed)
            print(f"  {mark}  {item:<48} paper {pv:>12}   computed {cv:>12}")
        return ok

    def summary(self):
        n, p = len(self.rows), sum(self.rows)
        print("\n" + "=" * 72)
        if p == n:
            print(f"  {n} checks, {p} passed, 0 failed.")
            print("  Every numerical claim in the manuscript matches the stored results.")
        else:
            print(f"  {n} checks, {p} passed, {n - p} FAILED.")
        print("=" * 72)
        return p == n


# =============================================================================
# Loading
# =============================================================================

def load(root):
    D = {}
    for name, sub in COHORTS.items():
        d = os.path.join(root, sub)
        z = np.load(os.path.join(d, "event_predictions.npz"), allow_pickle=True)
        ci_file = next(f for f in os.listdir(d) if f.startswith("ci_") and f.endswith(".json"))
        D[name] = {
            "y": z["y_true"].astype(float),
            "p": z["y_pred"].astype(float),
            "subj": np.array([str(e).split("|")[0] for e in z["event_id"]]),
            "pooled": json.load(open(os.path.join(d, "pooled_metrics.json"))),
            "ci": json.load(open(os.path.join(d, ci_file)))["ci"],
            "ablation": pd.read_csv(os.path.join(d, "split_severity_ablation.csv")),
            "baseline": pd.read_csv(os.path.join(d, "baseline_ladder.csv")),
            "perm": pd.read_csv(os.path.join(d, "label_permutation.csv")),
            "subject": pd.read_csv(os.path.join(d, "per_subject.csv")),
            "range": pd.read_csv(os.path.join(d, "by_glycemic_range.csv")),
            "feat": pd.read_csv(os.path.join(d, "feature_selection_frequency.csv")),
        }
    lag_dir = os.path.join(root, "physiocgm_lag")
    D["lag"] = pd.read_csv(os.path.join(lag_dir, "lag_sweep.csv")).sort_values("tau_min")
    D["support"] = json.load(open(os.path.join(lag_dir, "common_support_7arm.json")))
    t = os.path.join(root, "transfer")
    D["transfer"] = json.load(open(os.path.join(t, "transfer_metrics.json")))
    D["tperm"] = {
        "d1namo_to_physiocgm": pd.read_csv(os.path.join(t, "transfer_perm_d1namo_to_physiocgm.csv")),
        "physiocgm_to_d1namo": pd.read_csv(os.path.join(t, "transfer_perm_physiocgm_to_d1namo.csv")),
    }
    D["trange"] = {
        "d1namo_to_physiocgm": pd.read_csv(os.path.join(t, "transfer_d1namo_to_physiocgm_by_range.csv")),
        "physiocgm_to_d1namo": pd.read_csv(os.path.join(t, "transfer_physiocgm_to_d1namo_by_range.csv")),
    }
    d1_lag = os.path.join(root, "d1namo", "lag_sweep.csv")
    D["d1lag"] = pd.read_csv(d1_lag) if os.path.exists(d1_lag) else None
    return D


# =============================================================================
# Checks
# =============================================================================

def run(D, rep, n_boot):
    P = PAPER

    # ---------------- Table 1 ----------------
    rep.head("Table 1  Cohort characteristics and replication structure")
    for c in COHORTS:
        y = D[c]["y"]
        rep.check(f"{c} glucose events", P["t1.events"][c], len(y), 0)
        rep.check(f"{c} aligned ECG windows", P["t1.windows_aligned"][c],
                  D[c]["pooled"]["meta"]["n_windows"], 0)
        rep.check(f"{c} windows used for modelling", P["t1.windows_model"][c],
                  int(D[c]["subject"]["n_windows"].sum()), 0)
        rep.check(f"{c} glucose mean", P["t1.glucose_mean"][c], y.mean(), 0.051)
        rep.check(f"{c} glucose SD", P["t1.glucose_sd"][c], y.std(ddof=1), 0.051)
        rep.check(f"{c} glucose median", P["t1.glucose_median"][c], np.median(y), 0.051)
        rep.check(f"{c} glucose min", P["t1.glucose_min"][c], y.min(), 0.051)
        rep.check(f"{c} glucose max", P["t1.glucose_max"][c], y.max(), 0.051)
        lo, mid, hi = y < 70, (y >= 70) & (y <= 180), y > 180
        for lbl, m, k, kp in [("<70", lo, "t1.n_hypo", "t1.pct_hypo"),
                              ("70-180", mid, "t1.n_normo", "t1.pct_normo"),
                              (">180", hi, "t1.n_hyper", "t1.pct_hyper")]:
            rep.check(f"{c} events {lbl}", P[k][c], int(m.sum()), 0)
            rep.check(f"{c} events {lbl} (%)", P[kp][c], 100 * m.mean(), 0.051)

    # ---------------- Table 2 ----------------
    rep.head("Table 2  Split-severity ablation")
    for c in COHORTS:
        a = D[c]["ablation"]
        for i, prot in enumerate(["random window", "event-grouped", "LOSO", "LOSO+event"]):
            rep.check(f"{c} {prot} n", P["t2.n"][c][i], int(a.n[i]), 0.5)
            rep.check(f"{c} {prot} MAE", P["t2.mae"][c][i], a.mae[i], 0.051)
            rep.check(f"{c} {prot} RMSE", P["t2.rmse"][c][i], a.rmse[i], 0.051)
            rep.check(f"{c} {prot} R2", P["t2.r2"][c][i], a.r2[i], 0.0006)
            rep.check(f"{c} {prot} MARD", P["t2.mard"][c][i], a.mard[i], 0.051)
            rep.check(f"{c} {prot} Clarke A", P["t2.A"][c][i], a.clarke_A_pct[i], 0.051)
            rep.check(f"{c} {prot} Clarke B", P["t2.B"][c][i], a.clarke_B_pct[i], 0.051)
            rep.check(f"{c} {prot} Clarke D", P["t2.D"][c][i], a.clarke_D_pct[i], 0.051)

    # ---------------- Table 3 ----------------
    rep.head("Table 3  PhysioCGM lag sweep (seven arms)")
    L = D["lag"].reset_index(drop=True)
    rep.check("arms present", 7, len(L), 0)
    for i, tau in enumerate(P["t3.tau"]):
        row = L[L.tau_min == tau]
        if row.empty:
            rep.check(f"tau={tau:+g} present", "yes", "ABSENT", 0)
            continue
        row = row.iloc[0]
        rep.check(f"tau={tau:+g} n", P["t3.n"][i], int(row.n), 0.5)
        rep.check(f"tau={tau:+g} MAE", P["t3.mae"][i], row.mae, 0.006)
        rep.check(f"tau={tau:+g} R2", P["t3.r2"][i], row.r2, 0.0006)
        rep.check(f"tau={tau:+g} MARD", P["t3.mard"][i], row.mard, 0.006)
    rep.check("MAE spread", P["t3.spread_mae"], L.mae.max() - L.mae.min(), 0.006)
    rep.check("R2 spread", P["t3.spread_r2"], L.r2.max() - L.r2.min(), 0.0006)
    rep.check("MARD spread", P["t3.spread_mard"], L.mard.max() - L.mard.min(), 0.006)
    pp = D["PhysioCGM"]["perm"]
    rep.check("permutation R2 spread", P["t3.perm_spread_r2"], pp.r2.max() - pp.r2.min(), 0.0006)
    rep.check("permutation MARD spread", P["t3.perm_spread_mard"], pp.mard.max() - pp.mard.min(), 0.006)
    rep.check("common support (7 arms)", P["t3.common_support"], len(D["support"]), 0)
    rep.check("common support subjects", 10, len(set(e.split("|")[0] for e in D["support"])), 0)

    # ---------------- Table 4 ----------------
    rep.head("Table 4  Primary leakage-free result and cluster-bootstrap CIs")
    for c in COHORTS:
        y, p, s = D[c]["y"], D[c]["p"], D[c]["subj"]
        fns = {"r2": lambda i: r2(y[i], p[i]), "mard": lambda i: mard(y[i], p[i]),
               "mae": lambda i: mae(y[i], p[i]), "rmse": lambda i: rmse(y[i], p[i])}
        pts = {"r2": r2(y, p), "mard": mard(y, p), "mae": mae(y, p), "rmse": rmse(y, p)}
        tols = {"r2": 0.0006, "mard": 0.051, "mae": 0.051, "rmse": 0.051}
        for m in ["r2", "mard", "mae", "rmse"]:
            want, lo_w, hi_w = P[f"t4.{m}"][c]
            rep.check(f"{c} {m.upper()} point", want, pts[m], tols[m])
            lo, hi = cluster_bootstrap(None, s, fns[m], n_boot=n_boot)
            rep.check(f"{c} {m.upper()} CI low", lo_w, lo, max(tols[m], 0.06))
            rep.check(f"{c} {m.upper()} CI high", hi_w, hi, max(tols[m], 0.06))

    # ---------------- Table 5 ----------------
    rep.head("Table 5  Performance by glycaemic range")
    for c in COHORTS:
        g = D[c]["range"]
        for i, lbl in enumerate(["hypo", "normo", "hyper"]):
            rep.check(f"{c} {lbl} n", P["t5.n"][c][i], int(g.n[i]), 0.5)
            rep.check(f"{c} {lbl} MAE", P["t5.mae"][c][i], g.mae[i], 0.051)
            rep.check(f"{c} {lbl} R2", P["t5.r2"][c][i], g.r2[i], 0.051)
            rep.check(f"{c} {lbl} MARD", P["t5.mard"][c][i], g.mard[i], 0.051)

    # ---------------- Table 6 ----------------
    rep.head("Table 6  Cross-cohort transfer")
    for d in ["d1namo_to_physiocgm", "physiocgm_to_d1namo"]:
        t, b = D["transfer"][d]["transfer"], D["transfer"][d]["baseline"]
        pm = D["tperm"][d]
        rep.check(f"{d} transfer n", P["t6.transfer_n"][d], t["n"], 0.5)
        rep.check(f"{d} transfer MAE", P["t6.transfer_mae"][d], t["mae"], 0.051)
        rep.check(f"{d} transfer R2", P["t6.transfer_r2"][d], t["r2"], 0.0051)
        rep.check(f"{d} transfer MARD", P["t6.transfer_mard"][d], t["mard"], 0.051)
        rep.check(f"{d} baseline MAE", P["t6.baseline_mae"][d], b["mae"], 0.051)
        rep.check(f"{d} baseline R2", P["t6.baseline_r2"][d], b["r2"], 0.0051)
        rep.check(f"{d} baseline MARD", P["t6.baseline_mard"][d], b["mard"], 0.051)
        rep.check(f"{d} permutation R2", P["t6.perm_r2"][d], pm.r2.mean(), 0.0051)
        rep.check(f"{d} permutation MARD", P["t6.perm_mard"][d], pm.mard.mean(), 0.051)

    # ---------------- Inline ----------------
    rep.head("Inline claims in the text")
    pooled_mae, pooled_sd = [], []
    for c in COHORTS:
        y, p, s = D[c]["y"], D[c]["p"], D[c]["subj"]
        b, pm, sub, ft = D[c]["baseline"], D[c]["perm"], D[c]["subject"], D[c]["feat"]
        pooled = D[c]["pooled"]

        rep.check(f"{c} b0 MAE", P["in.b0_mae"][c], b.mae[0], 0.051)
        rep.check(f"{c} b1 R2", P["in.b1_r2"][c], b.r2[1], 0.0051)
        rep.check(f"{c} permutation R2 low", P["in.perm_r2_lo"][c], pm.r2.min(), 0.0051)
        rep.check(f"{c} permutation R2 high", P["in.perm_r2_hi"][c], pm.r2.max(), 0.0051)
        rep.check(f"{c} MARD delta vs permutation", P["in.perm_dmard"][c],
                  mard(y, p) - pm.mard.mean(), 0.051)
        rep.check(f"{c} MARD gap to subject-mean", P["in.gap_to_b1"][c],
                  mard(y, p) - b.mard[1], 0.051)
        rep.check(f"{c} inner-CV R2 min", P["in.inner_cv_min"][c], sub.inner_cv_r2.min(), 0.0051)
        rep.check(f"{c} inner-CV R2 max", P["in.inner_cv_max"][c], sub.inner_cv_r2.max(), 0.0051)
        rep.check(f"{c} inner-CV R2 median", P["in.inner_cv_med"][c], sub.inner_cv_r2.median(), 0.0051)

        zones = np.array([clarke_zone(a, q) for a, q in zip(y, p)])
        rep.check(f"{c} Clarke A (%)", P["in.clarke_A"][c], 100 * (zones == "A").mean(), 0.051)
        rep.check(f"{c} Clarke D (%)", P["in.clarke_D"][c], 100 * (zones == "D").mean(), 0.051)
        rep.check(f"{c} Clarke A+B model (%)", P["in.clarke_AB_model"][c],
                  100 * np.isin(zones, ["A", "B"]).mean(), 0.006)
        rep.check(f"{c} Clarke A+B constant predictor (%)", P["in.clarke_AB_constant"][c],
                  b.clarke_A_pct[0] + b.clarke_B_pct[0], 0.006)
        lo, hi = cluster_bootstrap(None, s,
                                   lambda i: 100 * (zones[i] == "D").mean(), n_boot=n_boot)
        want_lo, want_hi = P["in.zoneD_ci"][c]
        rep.check(f"{c} Clarke Zone D CI low", want_lo, lo, 0.6)
        rep.check(f"{c} Clarke Zone D CI high", want_hi, hi, 0.6)

        rep.check(f"{c} prediction range min", P["in.pred_min"][c], round(p.min()), 1)
        rep.check(f"{c} prediction range max", P["in.pred_max"][c], round(p.max()), 1)
        rep.check(f"{c} window-level R2", P["in.window_r2"][c], pooled["pooled_window"]["r2"], 0.0006)
        rep.check(f"{c} window-level MARD", P["in.window_mard"][c], pooled["pooled_window"]["mard"], 0.006)
        rep.check(f"{c} CI inflation factor", P["in.ci_inflation"][c],
                  np.sqrt(pooled["pooled_window"]["n"] / pooled["pooled_event"]["n"]), 0.051)
        rep.check(f"{c} corr(MAE, glucose SD)", P["in.corr_mae_sd"][c],
                  sub.mae.corr(sub.glucose_std), 0.005)
        rep.check(f"{c} subjects with positive R2", P["in.subjects_positive_r2"][c],
                  int((sub.r2 > 0).sum()), 0)
        rep.check(f"{c} features retained, min", P["in.nfeat_min"][c], sub.n_features.min(), 0)
        rep.check(f"{c} features retained, max", P["in.nfeat_max"][c], sub.n_features.max(), 0)
        rep.check(f"{c} features retained, median", P["in.nfeat_med"][c], sub.n_features.median(), 0)
        rep.check(f"{c} features in every fold", P["in.allfold_feats"][c],
                  int((ft.selection_frequency == 1.0).sum()), 0)
        rep.check(f"{c} HRV median selection frequency", P["in.hrv_median_freq"][c],
                  ft[ft["index"].isin(HRV_FEATURES)].selection_frequency.median(), 0.005)
        pooled_mae.append(sub[["mae", "glucose_std"]])

    rep.check("D1NAMO cohort-mean baseline R2", P["in.b2_r2_d1namo"],
              D["D1NAMO"]["baseline"].r2[2], 0.006)
    rep.check("inner-CV negative in all folds", P["in.folds_negative"],
              sum(int((D[c]["subject"].inner_cv_r2 < 0).sum()) for c in COHORTS), 0)
    allsub = pd.concat(pooled_mae)
    rep.check("pooled corr(MAE, glucose SD)", P["in.corr_pooled"],
              allsub.mae.corr(allsub.glucose_std), 0.005)
    rep.check("pooled Spearman rho", P["in.spearman_pooled"],
              allsub.mae.corr(allsub.glucose_std, method="spearman"), 0.005)
    sd = D["D1NAMO"]["subject"]
    rep.check("D1NAMO per-subject MARD min", P["in.mard_min_d1namo"], sd.mard.min(), 0.051)
    rep.check("D1NAMO per-subject MARD max", P["in.mard_max_d1namo"], sd.mard.max(), 0.051)
    rep.check("D1NAMO per-subject MARD median", P["in.mard_med_d1namo"], sd.mard.median(), 0.051)

    if D["d1lag"] is not None:
        dl = D["d1lag"]
        rep.check("D1NAMO lag common support", P["in.d1namo_lag_support"],
                  int(dl.common_support.iloc[0]), 0)
        rep.check("D1NAMO lag arm events, min", P["in.d1namo_lag_arms_min"], dl.n_events.min(), 0)
        rep.check("D1NAMO lag arm events, max", P["in.d1namo_lag_arms_max"], dl.n_events.max(), 0)

    fw, rv = D["trange"]["d1namo_to_physiocgm"], D["trange"]["physiocgm_to_d1namo"]
    rep.check("transfer hyperglycaemic MARD", P["in.transfer_hyper_mard"], fw.iloc[2].mard, 0.051)
    rep.check("transfer hyperglycaemic R2", P["in.transfer_hyper_r2"], fw.iloc[2].r2, 0.051)
    rep.check("transfer normoglycaemic MARD", P["in.transfer_normo_mard"], fw.iloc[1].mard, 0.051)
    rep.check("reverse transfer hypoglycaemic MARD", P["in.rev_hypo_mard"], rv.iloc[0].mard, 0.051)
    rep.check("reverse transfer hypoglycaemic R2", P["in.rev_hypo_r2"], rv.iloc[0].r2, 0.11)
    rep.check("reverse transfer normoglycaemic MARD", P["in.rev_normo_mard"], rv.iloc[1].mard, 0.051)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results",
                    help="directory holding d1namo/, physiocgm/, physiocgm_lag/, transfer/")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="bootstrap replicates (default 10000, matching the paper)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every check, not only failures")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"results directory not found: {args.results_dir}")

    print("=" * 72)
    print("  Verification of every numerical claim in the manuscript")
    print(f"  results dir : {os.path.abspath(args.results_dir)}")
    print(f"  bootstrap   : {args.n_boot:,} replicates, seed 42, subject clusters")
    print("  Nothing outside results/ is read. No model is trained.")
    print("=" * 72)

    D = load(args.results_dir)
    rep = Report(verbose=args.verbose)
    run(D, rep, args.n_boot)
    sys.exit(0 if rep.summary() else 1)


if __name__ == "__main__":
    main()
