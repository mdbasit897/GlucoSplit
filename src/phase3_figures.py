#!/usr/bin/env python3
"""
Phase 3 - publication figures from Phase 2 results.

Reads the CSV / JSON / NPZ that phase2_evaluate.py already wrote and produces
300-dpi PNG + vector PDF for every panel the reframed paper needs.

Usage
-----
    python phase3_figures.py --results-dir results --out-dir figures

Reviewer #3 asked for higher resolution and clear labels/legends. Everything
here is 300 dpi, vector PDF, colourblind-safe palette, font sizes >= 9 pt.

Figures produced
----------------
    fig1_split_severity        the money figure: R2 and MARD across 4 protocols
    fig2_leakage_proof         real LOSO vs permuted labels, side by side
    fig3_baseline_ladder       model vs subject-mean / cohort-mean baselines
    fig4_per_subject           per-subject MARD, ranked, T1D vs healthy
    fig5_glycemic_range        MARD by hypo / normo / hyper (with n annotated)
    fig6_clarke_grid           Clarke error grid from event_predictions.npz
    fig7_pred_vs_true          predicted vs reference scatter + identity line
    fig8_feature_selection     top-20 RFECV selection frequency
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

# ---- global style: publication defaults -------------------------------------
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
# Okabe-Ito colourblind-safe palette
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
     'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999',
     'sky': '#56B4E9', 'yellow': '#F0E442'}


def _save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _read_csv(results_dir, name):
    p = os.path.join(results_dir, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# =============================================================================
# Fig 1 - split severity (the central contribution)
# =============================================================================

def fig_split_severity(df, out_dir):
    """
    """
    if df is None:
        return
    labels = ['Random\nwindow split\n(as submitted)', 'Event-\ngrouped',
              'Subject-\ngrouped\n(LOSO)', 'LOSO +\nevent-level']
    labels = labels[:len(df)]
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    colors = [C['red'], C['orange'], C['sky'], C['blue']][:len(df)]

    # ---- panel (a): R2 -----------------------------------------------------
    r2 = np.asarray(df['r2'], dtype=float)
    b1 = ax1.bar(x, r2, color=colors, edgecolor='black', linewidth=0.6)
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_ylabel('R\u00b2 (coefficient of determination)')
    ax1.set_title('(a) Predictive R\u00b2 collapses under\nleakage-free evaluation')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)

    lo, hi = min(0.0, r2.min()), max(0.0, r2.max())
    span = (hi - lo) or 1.0
    pad = 0.045 * span
    for r, v in zip(b1, r2):
        ax1.text(r.get_x() + r.get_width() / 2, v + (pad if v >= 0 else -pad),
                 f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top',
                 fontsize=8.5, fontweight='bold')
    ax1.set_ylim(lo - 0.22 * span, hi + 0.15 * span)

    # ---- panel (b): MARD ---------------------------------------------------
    mard = np.asarray(df['mard'], dtype=float)
    b2 = ax2.bar(x, mard, color=colors, edgecolor='black', linewidth=0.6)
    ax2.axhline(15, color=C['green'], ls='--', lw=1.2, label='Clinical target (15%)')
    ax2.set_ylabel('MARD (%)')
    ax2.set_title('(b) Error rises under\nleakage-free evaluation')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)

    top = max(mard.max(), 15.0)
    for r, v in zip(b2, mard):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.018 * top,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.set_ylim(0, top * 1.30)

    half = max(1, len(mard) // 2)
    side = 'upper right' if mard[:half].max() >= mard[half:].max() else 'upper left'
    ax2.legend(loc=side, framealpha=0.95, fontsize=8.5, borderpad=0.5)

    fig.suptitle('Evaluation protocol determines reported accuracy',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, 'fig1_split_severity')


# =============================================================================
# Fig 2 - leakage proof: real vs permuted
# =============================================================================

def fig_leakage_proof(perm_df, pooled, out_dir):
    if perm_df is None or pooled is None:
        return
    real_r2 = pooled['pooled_event']['r2']
    real_mard = pooled['pooled_event']['mard']
    perm_r2 = perm_df['r2'].values
    perm_mard = perm_df['mard'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.axhline(real_r2, color=C['red'], lw=2, label=f'Real labels (LOSO): {real_r2:+.3f}')
    ax1.scatter(range(1, len(perm_r2) + 1), perm_r2, color=C['grey'], s=60,
                zorder=3, label='Permuted labels')
    ax1.axhline(perm_r2.mean(), color=C['grey'], ls='--', lw=1,
                label=f'Permuted mean: {perm_r2.mean():+.3f}')
    ax1.axhline(0, color='black', lw=0.6)
    ax1.set_xlabel('Permutation run'); ax1.set_ylabel('R²')
    ax1.set_title('(a) Real performance is\nindistinguishable from chance')
    ax1.set_xticks(range(1, len(perm_r2) + 1))
    ax1.legend(fontsize=8, loc='best')

    ax2.axhline(real_mard, color=C['red'], lw=2, label=f'Real labels: {real_mard:.1f}%')
    ax2.scatter(range(1, len(perm_mard) + 1), perm_mard, color=C['grey'], s=60,
                zorder=3, label='Permuted labels')
    ax2.axhline(perm_mard.mean(), color=C['grey'], ls='--', lw=1,
                label=f'Permuted mean: {perm_mard.mean():.1f}%')
    ax2.set_xlabel('Permutation run'); ax2.set_ylabel('MARD (%)')
    ax2.set_title('(b) Error matches the\nlabel-shuffled control')
    ax2.set_xticks(range(1, len(perm_mard) + 1))
    ax2.legend(fontsize=8, loc='best')

    fig.suptitle('Label-permutation control confirms no recoverable ECG–glucose signal',
                 fontsize=11.5, fontweight='bold', y=1.02)
    _save(fig, out_dir, 'fig2_leakage_proof')


# =============================================================================
# Fig 3 - baseline ladder
# =============================================================================

def fig_baseline_ladder(bl, pooled, out_dir):
    if bl is None:
        return
    rows = list(bl['label'])
    r2 = list(bl['r2']); mard = list(bl['mard'])
    if pooled is not None:
        rows.append('ECG model (LOSO)')
        r2.append(pooled['pooled_event']['r2'])
        mard.append(pooled['pooled_event']['mard'])

    short = [r.split('(')[0].replace('b0 ', '').replace('b1 ', '')
             .replace('b2 ', '').strip() for r in rows]
    y = np.arange(len(rows))
    colors = [C['grey'], C['green'], C['sky'], C['red']][:len(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.barh(y, r2, color=colors, edgecolor='black', linewidth=0.6)
    ax1.axvline(0, color='black', lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(short, fontsize=9)
    ax1.set_xlabel('R²'); ax1.set_title('(a) R² — higher is better')
    ax1.invert_yaxis()
    for i, v in enumerate(r2):
        ax1.text(v + (0.005 if v >= 0 else -0.005), i, f'{v:+.3f}',
                 va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)

    ax2.barh(y, mard, color=colors, edgecolor='black', linewidth=0.6)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel('MARD (%)'); ax2.set_title('(b) MARD — lower is better')
    ax2.invert_yaxis()
    for i, v in enumerate(mard):
        ax2.text(v + 0.4, i, f'{v:.1f}', va='center', fontsize=8.5)

    fig.suptitle('The ECG model does not beat a subject-mean baseline',
                 fontsize=11.5, fontweight='bold', y=1.04)
    _save(fig, out_dir, 'fig3_baseline_ladder')


# =============================================================================
# Fig 4 - per-subject MARD
# =============================================================================

def fig_per_subject(ps, out_dir, cohort_labels=('T1D', 'Healthy'), title=None):
    """
    Per-subject MARD, ranked, coloured by cohort.

    `cohort_labels` sets the LEGEND text only; it does not change the data.
    The per_subject.csv 'cohort' column stores the internal values 'T1D' and
    'healthy', which on PhysioCGM correspond to the c1 and c2 RECORDING groups,
    not to disease status -- direct inspection of the CGM records shows neither
    PhysioCGM group is normoglycaemic. Pass ('Cohort c1', 'Cohort c2') for that
    dataset so the legend does not assert a disease contrast that is not there.
    """
    if ps is None:
        return
    lab_a, lab_b = cohort_labels
    d = ps.copy()
    if 'cohort' not in d.columns:
        d['cohort'] = d['label'].astype(str).str.startswith('D').map(
            {True: 'T1D', False: 'healthy'})
    d = d.sort_values('mard')
    colors = [C['red'] if c == 'T1D' else C['blue'] for c in d['cohort']]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(d))
    ax.bar(x, d['mard'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(15, color=C['green'], ls='--', lw=1.2, label='Clinical target (15%)')
    ax.set_xticks(x); ax.set_xticklabels(d['label'], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('MARD (%)'); ax.set_xlabel('Subject (sorted by MARD)')
    # Neutral by default: 'wide heterogeneity' is true of D1NAMO (MARD 6.5-97.4%)
    # but false of PhysioCGM (23.9-34.8%). Cohort-specific claims belong in the
    # manuscript caption, not baked into the figure.
    ax.set_title(title or 'Per-subject MARD under leave-one-subject-out evaluation')

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C['red'], label=lab_a),
               Patch(facecolor=C['blue'], label=lab_b),
               plt.Line2D([0], [0], color=C['green'], ls='--', label='Clinical target (15%)')]
    ax.legend(handles=handles, loc='upper left')

    ncol = 'n_events' if 'n_events' in d.columns else ('n' if 'n' in d.columns else None)
    if ncol and len(d) <= 30:
        for i, (_, r) in enumerate(d.iterrows()):
            ax.text(i, r['mard'] + 1, f"n={int(r[ncol])}",
                    ha='center', fontsize=6.5, color=C['grey'])
    _save(fig, out_dir, 'fig4_per_subject')

    # companion: does MARD track events-per-subject rather than cohort?
    if ncol:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for coh, col in (('T1D', C['red']), ('healthy', C['blue'])):
            sub = d[d['cohort'] == coh]
            ax.scatter(sub[ncol], sub['mard'], color=col, s=55,
                       edgecolor='black', linewidth=0.5,
                       label=lab_a if coh == 'T1D' else lab_b, zorder=3)
        if len(d) > 2:
            r = np.corrcoef(d[ncol], d['mard'])[0, 1]
            ax.text(0.95, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', fc='white', ec=C['grey']))
        ax.set_xlabel('Glucose events per subject')
        ax.set_ylabel('MARD (%)')
        ax.set_title('Is heterogeneity driven by label count,\nnot autonomic status?')
        ax.legend()
        _save(fig, out_dir, 'fig4b_mard_vs_events')


# =============================================================================
# Fig 5 - MARD by glycemic range (with n annotated)
# =============================================================================

def fig_glycemic_range(gr, out_dir):
    if gr is None:
        return
    d = gr.copy()
    short = [l.split('(')[0].strip().capitalize() for l in d['label']]
    x = np.arange(len(d))
    colors = [C['blue'], C['green'], C['orange']][:len(d)]

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    bars = ax.bar(x, d['mard'], color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(15, color=C['green'], ls='--', lw=1, label='Clinical target (15%)')
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('MARD (%)')
    ax.set_title('MARD by glycemic range')
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r['mard'] + max(d['mard']) * 0.01,
                f"{r['mard']:.1f}%\n(n={int(r['n'])})", ha='center',
                va='bottom', fontsize=8.5, fontweight='bold')
    ax.legend()
    ax.text(0.5, -0.22,
            'Note: dysglycemic events are too few (annotated n) for reliable estimation.',
            transform=ax.transAxes, ha='center', fontsize=7.5, style='italic', color=C['grey'])
    _save(fig, out_dir, 'fig5_glycemic_range')


# =============================================================================
# Fig 6 - Clarke error grid
# =============================================================================

def _clarke_zone(ref, pred):
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


def fig_clarke_grid(npz_path, out_dir):
    if not os.path.exists(npz_path):
        print("  (skip clarke grid: event_predictions.npz not found)")
        return
    dat = np.load(npz_path, allow_pickle=True)
    ref, pred = dat['y_true'].astype(float), dat['y_pred'].astype(float)

    fig, ax = plt.subplots(figsize=(6, 6))
    mx = max(400, ref.max() * 1.1, pred.max() * 1.1)
    ax.plot([0, mx], [0, mx], color=C['grey'], lw=0.8, zorder=1)

    # zone boundary lines (standard Clarke)
    ax.plot([0, 175 / 3], [70, 70], 'k-', lw=0.6)
    ax.plot([175 / 3, 400], [70, 70], 'k-', lw=0.6)
    ax.plot([70, 70], [84, 400], 'k-', lw=0.6)
    ax.plot([0, 70], [180, 180], 'k-', lw=0.6)
    ax.plot([70, 290], [180, 400], 'k-', lw=0.6)
    ax.plot([70, 70], [0, 56], 'k-', lw=0.6)
    ax.plot([70, 400], [56, 320], 'k-', lw=0.6)
    ax.plot([180, 180], [0, 70], 'k-', lw=0.6)
    ax.plot([180, 400], [70, 70], 'k-', lw=0.6)
    ax.plot([240, 240], [70, 180], 'k-', lw=0.6)
    ax.plot([240, 400], [180, 180], 'k-', lw=0.6)
    ax.plot([130, 180], [0, 70], 'k-', lw=0.6)

    zones = np.array([_clarke_zone(r, p) for r, p in zip(ref, pred)])
    zcol = {'A': C['green'], 'B': C['sky'], 'C': C['yellow'],
            'D': C['orange'], 'E': C['red']}
    for z in 'ABCDE':
        m = zones == z
        if m.any():
            ax.scatter(ref[m], pred[m], s=35, color=zcol[z], edgecolor='black',
                       linewidth=0.4, label=f'Zone {z} ({100*m.mean():.1f}%)', zorder=3)

    for lbl, xx, yy in [('A', 130, 130), ('B', 260, 230), ('C', 100, 300),
                        ('D', 340, 120), ('E', 30, 300)]:
        ax.text(xx, yy, lbl, fontsize=13, fontweight='bold', color='black', alpha=0.35)

    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel('Reference glucose (mg/dL)')
    ax.set_ylabel('Predicted glucose (mg/dL)')
    ax.set_title('Clarke Error Grid (event-level, LOSO)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    _save(fig, out_dir, 'fig6_clarke_grid')


# =============================================================================
# Fig 7 - predicted vs true
# =============================================================================

def fig_pred_vs_true(npz_path, out_dir):
    if not os.path.exists(npz_path):
        return
    dat = np.load(npz_path, allow_pickle=True)
    ref, pred = dat['y_true'].astype(float), dat['y_pred'].astype(float)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    mn, mx = min(ref.min(), pred.min()) - 10, max(ref.max(), pred.max()) + 10
    ax.plot([mn, mx], [mn, mx], color=C['grey'], ls='--', lw=1, label='Identity')
    ax.scatter(ref, pred, s=40, color=C['blue'], edgecolor='black',
               linewidth=0.4, alpha=0.8, zorder=3)
    ax.axhline(np.mean(ref), color=C['red'], lw=1, ls=':',
               label=f'Predicting mean ({np.mean(ref):.0f})')
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel('Reference glucose (mg/dL)')
    ax.set_ylabel('Predicted glucose (mg/dL)')
    ax.set_title('Predicted vs reference (event-level)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    _save(fig, out_dir, 'fig7_pred_vs_true')


# =============================================================================
# Fig 8 - feature selection frequency
# =============================================================================

def fig_feature_selection(fs, out_dir, top=20):
    if fs is None:
        return
    col = 'selection_frequency' if 'selection_frequency' in fs.columns else fs.columns[-1]
    name_col = 'index' if 'index' in fs.columns else fs.columns[0]
    d = fs.sort_values(col, ascending=False).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(d))
    ax.barh(y, d[col], color=C['purple'], edgecolor='black', linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(d[name_col], fontsize=8)
    ax.set_xlabel('Selection frequency across LOSO folds')
    ax.set_title(f'Top {top} features by RFECV selection stability')
    ax.set_xlim(0, 1)
    _save(fig, out_dir, 'fig8_feature_selection')


# =============================================================================
# Driver
# =============================================================================

def fig_lag_overlap(lag_glob, out_dir):
    """
    Pairwise shared-event heatmap across lag configurations.

    For a zero-common-support sweep, a MARD-vs-tau curve is misleading because
    the arms are not comparable. This overlap matrix shows overlap collapsing to
    zero beyond one step, proving the arms align to disjoint reference sets.
    Reconstructs the matrix from the tau pkl files; no stored matrix needed.
    """
    import glob as _glob
    import pickle as _pkl
    paths = sorted(_glob.glob(lag_glob))
    if len(paths) < 2:
        print("  (skip lag overlap: need >=2 tau pkl files)")
        return

    arms = []
    for p in paths:
        with open(p, 'rb') as f:
            d = _pkl.load(f)
        arms.append((d['meta']['tau_minutes'], set(d['glucose_events'])))
    arms.sort(key=lambda t: t[0])
    taus = [t for t, _ in arms]
    sets = [s for _, s in arms]
    n = len(taus)

    M = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            M[i, j] = len(sets[i] & sets[j])
    common_all = set.intersection(*sets)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(M, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f'{t:+g}' for t in taus])
    ax.set_yticklabels([f'{t:+g}' for t in taus])
    ax.set_xlabel('Lag τ (min)'); ax.set_ylabel('Lag τ (min)')
    ax.set_title('Shared glucose events between lag configurations\n'
                 '(diagonal = events per arm; off-diagonal → 0)', fontsize=11)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, str(v), ha='center', va='center',
                    color='white' if v > M.max() * 0.6 else 'black',
                    fontsize=9, fontweight='bold' if i == j else 'normal')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('shared events')
    ax.text(0.5, -0.20,
            f'Common support across all {n} arms = {len(common_all)}. '
            'Overlap vanishes beyond one step,\nso lag settings align to disjoint '
            'reference sets — cross-lag comparison is ill-posed.',
            transform=ax.transAxes, ha='center', fontsize=7.8,
            style='italic', color='#555555')
    _save(fig, out_dir, 'fig9_lag_overlap')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out-dir', default='figures')
    ap.add_argument('--lag-glob', default='preprocessed_data_tau*.pkl',
                    help='glob for tau pkl files (lag overlap figure)')
    ap.add_argument('--cohort-labels', default='T1D,Healthy',
                    help="legend labels for the two cohorts, comma separated. "
                         "Use 'Cohort c1,Cohort c2' for PhysioCGM, whose group "
                         "prefixes denote recording cohorts, not disease status.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    R = args.results_dir

    pooled = None
    pj = os.path.join(R, 'pooled_metrics.json')
    if os.path.exists(pj):
        with open(pj) as f:
            pooled = json.load(f)

    print("generating figures...")
    fig_split_severity(_read_csv(R, 'split_severity_ablation.csv'), args.out_dir)
    fig_leakage_proof(_read_csv(R, 'label_permutation.csv'), pooled, args.out_dir)
    fig_baseline_ladder(_read_csv(R, 'baseline_ladder.csv'), pooled, args.out_dir)
    fig_per_subject(_read_csv(R, 'per_subject.csv'), args.out_dir,
                    cohort_labels=tuple(args.cohort_labels.split(',', 1)))
    fig_glycemic_range(_read_csv(R, 'by_glycemic_range.csv'), args.out_dir)
    fig_clarke_grid(os.path.join(R, 'event_predictions.npz'), args.out_dir)
    fig_pred_vs_true(os.path.join(R, 'event_predictions.npz'), args.out_dir)
    fig_feature_selection(_read_csv(R, 'feature_selection_frequency.csv'), args.out_dir)
    fig_lag_overlap(args.lag_glob, args.out_dir)
    print(f"\ndone -> {args.out_dir}/  (PNG for preview, PDF for submission)")


if __name__ == '__main__':
    main()
