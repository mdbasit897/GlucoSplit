#!/usr/bin/env python3
"""
Phase 4 figures - cross-dataset transfer, standalone.
  - Direction labels cleaned to readable names (no raw pickle filenames).
  - Permutation CSVs matched robustly; a missing file is annotated, not
    silently dropped.
  - fig11 title/message adapts: no longer claims "collapses to 0" when transfer
    bars are strongly negative. States the honest two-part finding.

Usage
-----
    python phase4_figures.py --transfer-dir transfer_results \
        --d1namo-pooled results/pooled_metrics.json \
        --physiocgm-pooled PhysioCGM/results/pooled_metrics.json \
        --out-dir figures
"""

import argparse, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
     'red': '#D55E00', 'grey': '#999999', 'sky': '#56B4E9'}


def _save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _read_csv(p): return pd.read_csv(p) if os.path.exists(p) else None
def _read_json(p):
    if p and os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None


def _clean_direction(raw, na, nb):
    src, _, tgt = raw.partition('->')
    def pretty(s):
        return (nb or 'PhysioCGM') if 'physiocgm' in s.strip().lower() else (na or 'D1NAMO')
    return f"{pretty(src)} → {pretty(tgt)}"


def _find_perm(tdir, raw_direction):
    src_is_pcgm = 'physiocgm' in raw_direction.partition('->')[0].lower()
    f = 'transfer_perm_physiocgm_to_d1namo.csv' if src_is_pcgm \
        else 'transfer_perm_d1namo_to_physiocgm.csv'
    df = _read_csv(os.path.join(tdir, f))
    if df is not None and {'r2', 'mard'}.issubset(df.columns):
        return df['r2'].mean(), df['mard'].mean()
    return np.nan, np.nan


def fig_transfer(tdir, out_dir, na, nb):
    summ = _read_csv(os.path.join(tdir, 'transfer_summary.csv'))
    if summ is None:
        print("  (skip fig10: transfer_summary.csv not found)"); return
    directions = list(dict.fromkeys(summ['direction']))
    rows = []
    for d in directions:
        t = summ[(summ['direction'] == d) & (summ['kind'] == 'transfer')]
        b = summ[(summ['direction'] == d) & (summ['kind'] == 'baseline_source_mean')]
        pr2, pmard = _find_perm(tdir, d)
        rows.append({'label': _clean_direction(d, na, nb),
                     'transfer_r2': float(t['r2'].iloc[0]) if len(t) else np.nan,
                     'baseline_r2': float(b['r2'].iloc[0]) if len(b) else np.nan,
                     'perm_r2': pr2,
                     'transfer_mard': float(t['mard'].iloc[0]) if len(t) else np.nan,
                     'baseline_mard': float(b['mard'].iloc[0]) if len(b) else np.nan,
                     'perm_mard': pmard})
    D = pd.DataFrame(rows)
    have_perm = not D['perm_r2'].isna().all()
    x = np.arange(len(D)); w = 0.26 if have_perm else 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    def _panel(ax, tc, bc, pc, ylabel, title, tline=None, tlbl=None):
        off = -w if have_perm else -w / 2
        ax.bar(x + off, D[tc], w, label='Transfer model', color=C['red'],
               edgecolor='black', linewidth=0.5)
        ax.bar(x + (0 if have_perm else w / 2), D[bc], w, label='Source-mean baseline',
               color=C['grey'], edgecolor='black', linewidth=0.5)
        if have_perm:
            ax.bar(x + w, D[pc], w, label='Permutation floor', color=C['sky'],
                   edgecolor='black', linewidth=0.5, hatch='//')
        if tline is not None:
            ax.axhline(tline, color=C['green'], ls='--', lw=1, label=tlbl)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(D['label'], fontsize=9)
        ax.legend(fontsize=8, loc='best')

    _panel(ax1, 'transfer_r2', 'baseline_r2', 'perm_r2', 'R²',
           '(a) Transfer R² — higher is better')
    ax1.axhline(0, color='black', lw=0.8)
    _panel(ax2, 'transfer_mard', 'baseline_mard', 'perm_mard', 'MARD (%)',
           '(b) Transfer MARD — lower is better', tline=15, tlbl='Clinical target (15%)')
    if not have_perm:
        fig.text(0.5, -0.02, 'Permutation-floor bars omitted (permutation CSVs not found).',
                 ha='center', fontsize=7.5, style='italic', color=C['grey'])
    fig.suptitle('Cross-dataset transfer performs worse than a source-mean baseline',
                 fontsize=12, fontweight='bold', y=1.03)
    _save(fig, out_dir, 'fig10_transfer')
    return D


def fig_summary(tdir, d1p, pcp, out_dir, na, nb):
    summ = _read_csv(os.path.join(tdir, 'transfer_summary.csv'))
    d1 = _read_json(d1p); pc = _read_json(pcp)
    labels, r2s, mards, colors = [], [], [], []
    if d1:
        labels.append(f'{na}\n(within, LOSO)')
        r2s.append(d1['pooled_event']['r2']); mards.append(d1['pooled_event']['mard']); colors.append(C['blue'])
    if pc:
        labels.append(f'{nb}\n(within, LOSO)')
        r2s.append(pc['pooled_event']['r2']); mards.append(pc['pooled_event']['mard']); colors.append(C['green'])
    if summ is not None:
        for d in dict.fromkeys(summ['direction']):
            t = summ[(summ['direction'] == d) & (summ['kind'] == 'transfer')]
            if len(t):
                labels.append(_clean_direction(d, na, nb).replace(' → ', '\n→ '))
                r2s.append(float(t['r2'].iloc[0])); mards.append(float(t['mard'].iloc[0])); colors.append(C['red'])
    if not labels:
        print("  (skip fig11: no data)"); return
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(x, r2s, color=colors, edgecolor='black', linewidth=0.6)
    ax1.axhline(0, color='black', lw=0.8); ax1.set_ylabel('R²')
    ax1.set_title('(a) Within-dataset R² ≈ 0; cross-dataset transfer is negative')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(r2s):
        ax1.text(i, v + (0.01 if v >= 0 else -0.015), f'{v:+.3f}',
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax2.bar(x, mards, color=colors, edgecolor='black', linewidth=0.6)
    ax2.axhline(15, color=C['green'], ls='--', lw=1, label='Clinical target (15%)')
    ax2.set_ylabel('MARD (%)'); ax2.set_title('(b) MARD across all evaluations')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8); ax2.legend(fontsize=8)
    for i, v in enumerate(mards):
        ax2.text(i, v + 0.6, f'{v:.1f}', ha='center', fontsize=8)
    cat = [Patch(facecolor=C['blue'], label=f'{na} within'),
           Patch(facecolor=C['green'], label=f'{nb} within'),
           Patch(facecolor=C['red'], label='Cross-dataset transfer')]
    ax1.legend(handles=cat, fontsize=7.5, loc='lower left')
    fig.suptitle('No recoverable ECG–glucose signal within OR across cohorts',
                 fontsize=12, fontweight='bold', y=1.03)
    _save(fig, out_dir, 'fig11_evaluation_summary')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transfer-dir', default='transfer_results')
    ap.add_argument('--d1namo-pooled', default='results/pooled_metrics.json')
    ap.add_argument('--physiocgm-pooled', default='PhysioCGM/results/pooled_metrics.json')
    ap.add_argument('--source-name-a', dest='na', default='D1NAMO')
    ap.add_argument('--target-name-a', dest='nb', default='PhysioCGM')
    ap.add_argument('--out-dir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print("generating phase-4 figures...")
    fig_transfer(args.transfer_dir, args.out_dir, args.na, args.nb)
    fig_summary(args.transfer_dir, args.d1namo_pooled, args.physiocgm_pooled,
                args.out_dir, args.na, args.nb)
    print(f"\ndone -> {args.out_dir}/")


if __name__ == '__main__':
    main()
