#!/usr/bin/env python3
"""
panel_ana.py
============
N-panel summary vs A' mass, built from the metrics roc_disc_ana.py wrote.
Each panel is one metric; you switch panels on and off in the CONFIG block, so
the same script makes a 2-, 3-, 4- or 5-panel figure without editing any plotting
code. Panels appear left to right in the order listed.

    PANELS = ['auc', 1e-4, 1e-3]            -> 3 panels
    PANELS = ['auc', 1e-6, 1e-5, 1e-4, 1e-3] -> 5 panels

Usage:
    denv python3 panel_ana.py --metrics plots/metrics.npz --out plots/panels.png
    denv python3 panel_ana.py --metrics plots/metrics.csv --out p.png   # csv works too
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================== CONFIG ======================================
# Panels to draw, left to right. 'auc' or a background-efficiency working point.
PANELS = ['auc', 1e-4, 1e-3]

# Models to show, and the order of the legend. Empty = all models in the metrics.
MODELS = []                       # e.g. ['Humberto', 'Humberto (new)']

# Per-model style. Anything missing falls back to the matplotlib cycle.
STYLE = {
    'Humberto':       dict(color='tab:blue',   marker='s'),
    'Humberto (new)': dict(color='tab:red',    marker='o'),
    'SegMip':         dict(color='tab:green',  marker='^'),
    'Recoil':         dict(color='tab:purple', marker='D'),
}

# Fallback palette for models not listed in STYLE. Without this, matplotlib's
# colour cycle restarts on every axes and an unstyled model silently draws in the
# same colour as a styled one.
FALLBACK_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange',
                   'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'black']
FALLBACK_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

TITLES = {'auc': 'ROC AUC'}       # working points get 'Signal eff @ bkg eff 1e-4' etc.
SUPTITLE = None                   # None = auto from the selection recorded in the metrics
LOG_X    = True                   # mass axis on a log scale
FIGSIZE_PER_PANEL = (5.3, 5.2)
# ============================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--metrics', required=True, help='metrics.npz or metrics.csv from roc_disc_ana.py')
ap.add_argument('--out', default='panels.png')
a = ap.parse_args()

# ----------------------------------------------------------------- load
if a.metrics.endswith('.npz'):
    z = np.load(a.metrics, allow_pickle=True)
    cols = {k: z[k] for k in z.files}
    rows = [{k: cols[k][i] for k in cols if cols[k].ndim == 1 and len(cols[k]) == len(cols['model'])}
            for i in range(len(cols['model']))]
    available = [float(x) for x in cols['working_points']]
    selection = [str(x) for x in cols['selection']] if 'selection' in cols else []
else:
    rows = list(csv.DictReader(open(a.metrics)))
    for r_ in rows:
        r_['auc'] = float(r_['auc']); r_['mass_mev'] = int(r_['mass_mev'])
        for k in list(r_):
            if k.startswith('eff_'):
                r_[k] = float(r_[k])
    available = sorted({float(k[4:]) for k in rows[0] if k.startswith('eff_')})
    selection = []

models = MODELS or sorted({str(r_['model']) for r_ in rows}, key=lambda m: [str(x) for x in
          [r_['model'] for r_ in rows]].index(m))
masses = sorted({int(r_['mass_mev']) for r_ in rows})


def column(p):
    if p == 'auc':
        return 'auc'
    key = 'eff_%g' % p
    if key not in rows[0]:
        raise SystemExit('working point %g was not recorded.\nAvailable: %s\n'
                         'Add it to WORKING_POINTS in roc_disc_ana.py and re-run that first.'
                         % (p, ', '.join('%g' % x for x in available)))
    return key


def title(p):
    if p in TITLES:
        return TITLES[p]
    return 'ROC AUC' if p == 'auc' else 'Signal eff @ bkg eff %g' % p


panels = list(PANELS)
if not panels:
    raise SystemExit('PANELS is empty -- nothing to draw')
cols_ = [column(p) for p in panels]

fig, axes = plt.subplots(1, len(panels), figsize=(FIGSIZE_PER_PANEL[0] * len(panels),
                                                 FIGSIZE_PER_PANEL[1]))
if len(panels) == 1:
    axes = [axes]
for ax, p, col in zip(axes, panels, cols_):
    for m in models:
        pts = sorted([(int(r_['mass_mev']), float(r_[col])) for r_ in rows if str(r_['model']) == m])
        if not pts:
            continue
        x = [q[0] for q in pts]; y = [q[1] for q in pts]
        i = models.index(m)
        st = STYLE.get(m, {})
        # NB: do not name these 'col' -- that is the metric column of the outer loop
        mcolor = st.get('color', FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        mmark = st.get('marker', FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)])
        ax.plot(x, y, mmark + '-', color=mcolor, lw=2, ms=7, label=m)
    if LOG_X:
        ax.set_xscale('log')
    ax.set_xticks(masses); ax.set_xticklabels([str(x) for x in masses])
    ax.set_xlabel("A' mass [MeV]")
    ax.set_title(title(p), fontsize=11)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('metric value')
axes[-1].legend(fontsize=9, loc='lower right')
sup = SUPTITLE if SUPTITLE is not None else (
    'BDT comparison vs mass' + ('  (%s)' % ' + '.join(selection) if selection else ''))
fig.suptitle(sup, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
od = os.path.dirname(os.path.abspath(a.out))
if od:
    os.makedirs(od, exist_ok=True)
fig.savefig(a.out, dpi=150, bbox_inches='tight')
print('wrote %s  (%d panels: %s)' % (a.out, len(panels),
      ', '.join(str(p) for p in panels)))
print('models: %s' % ', '.join(models))
