#!/usr/bin/env python3
"""
roc_disc_ana.py
===============
ROC curves and discriminator distributions from the plotting .npz, drawn with the
HOUSE drawers in plot_roc_curves.py / plot_disc_comparison.py so every figure keeps
the same style (LDMX header, mass colour pairs, logx + linear).

It also records, for every model and mass, the signal efficiency at a list of
background-efficiency working points (1e-6, 1e-5, 1e-4, 1e-3 by default) plus the
AUC, and writes them to `<OUT>/metrics.csv` + `<OUT>/metrics.npz`.
panel_ana.py turns those into the N-panel summary, so the heavy ROC work happens
once here and the panel plot is instant to re-style.

Outputs under OUTDIR:
    disc_linear_<model>.png / disc_neglog_<model>.png   bkg + 4 masses, normalized
    <A>_vs_<B>/ROC_combined_bdt_{logx,linear}.png       4 masses x 2 models (the 4v4)
    <A>_vs_<B>/roc_<mass>/ROC_bdt_comparison_*.png      per-mass head-to-head
    metrics.csv, metrics.npz

Usage:
    denv python3 roc_disc_ana.py --bkg bkg_plot.npz --sig sig_plot.npz --out plots/
"""
import argparse, csv, os, sys
import numpy as np
from sklearn import metrics as skm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ROOT as r
import plot_roc_curves as prc
import plot_disc_comparison as pdc

# ============================== CONFIG ======================================
# Which models to plot, and the label each gets in legends. Leave MODELS empty
# to use every model found in BOTH npz files.
MODELS = {}                      # e.g. {'V15R_base': 'Humberto', 'H3_s202501': 'Humberto (new)'}

# Pairs for the 4v4 ROC canvases. Empty = every consecutive pair of MODELS.
# The FIRST of each pair is drawn in the darker shade.
PAIRS = []                       # e.g. [('V15R_base', 'H3_s202501')]

# Background-efficiency working points recorded for the panel plot.
# A point needs ~10/eff background events to be meaningful: 1e-6 wants >1e7.
WORKING_POINTS = [1e-6, 1e-5, 1e-4, 1e-3]

# Event selection applied to BOTH signal and background before anything is drawn.
SEL_TRIGGER      = True
SEL_FIDUCIAL     = True    # keep True: non-fiducial events carry a degenerate
                           # all-zero feature vector and a per-model constant disc,
                           # which otherwise dominates the deep tail
SEL_PRESELECTION = True
SEL_TRACKER_VETO = True
SEL_NSTRAIGHT    = None    # e.g. 3 to also require n_straight < 3 (post-BDT cut)
SEL_HCAL         = False   # True to also require the HCal veto (post-BDT cut)

MASSES = [('1 MeV', '1MeV'), ('10 MeV', '10MeV'), ('100 MeV', '100MeV'), ('1000 MeV', '1000MeV')]
# ============================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--bkg', required=True)
ap.add_argument('--sig', required=True)
ap.add_argument('--out', default='plots')
ap.add_argument('--models', default=None,
                help="override the MODELS config: 'key=Label,key2=Label2' or just 'key,key2'")
ap.add_argument('--max-bkg', type=int, default=None,
                help='randomly subsample background to this many events (speeds up ROC; '
                     'do NOT use when quoting 1e-6 working points)')
a = ap.parse_args()
r.gROOT.SetBatch(True)
os.makedirs(a.out, exist_ok=True)


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


B, S = load(a.bkg), load(a.sig)
bm = set(str(x) for x in B['_models'])
sm = set(str(x) for x in S['_models'])
if a.models:
    models = {}
    for tok in a.models.split(','):
        tok = tok.strip()
        if not tok:
            continue
        k, _, lab = tok.partition('=')
        models[k.strip()] = (lab.strip() or k.strip())
else:
    models = MODELS or {k: k for k in sorted(bm & sm)}
missing = [k for k in models if k not in bm or k not in sm]
if missing:
    raise SystemExit('model(s) %s not in both npz files\n  bkg: %s\n  sig: %s'
                     % (missing, sorted(bm), sorted(sm)))


def selection(d):
    m = np.ones(len(d['trig']), bool)
    if SEL_TRIGGER:      m &= d['trig'].astype(bool)
    if SEL_FIDUCIAL:     m &= d['fiducial'].astype(bool)
    if SEL_PRESELECTION: m &= d['presel'].astype(bool)
    if SEL_TRACKER_VETO: m &= d['trk'].astype(bool)
    if SEL_NSTRAIGHT is not None: m &= d['nstr'] < SEL_NSTRAIGHT
    if SEL_HCAL:         m &= d['hcal_pass'].astype(bool)
    return m


mb, ms = selection(B), selection(S)
if a.max_bkg and mb.sum() > a.max_bkg:
    idx = np.where(mb)[0]
    drop = np.random.default_rng(0).choice(idx, len(idx) - a.max_bkg, replace=False)
    mb[drop] = False
sf = [str(x) for x in S['_files']]
smass = {mk: np.array(['_%s_' % mk in f for f in sf])[S['fileidx']] & ms for _l, mk in MASSES}

bkg = {lab: B['disc_' + k][mb] for k, lab in models.items()}
sig = {lab: {ml: S['disc_' + k][smass[mk]] for ml, mk in MASSES} for k, lab in models.items()}
cuts = [n for n, on in [('trigger', SEL_TRIGGER), ('fiducial', SEL_FIDUCIAL),
                        ('preselection', SEL_PRESELECTION), ('tracker veto', SEL_TRACKER_VETO),
                        ('n_straight<%s' % SEL_NSTRAIGHT, SEL_NSTRAIGHT is not None),
                        ('hcal veto', SEL_HCAL)] if on]
print('selection: %s' % (' + '.join(cuts) if cuts else 'NONE'))
print('background: %s events' % format(int(mb.sum()), ','))
print('signal    : %s' % ', '.join('%s %s' % (ml, format(int(smass[mk].sum()), ',')) for ml, mk in MASSES))
print('models    : %s\n' % ', '.join(models.values()))
worst = min(WORKING_POINTS)
if mb.sum() * worst < 10:
    print('*** WARNING: %s background events support a %g working point only to +-%.0f%% '
          '(expect ~%.1f events). Treat it as indicative. ***\n'
          % (format(int(mb.sum()), ','), worst, 100 / max(np.sqrt(mb.sum() * worst), 1e-9),
             mb.sum() * worst))

# ------------------------------------------------------ disc distributions
for lab in models.values():
    lin, neg, names = [], [], []
    for key, nice, d in [('bkg', 'Ecal PN (bkg)', bkg[lab])] + \
                        [(mk, '%s Signal' % ml, sig[lab][ml]) for ml, mk in MASSES]:
        h = r.TH1F('d_%s_%s' % (lab, key), '', 100, 0, 1)
        pdc._fill_th1_from_array(h, d)
        hl = r.TH1F('l_%s_%s' % (lab, key), '', 100, 0, 7)
        pdc._fill_th1_from_array(hl, -np.log10(1.0 - d[d < 1.0].astype('f8')))
        lin.append(h); neg.append(hl); names.append(nice)
    pdc.drawMultHists(lin, names, 'BDT Discriminator Score',
                      os.path.join(a.out, 'disc_linear_%s' % lab), lab,
                      scale=True, log=True, ytitle='Normalized Events', legendPOS='upper left')
    pdc.drawMultHists(neg, names, '-log_{10}(1 - BDT Disc Score)',
                      os.path.join(a.out, 'disc_neglog_%s' % lab), lab,
                      scale=True, log=True, ytitle='Normalized Events', legendPOS='upper left')

# ------------------------------------------------------------------- ROC
labs = list(models.values())
pairs = [(models[x], models[y]) for x, y in PAIRS] if PAIRS else \
        [(labs[i], labs[i + 1]) for i in range(len(labs) - 1)]
for A, Bl in pairs:
    od = os.path.join(a.out, '%s_vs_%s' % (A.replace(' ', '_'), Bl.replace(' ', '_')))
    os.makedirs(od, exist_ok=True)
    bb = {A: bkg[A], Bl: bkg[Bl]}
    ss = {A: sig[A], Bl: sig[Bl]}
    prc.plot_combined_bdt_roc(bb, ss, {A: A, Bl: Bl}, dict(MASSES), od)
    for ml, mk in MASSES:
        prc.plot_bdt_comparison_roc(bb, {A: sig[A][ml], Bl: sig[Bl][ml]}, ml,
                                    os.path.join(od, 'roc_%s' % mk))

# --------------------------------------------------- metrics for the panels
rows = []
print('\n%-22s %-9s %10s %s' % ('model', 'mass', 'AUC',
                                ' '.join('%11s' % ('@%g' % w) for w in WORKING_POINTS)))
for lab in labs:
    b = bkg[lab]
    thr = {w: np.quantile(b, 1 - w) for w in WORKING_POINTS}
    for ml, mk in MASSES:
        s = sig[lab][ml]
        y = np.concatenate([np.zeros(len(b)), np.ones(len(s))])
        fpr, tpr, _ = skm.roc_curve(y, np.concatenate([b, s]))
        auc = skm.auc(fpr, tpr)
        eff = {w: float((s > thr[w]).mean()) for w in WORKING_POINTS}
        rows.append(dict(model=lab, mass=ml, mass_mev=int(mk.replace('MeV', '')), auc=auc,
                         **{'eff_%g' % w: eff[w] for w in WORKING_POINTS}))
        print('%-22s %-9s %10.5f %s' % (lab, ml, auc, ' '.join('%11.4f' % eff[w] for w in WORKING_POINTS)))

with open(os.path.join(a.out, 'metrics.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
np.savez(os.path.join(a.out, 'metrics.npz'),
         models=np.array(labs), masses=np.array([m for m, _ in MASSES]),
         working_points=np.array(WORKING_POINTS), selection=np.array(cuts),
         **{k: np.array([r[k] for r in rows]) for k in rows[0]})
print('\nwrote %s/metrics.csv and metrics.npz  -> feed to panel_ana.py' % a.out)
