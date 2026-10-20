#!/usr/bin/env python3
"""
cutflow_table_ana.py
====================
Cutflow from the .npz written by extract_cutflow_npz.py. Pure numpy -- no ROOT,
no uproot -- so it runs in seconds and can be re-run freely while tuning cuts.

Handles ANY number of BDTs at once. Each model gets its OWN disc cut, set in the
CONFIG block below (not on the command line), because every model has its own
operating point and mixing them up is the easiest way to publish a wrong number.

Prints the table and writes three CSVs:
    <out>_cutflow.csv    one row per (model, cut): bkg count, sig eff per mass
    <out>_summary.csv    the FINAL row of every model, side by side
    <out>_survivors.csv  every surviving background event, with disc under ALL models

Usage:
    denv python3 cutflow_table_ana.py --bkg bkgfull.npz --sig sig.npz --out myrun
    denv python3 cutflow_table_ana.py --bkg b.npz --sig s.npz --scan Humberto   # pick a cut

--scan prints, for one model, the disc cut that would leave n = 0,1,2,... background
events after every other cut. Use it to fill in MODEL_CUTS below.
"""
import argparse, csv, os
import numpy as np

# ============================== CONFIG ======================================
# Disc cut per model. Keys are model names as stored in the npz (the EcalVeto
# branch minus 'EcalVeto_' and '_BDT'); aliases on the right are also accepted.
# Models present in the npz but absent here fall back to DEFAULT_CUT.

MODEL_CUTS = {
    'H3_s202501': 0.9546506405,   # Humberto
    'V15R_base':  0.9540773034,   # Humberto (old order)
    'reco_v1':    0.99741,        # SegMip
}
DEFAULT_CUT = 0.95

FIDUCIAL   = True      # include the Fiducial row and use it as the 100% denominator
HCAL_MODE  = 'flag'    
MAXPE      = 8.0
NSTRAIGHT  = 3         # keep events with n_straight_tracks < NSTRAIGHT
MAX_SURVIVORS_PRINTED = 40
# ============================================================================

MASSES = ['1MeV', '10MeV', '100MeV', '1000MeV']
ap = argparse.ArgumentParser()
ap.add_argument('--bkg', required=True)
ap.add_argument('--sig', required=True)
ap.add_argument('--out', default=None, help='CSV prefix (omit to skip CSVs)')
ap.add_argument('--outdir', default='.')
ap.add_argument('--models', default=None,
                help='comma-separated model names to include (default: every model in both npz files)')
ap.add_argument('--scan', default=None, help='model name: print bkg count vs disc cut, then exit')
a = ap.parse_args()

DENOM_ROW = 'Fiducial' if FIDUCIAL else 'Pre-selection skim'


def materialise(path):
    """np.load on an .npz is LAZY: every z[key] re-decompresses that whole array.
    The cut masks touch the same arrays once per model per cut, which turns a
    seconds-long job into minutes of repeated DEFLATE. Read each array once."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


B, S = materialise(a.bkg), materialise(a.sig)
bm = set(str(x) for x in B['_models'])
sm = set(str(x) for x in S['_models'])

# Pair up models present in BOTH files. Auto-discovers everything, so a newly
# added BDT is never silently dropped from the table.
pairs = []
for k in sorted(bm & sm):
    pairs.append((k, k, k))
# SegMip is 'reco' in delivered background and 'reco_v1' in anything we reco'd.
if not (bm & sm):
    sb = next((k for k in ('reco', 'reco_v1', 'SegMip') if k in bm), None)
    ss = next((k for k in ('reco', 'reco_v1', 'SegMip') if k in sm), None)
    if sb and ss:
        pairs.append(('SegMip', sb, ss))
else:
    sb = next((k for k in ('reco', 'reco_v1') if k in bm), None)
    ss = next((k for k in ('reco', 'reco_v1') if k in sm), None)
    if sb and ss and sb != ss:
        pairs.append(('SegMip', sb, ss))
if a.models:
    want = [x.strip() for x in a.models.split(',') if x.strip()]
    pairs = [q for q in pairs if q[0] in want or q[1] in want]
    missing = [w for w in want if not any(w in (q[0], q[1]) for q in pairs)]
    if missing:
        raise SystemExit('requested model(s) %s not in both npz files; available: %s'
                         % (missing, sorted(bm & sm)))
if not pairs:
    raise SystemExit('no model is present in both npz files\n  bkg: %s\n  sig: %s'
                     % (sorted(bm), sorted(sm)))


def cut_for(name, bkey):
    for k in (name, bkey):
        if k in MODEL_CUTS:
            return MODEL_CUTS[k]
    return DEFAULT_CUT


def hc(d):
    return d['hcal_pass'] if HCAL_MODE == 'flag' else (d['maxpe'] < MAXPE)


fl = [str(x) for x in S['_files']]
sig_mass = {m: np.array([('_%s_' % m) in fl[i] for i in range(len(fl))])[S['fileidx']] for m in MASSES}
bfiles = [str(x) for x in B['_files']]
nb0 = len(B['trig'])

print('bkg %s events (%s)' % (format(nb0, ','), os.path.basename(a.bkg)))
print('sig %s events (%s)' % (format(len(S['trig']), ','), os.path.basename(a.sig)))
print('models: %s' % ', '.join('%s @ %.10g' % (p[0], cut_for(p[0], p[1])) for p in pairs))
print('fiducial=%s  hcal=%s  nstraight<%d\n' % (FIDUCIAL, HCAL_MODE, NSTRAIGHT))

# ------------------------------------------------------------------ scan mode
if a.scan:
    kb = next((p[1] for p in pairs if p[0] == a.scan or p[1] == a.scan), a.scan)
    if 'disc_' + kb not in B:
        raise SystemExit('model %r not in background npz (have %s)' % (a.scan, sorted(bm)))
    fixed = B['trig'] & B['presel'] & B['trk'] & (B['nstr'] < NSTRAIGHT) & hc(B)
    if FIDUCIAL:
        fixed &= B['fiducial']
    d = np.sort(B['disc_' + kb][fixed])[::-1]
    print('=== %s: %d background events survive all non-disc cuts ===' % (a.scan, len(d)))
    print('%6s %18s' % ('n left', 'disc cut'))
    for i, v in enumerate(d[:30]):
        print('%6d %18.10f' % (i, np.nextafter(v, 1.0)))
    raise SystemExit

# ------------------------------------------------------------------- cutflow
rows, summary, survivors = [], [], []
for disp, kb, ks in pairs:
    THR = cut_for(disp, kb)
    CUTS = [('Triggered', lambda d, k: d['trig'])]
    if FIDUCIAL:
        CUTS.append(('Fiducial', lambda d, k: d['fiducial']))
    CUTS += [('Pre-selection skim', lambda d, k: d['presel']),
             ('Tracker veto', lambda d, k: d['trk']),
             ('Ecal veto (>=%.7g)' % THR, lambda d, k: d['disc_' + k] >= THR),
             ('N_straight < %d' % NSTRAIGHT, lambda d, k: d['nstr'] < NSTRAIGHT),
             (('HCal maxPE < %g' % MAXPE) if HCAL_MODE == 'maxpe' else 'HCal veto (passes_veto_)',
              lambda d, k: hc(d))]

    print('=' * 96)
    print('%s      disc >= %.10g' % (disp, THR))
    print('=' * 96)
    hdr = '%-26s | %14s | %s' % ('Cut', 'bkg', '  '.join('%9s' % m for m in MASSES))
    print(hdr); print('-' * len(hdr))

    mb = np.ones(nb0, bool)
    ms = {m: sig_mass[m].copy() for m in MASSES}
    den = {m: max(int(sig_mass[m].sum()), 1) for m in MASSES}
    for nm, fn in CUTS:
        mb &= fn(B, kb)
        line = '%-26s | %14s |' % (nm, format(int(mb.sum()), ','))
        rec = {'model': disp, 'disc_cut': '%.10g' % THR, 'cut': nm, 'bkg_remaining': int(mb.sum())}
        for m in MASSES:
            ms[m] &= fn(S, ks)
            if nm == DENOM_ROW:
                den[m] = max(int(ms[m].sum()), 1)
            eff = 100.0 * ms[m].sum() / den[m]
            line += '  %8.1f%%' % eff
            rec['sig_eff_' + m] = round(eff, 3)
        print(line)
        rows.append(rec)
    last = dict(rows[-1]); last['cut'] = 'FINAL'
    summary.append(last)

    surv = np.where(mb)[0]
    surv = surv[np.argsort(-B['disc_' + kb][surv])]
    print('\n  surviving background: %d' % len(surv))
    for i, j in enumerate(surv[:MAX_SURVIVORS_PRINTED]):
        print('   %4d disc=%.8f run=%-11s evt=%-10s Nstr=%d maxPE=%.1f %s'
              % (i + 1, B['disc_' + kb][j], B['run'][j], B['evt'][j],
                 B['nstr'][j], B['maxpe'][j], bfiles[B['fileidx'][j]][:44]))
    if len(surv) > MAX_SURVIVORS_PRINTED:
        print('   ... %d more' % (len(surv) - MAX_SURVIVORS_PRINTED))
    for j in surv:
        rec = {'model': disp, 'run': int(B['run'][j]), 'event': int(B['evt'][j]),
               'n_straight': int(B['nstr'][j]), 'maxPE': float(B['maxpe'][j]),
               'hcal_pass': bool(B['hcal_pass'][j]), 'file': bfiles[B['fileidx'][j]]}
        for d2 in sorted(bm):
            rec['disc_' + d2] = float(B['disc_' + d2][j])
        survivors.append(rec)

    # the cuts that would leave exactly 1 and 0 background events, with sig eff
    fixed = B['trig'] & B['presel'] & B['trk'] & (B['nstr'] < NSTRAIGHT) & hc(B)
    sfix = S['trig'] & S['presel'] & S['trk'] & (S['nstr'] < NSTRAIGHT) & hc(S)
    if FIDUCIAL:
        fixed &= B['fiducial']; sfix &= S['fiducial']
    d_all = np.sort(B['disc_' + kb][fixed])[::-1]
    if len(d_all):
        want = ([('1 event', float(np.nextafter(d_all[1], 1.0)))] if len(d_all) >= 2 else []) \
             + [('0 events', float(np.nextafter(d_all[0], 1.0)))]
        print('\n  %-10s %18s | %s' % ('survivors', 'disc cut', '  '.join('%9s' % m for m in MASSES)))
        for lab, thr in want:
            eff = [100.0 * (sig_mass[m] & sfix & (S['disc_' + ks] >= thr)).sum() / den[m] for m in MASSES]
            print('  %-10s %18.10f | %s   (bkg %d)'
                  % (lab, thr, '  '.join('%8.1f%%' % e for e in eff),
                     int((fixed & (B['disc_' + kb] >= thr)).sum())))
    print()

print('signal efficiencies are relative to the %s row' % DENOM_ROW)

if a.out:
    os.makedirs(a.outdir, exist_ok=True)

    def dump(path, data):
        if not data:
            open(path, 'w').write(''); return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
        print('  %-56s %d rows' % (path, len(data)))

    print()
    dump(os.path.join(a.outdir, a.out + '_cutflow.csv'), rows)
    dump(os.path.join(a.outdir, a.out + '_summary.csv'), summary)
    dump(os.path.join(a.outdir, a.out + '_survivors.csv'), survivors)
