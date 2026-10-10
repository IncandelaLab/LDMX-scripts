#!/usr/bin/env python3
"""
fom_ana.py
==========
Punzi figure of merit, from the same .npz the cutflow uses.

    FOM = eff / (a/2 + sqrt(B))        Punzi, default a = 3 (3 sigma)

For each target number of surviving background events it SOLVES for the disc cut
that leaves exactly that many, then reports signal efficiency per mass and the FOM.
This is where the recommended operating point comes from: the 1-event row is the
Punzi optimum, and the 0-event row is the zero-background cut.

Every cut is applied exactly as in cutflow_table_ana.py, so the numbers line up
with the cutflow table row for row.

Usage:
    denv python3 fom_ana.py --bkg bkgfull.npz --sig sig.npz
    denv python3 fom_ana.py --bkg b.npz --sig s.npz --models H3_s202501,V15R_base
    denv python3 fom_ana.py --bkg b.npz --sig s.npz --csv-dir fom_out/

The cut for exactly N survivors is nextafter(d[N]), where d is the surviving
background disc values sorted descending: raising the cut only ever removes
events, and no other cut depends on it.

QUOTE THE FULL PRECISION. These cuts compare against float32 disc values;
truncating 0.9546506405 to 0.9546506 makes the boundary event tie with the cut
and survive, turning a 1-event cut into 2 survivors.
"""
import argparse, csv, os
import numpy as np

# ============================== CONFIG ======================================
TARGETS    = [0, 1, 2, 3, 5, 9, 20]   # background events to solve for
PUNZI_A    = 3.0                      # FOM = eff / (a/2 + sqrt(B))
DENOM      = 'fiducial'               # 'fiducial' or 'triggered'
HCAL_MODE  = 'flag'                   # 'flag' = HcalVeto/passes_veto_ ; 'maxpe'
MAXPE      = 8.0
NSTRAIGHT  = 3
# ============================================================================

MASSES = ['1MeV', '10MeV', '100MeV', '1000MeV']
ap = argparse.ArgumentParser()
ap.add_argument('--bkg', required=True)
ap.add_argument('--sig', required=True)
ap.add_argument('--models', default=None,
                help='comma-separated model names (default: every model in both npz files)')
ap.add_argument('--csv-dir', dest='csv_dir', default=None, help='write fom_<model>.csv here')
a = ap.parse_args()


def materialise(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


B, S = materialise(a.bkg), materialise(a.sig)
bm = set(str(x) for x in B['_models'])
sm = set(str(x) for x in S['_models'])
models = [x.strip() for x in a.models.split(',')] if a.models else sorted(bm & sm)
if not models:
    raise SystemExit('no model in both npz files\n  bkg: %s\n  sig: %s' % (sorted(bm), sorted(sm)))


def resolve(ms, name):
    if name in ms:
        return name
    if name in ('SegMip', 'segmip'):
        for k in ('reco', 'reco_v1'):
            if k in ms:
                return k
    raise SystemExit('model %r not found; available: %s' % (name, sorted(ms)))


def hc(d):
    return d['hcal_pass'] if HCAL_MODE == 'flag' else (d['maxpe'] < MAXPE)


# everything EXCEPT the disc cut
bfix = B['trig'] & B['fiducial'] & B['presel'] & B['trk'] & (B['nstr'] < NSTRAIGHT) & hc(B)
sfix = S['trig'] & S['fiducial'] & S['presel'] & S['trk'] & (S['nstr'] < NSTRAIGHT) & hc(S)

fl = [str(x) for x in S['_files']]
mass_mask = {m: np.array([('_%s_' % m) in fl[i] for i in range(len(fl))])[S['fileidx']] for m in MASSES}
if DENOM == 'fiducial':
    den = {m: max(int((mass_mask[m] & S['trig'] & S['fiducial']).sum()), 1) for m in MASSES}
else:
    den = {m: max(int((mass_mask[m] & S['trig']).sum()), 1) for m in MASSES}

if a.csv_dir:
    os.makedirs(a.csv_dir, exist_ok=True)

for name in models:
    kb, ks = resolve(bm, name), resolve(sm, name)
    d_sorted = np.sort(B['disc_' + kb][bfix])[::-1]
    print('\n' + '=' * 78)
    print('model: %s   bkg after all non-disc cuts: %d   (sample %s events)'
          % (name, len(d_sorted), format(len(B['trig']), ',')))
    print('FOM = eff / (%.1f + sqrt(B))   [Punzi a=%.0f]   eff relative to %s'
          % (PUNZI_A / 2, PUNZI_A, DENOM.upper()))
    print('=' * 78)
    hdr = '%6s  %16s  %9s %9s %9s %9s  %9s' % ('bkg', 'disc cut', *MASSES, 'FOM(1MeV)')
    print(hdr); print('-' * len(hdr))
    rows = []
    for tgt in TARGETS:
        if tgt >= len(d_sorted):
            print('%6d  %16s  (only %d background events available)' % (tgt, '-', len(d_sorted)))
            continue
        thr = float(np.nextafter(d_sorted[tgt], 1.0))
        effs = [100.0 * (mass_mask[m] & sfix & (S['disc_' + ks] >= thr)).sum() / den[m] for m in MASSES]
        fom = (effs[0] / 100.0) / (PUNZI_A / 2 + np.sqrt(tgt))
        nb = int((bfix & (B['disc_' + kb] >= thr)).sum())
        print('%6d  %16.10f  %8.2f%% %8.2f%% %8.2f%% %8.2f%%  %9.3f%s'
              % (tgt, thr, *effs, fom, '' if nb == tgt else '   (check: %d)' % nb))
        rows.append([tgt, thr] + effs + [fom])
    if a.csv_dir:
        p = os.path.join(a.csv_dir, 'fom_%s.csv' % name)
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['bkg', 'disc_cut'] + ['sig_eff_' + m for m in MASSES] + ['FOM_1MeV'])
            for r in rows:
                w.writerow([r[0], '%.10f' % r[1]] + ['%.2f' % x for x in r[2:6]] + ['%.3f' % r[6]])
        print('wrote %s' % p)
