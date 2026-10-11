#!/usr/bin/env python3
"""
extract_plotting_npz.py
=======================
Builds the .npz that the plotting scripts read: the discriminator of every BDT
branch plus the cut flags needed to select events, for signal and background.

Three ways to make one
----------------------
1. From an existing CUTFLOW npz (no ROOT read at all -- instant). The cutflow npz
   already contains everything plotting needs:
       denv python3 extract_plotting_npz.py --from-cutflow bkgfull.npz --out bkg_plot.npz

2. Straight from scored ROOT files:
       denv python3 extract_plotting_npz.py --in <scored_dir> --out bkg_plot.npz
       denv python3 extract_plotting_npz.py --in <sig_dir>    --out sig_plot.npz --is-sig

3. In slices for a big sample, then merged (same pattern as the cutflow extractor):
       denv python3 extract_plotting_npz.py --in <dir> --out chunks/c_0.npz --start 0 --stop 30
       denv python3 extract_plotting_npz.py --merge chunks/ --out bkg_plot.npz

Stored per event: trig, fiducial, presel, trk, nstr, hcal_pass, fileidx,
disc_<model> for every BDT, plus _files / _models / _is_sig metadata.
Signal mass is derived from the filename (`_1MeV_` etc.) by the plotting scripts.

Branch names are resolved PER FILE by prefix: pass names differ between samples
(background `EcalPreselectionDecision_ecal_pres` vs signal `_Skim`, `_reco` vs
`_reco_v1`). Hardcoding one silently empties the other sample.
"""
import argparse, glob, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='indir', help='directory of scored .root files')
ap.add_argument('--out', required=True)
ap.add_argument('--from-cutflow', dest='from_cutflow', default=None,
                help='convert an existing cutflow .npz (no ROOT read)')
ap.add_argument('--merge', default=None, help='directory of chunk .npz files')
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--stop', type=int, default=-1)
ap.add_argument('--is-sig', dest='is_sig', action='store_true')
ap.add_argument('--models', default=None, help='comma-separated BDT names to keep')
a = ap.parse_args()

KEEP = ['trig', 'fiducial', 'presel', 'trk', 'nstr', 'hcal_pass', 'fileidx']


# ------------------------------------------------- 1. convert a cutflow npz
if a.from_cutflow:
    z = np.load(a.from_cutflow, allow_pickle=True)
    out = {k: z[k] for k in z.files if k in KEEP or k.startswith('disc_')}
    missing = [k for k in KEEP if k not in out]
    if missing:
        raise SystemExit('cutflow npz is missing %s -- was it made by extract_cutflow_npz.py?' % missing)
    for meta in ('_files', '_models', '_is_sig'):
        if meta in z.files:
            out[meta] = z[meta]
    np.savez_compressed(a.out, **out)
    print('converted %s -> %s  (%s events, models: %s)'
          % (a.from_cutflow, a.out, format(len(out['trig']), ','),
             ', '.join(str(x) for x in out.get('_models', []))))
    raise SystemExit


# ------------------------------------------------------------- 2. merge mode
if a.merge:
    paths = sorted(glob.glob(os.path.join(a.merge, '*.npz')))
    if not paths:
        raise SystemExit('no .npz in ' + a.merge)
    zs = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if '_slice' not in z.files:
            raise SystemExit('%s has no _slice metadata' % p)
        zs.append((int(z['_slice'][0]), int(z['_slice'][1]), p, z))
    zs.sort(key=lambda x: x[0])
    models, expect = None, zs[0][0]
    for s, e, p, z in zs:
        if s != expect:
            raise SystemExit('COVERAGE GAP/OVERLAP: expected %d, got %d (%s)' % (expect, s, p))
        expect = e
        m = tuple(sorted(str(x) for x in z['_models']))
        if models is None:
            models = m
        elif m != models:
            raise SystemExit('MODEL MISMATCH in %s: %s vs %s' % (p, m, models))
    keys = [k for k in zs[0][3].files if not k.startswith('_')]
    out = {k: np.concatenate([z[k] for _s, _e, _p, z in zs]) for k in keys}
    out['_files'] = np.concatenate([z['_files'] for _s, _e, _p, z in zs])
    out['_models'] = np.array(models)
    out['_is_sig'] = zs[0][3]['_is_sig']
    n = len(out['trig'])
    for k in keys:
        if len(out[k]) != n:
            raise SystemExit('LENGTH MISMATCH %s' % k)
    np.savez_compressed(a.out, **out)
    print('merged %d chunks -> %s (%s events, models: %s)'
          % (len(zs), a.out, format(n, ','), ', '.join(models)))
    raise SystemExit


# ----------------------------------------------------------- 3. extract mode
import uproot
if not a.indir:
    raise SystemExit('give one of --in, --from-cutflow, --merge')
allfiles = [f for f in sorted(glob.glob(a.indir + '/*.root')) if os.path.getsize(f) > 100000]
stop = len(allfiles) if a.stop < 0 else min(a.stop, len(allfiles))
files = allfiles[a.start:stop]
if not files:
    raise SystemExit('no files in slice [%d:%d)' % (a.start, stop))
tops = sorted(set(k.split('/')[0] for k in uproot.open(files[0])['LDMX_Events'].keys()))


def pick(prefix, what):
    c = [x for x in tops if x.startswith(prefix)]
    if not c:
        raise SystemExit('no %s collection (prefix %r)' % (what, prefix))
    return sorted(c, key=len)[0]


MIP = pick('EcalMipInfo', 'MIP')
TRK = pick('TrackerVeto', 'tracker veto')
HV = pick('HcalVeto', 'HCal veto')
PRE = pick('EcalPreselectionDecision', 'preselection')
MODELS = {}
for c in tops:
    if c.startswith('EcalVeto_'):
        n = c[len('EcalVeto_'):]
        if not n.startswith('mip_'):
            MODELS[n.replace('_BDT', '')] = c
if a.models:
    want = [x.strip() for x in a.models.split(',')]
    miss = [w for w in want if w not in MODELS]
    if miss:
        raise SystemExit('model(s) %s not in file; available %s' % (miss, sorted(MODELS)))
    MODELS = {k: v for k, v in MODELS.items() if k in want}
if not MODELS:
    raise SystemExit('no EcalVeto BDT branches found')
pref = [c for c in MODELS.values() if c.endswith('_reco') or c.endswith('_reco_v1')]
FID = sorted(pref or list(MODELS.values()), key=len)[0]
print('slice [%d:%d) of %s (%d files)\nmodels: %s' % (a.start, stop, a.indir, len(files), ', '.join(sorted(MODELS))))

acc = {k: [] for k in KEEP}
dacc = {k: [] for k in MODELS}
t0 = time.time()
for i, f in enumerate(files):
    t = uproot.open(f)['LDMX_Events']
    n = t.num_entries
    if n == 0:
        continue

    def arr(k, d_):
        try:
            return np.asarray(t[k].array(library='np'))
        except Exception:
            return np.full(n, d_)

    acc['trig'].append(arr('Trigger_sim/pass_', True).astype(bool))
    try:                                   # bare bool branch, no sub-fields
        acc['presel'].append(np.asarray(t[PRE].array(library='np')).astype(bool))
    except Exception:
        pk = [k for k in t.keys() if k.startswith(PRE + '/')]
        if not pk:
            raise SystemExit('cannot read %s in %s' % (PRE, f))
        acc['presel'].append(np.asarray(t[pk[0]].array(library='np')).astype(bool))
    acc['trk'].append(arr(TRK + '/passes_veto_', True).astype(bool))
    acc['fiducial'].append(arr(FID + '/fiducial_', True).astype(bool))
    acc['nstr'].append(np.asarray(t[MIP + '/n_straight_tracks_'].array(library='np')).astype(np.int16))
    acc['hcal_pass'].append(arr(HV + '/passes_veto_', True).astype(bool))
    acc['fileidx'].append(np.full(n, a.start + i, np.int32))
    for k, col in MODELS.items():
        dacc[k].append(np.asarray(t[col + '/disc_value_'].array(library='np')).astype(np.float32))
    if (i + 1) % 20 == 0 or i + 1 == len(files):
        print('  %4d/%d files  %10s events  %.1f min'
              % (i + 1, len(files), format(sum(len(x) for x in acc['trig']), ','),
                 (time.time() - t0) / 60), flush=True)

out = {k: np.concatenate(v) for k, v in acc.items()}
for k, v in dacc.items():
    out['disc_' + k] = np.concatenate(v)
out['_files'] = np.array([os.path.basename(f) for f in files])
out['_models'] = np.array(sorted(MODELS))
out['_is_sig'] = np.array([a.is_sig])
out['_slice'] = np.array([a.start, stop])
od = os.path.dirname(os.path.abspath(a.out))
if od:
    os.makedirs(od, exist_ok=True)
np.savez_compressed(a.out, **out)
print('wrote %s (%s events, %.1f min)' % (a.out, format(len(out['trig']), ','), (time.time() - t0) / 60))
