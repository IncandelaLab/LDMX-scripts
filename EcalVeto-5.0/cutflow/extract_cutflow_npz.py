#!/usr/bin/env python3
"""
extract_cutflow_npz.py
======================
ONE ROOT pass over scored files -> a .npz holding every per-event quantity the
cutflow needs, for ANY number of EcalVeto BDT branches found in the files.

Stored per event:
    trig, fiducial, tracking_fiducial, presel, trk, nstr, maxpe, hcal_pass,
    run, evt, fileidx,  and  disc_<model>  for each BDT branch kept
Plus metadata: _files, _models, _indir, _is_sig, _slice

Two modes
---------
EXTRACT (default):
    denv python3 extract_cutflow_npz.py --in <scored_dir> --out bkg.npz
    denv python3 extract_cutflow_npz.py --in <sig_dir>    --out sig.npz --is-sig
    # only some BDTs:
    denv python3 extract_cutflow_npz.py --in <dir> --out b.npz --models Humberto,reco_v1
    # one slice, for parallel SLURM jobs:
    denv python3 extract_cutflow_npz.py --in <dir> --out chunks/chunk_0.npz --start 0 --stop 30

MERGE (combine the slices):
    denv python3 extract_cutflow_npz.py --merge chunks/ --out bkgfull.npz

"""
import argparse, glob, os, sys, time
import numpy as np
import uproot
import awkward as ak

ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='indir', help='directory of scored .root files')
ap.add_argument('--out', required=True, help='output .npz')
ap.add_argument('--merge', default=None, help='MERGE MODE: directory of chunk_*.npz')
ap.add_argument('--start', type=int, default=0, help='first file index of the slice')
ap.add_argument('--stop', type=int, default=-1, help='one past the last file index (-1 = all)')
ap.add_argument('--is-sig', dest='is_sig', action='store_true')
ap.add_argument('--models', default=None,
                help='comma-separated BDT names to keep (default: every EcalVeto_* found). '
                     'Names are the branch minus the EcalVeto_ prefix and _BDT suffix.')
a = ap.parse_args()


# ----------------------------------------------------------------- merge mode
if a.merge:
    paths = sorted(glob.glob(os.path.join(a.merge, '*.npz')))
    if not paths:
        raise SystemExit('no .npz chunks in ' + a.merge)
    zs = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if '_slice' not in z.files:
            raise SystemExit('%s has no _slice metadata -- not a chunk from this script' % p)
        zs.append((int(z['_slice'][0]), int(z['_slice'][1]), p, z))
    zs.sort(key=lambda x: x[0])
    models, expect = None, zs[0][0]
    for s, e, p, z in zs:
        if s != expect:
            raise SystemExit('COVERAGE GAP/OVERLAP: expected start %d, got %d (%s)' % (expect, s, p))
        expect = e
        m = tuple(sorted(str(x) for x in z['_models']))
        if models is None:
            models = m
        elif m != models:
            raise SystemExit('MODEL MISMATCH in %s: %s vs %s' % (p, m, models))
    print('%d chunks, files [%d:%d), models: %s' % (len(zs), zs[0][0], zs[-1][1], ', '.join(models)))
    keys = [k for k in zs[0][3].files if not k.startswith('_')]
    out = {k: np.concatenate([z[k] for _s, _e, _p, z in zs]) for k in keys}
    out['_files'] = np.concatenate([z['_files'] for _s, _e, _p, z in zs])
    out['_models'] = np.array(models)
    out['_indir'] = zs[0][3]['_indir']
    out['_is_sig'] = zs[0][3]['_is_sig']
    n = len(out['trig'])
    for k in keys:
        if len(out[k]) != n:
            raise SystemExit('LENGTH MISMATCH %s: %d vs %d' % (k, len(out[k]), n))
    print('merged events: %s   files: %d   fileidx %d..%d'
          % (format(n, ','), len(out['_files']), out['fileidx'].min(), out['fileidx'].max()))
    np.savez_compressed(a.out, **out)
    print('wrote %s (%.0f MB)' % (a.out, os.path.getsize(a.out) / 1e6))
    raise SystemExit


# --------------------------------------------------------------- extract mode
if not a.indir:
    raise SystemExit('--in is required unless --merge is given')
allfiles = [f for f in sorted(glob.glob(a.indir + '/*.root')) if os.path.getsize(f) > 100000]
stop = len(allfiles) if a.stop < 0 else min(a.stop, len(allfiles))
files = allfiles[a.start:stop]
if not files:
    raise SystemExit('no files in slice [%d:%d) of %s' % (a.start, stop, a.indir))
tops = sorted(set(k.split('/')[0] for k in uproot.open(files[0])['LDMX_Events'].keys()))


def pick(prefix, what):
    """Resolve a collection by PREFIX -- pass names differ between samples."""
    c = [x for x in tops if x.startswith(prefix)]
    if not c:
        raise SystemExit('no %s collection (prefix %r) in %s' % (what, prefix, files[0]))
    return sorted(c, key=len)[0]


MIP = pick('EcalMipInfo', 'MIP')
TRK = pick('TrackerVeto', 'tracker veto')
HC = pick('HcalRecHits', 'HCal rec hits')
HV = pick('HcalVeto', 'HCal veto')
PRE = pick('EcalPreselectionDecision', 'preselection')

MODELS = {}
for c in tops:
    if not c.startswith('EcalVeto_'):
        continue
    n = c[len('EcalVeto_'):]
    if n.startswith('mip_'):          # the MIP-pass re-run veto, not a BDT
        continue
    MODELS[n.replace('_BDT', '')] = c
if a.models:
    want = [x.strip() for x in a.models.split(',') if x.strip()]
    missing = [w for w in want if w not in MODELS]
    if missing:
        raise SystemExit('requested model(s) %s not in file; available: %s'
                         % (missing, sorted(MODELS)))
    MODELS = {k: v for k, v in MODELS.items() if k in want}
if not MODELS:
    raise SystemExit('no EcalVeto BDT branches found in ' + files[0])
# fiducial_ lives on EcalVeto; prefer the reco collection, else any model
pref = [c for c in MODELS.values() if c.endswith('_reco') or c.endswith('_reco_v1')]
FIDSRC = sorted(pref or list(MODELS.values()), key=len)[0]

print('slice [%d:%d) of %s (%d files)' % (a.start, stop, a.indir, len(files)))
print('cols  : %s | %s | %s | %s | %s | fid<-%s' % (MIP, TRK, HC, HV, PRE, FIDSRC))
print('models: %s' % ', '.join(sorted(MODELS)), flush=True)

acc = {k: [] for k in ['trig', 'presel', 'trk', 'nstr', 'maxpe', 'hcal_pass',
                       'fiducial', 'tracking_fiducial', 'run', 'evt', 'fileidx']}
dacc = {k: [] for k in MODELS}
t0, nbad = time.time(), 0
for i, f in enumerate(files):
    try:
        t = uproot.open(f)['LDMX_Events']
        n = t.num_entries
        if n == 0:
            continue
    except Exception as ex:
        nbad += 1
        print('  [unreadable] %s: %s' % (os.path.basename(f)[:50], str(ex)[:60]), flush=True)
        continue

    def arr(k, d_):
        try:
            return np.asarray(t[k].array(library='np'))
        except Exception:
            return np.full(n, d_)

    acc['trig'].append(arr('Trigger_sim/pass_', True).astype(bool))
    # EcalPreselectionDecision is a TOP-LEVEL bool branch with no sub-fields.
    try:
        acc['presel'].append(np.asarray(t[PRE].array(library='np')).astype(bool))
    except Exception:
        pk = [k for k in t.keys() if k.startswith(PRE + '/')]
        if not pk:
            raise SystemExit('cannot read %s in %s' % (PRE, f))
        acc['presel'].append(np.asarray(t[pk[0]].array(library='np')).astype(bool))
    acc['trk'].append(arr(TRK + '/passes_veto_', True).astype(bool))
    acc['nstr'].append(np.asarray(t[MIP + '/n_straight_tracks_'].array(library='np')).astype(np.int16))
    try:
        pe = t[HC + '/' + HC + '.pe_'].array(library='ak')
        acc['maxpe'].append(ak.to_numpy(ak.fill_none(ak.max(pe, axis=1), 0.0)).astype(np.float32))
    except Exception:
        acc['maxpe'].append(np.zeros(n, np.float32))
    # the OFFICIAL HCal decision, not the same as maxPE<8 (it also passes some
    # high-PE events on time/section criteria). Stored alongside maxpe so the
    # cutflow can use either.
    acc['hcal_pass'].append(arr(HV + '/passes_veto_', True).astype(bool))
    acc['fiducial'].append(arr(FIDSRC + '/fiducial_', True).astype(bool))
    acc['tracking_fiducial'].append(arr(FIDSRC + '/tracking_fiducial_', True).astype(bool))
    for k, col in MODELS.items():
        dacc[k].append(np.asarray(t[col + '/disc_value_'].array(library='np')).astype(np.float32))
    acc['run'].append(arr('EventHeader/run_', -1).astype(np.int64))
    acc['evt'].append(arr('EventHeader/event_number_', -1).astype(np.int64))
    acc['fileidx'].append(np.full(n, a.start + i, np.int32))    # GLOBAL index
    if (i + 1) % 10 == 0 or i + 1 == len(files):
        print('  %4d/%d files  %10s events  %5.1f min'
              % (i + 1, len(files), format(sum(len(x) for x in acc['trig']), ','),
                 (time.time() - t0) / 60), flush=True)

out = {k: np.concatenate(v) for k, v in acc.items() if v}
for k, v in dacc.items():
    out['disc_' + k] = np.concatenate(v)
out['_files'] = np.array([os.path.basename(f) for f in files])
out['_models'] = np.array(sorted(MODELS))
out['_indir'] = np.array([a.indir])
out['_is_sig'] = np.array([a.is_sig])
out['_slice'] = np.array([a.start, stop])
od = os.path.dirname(os.path.abspath(a.out))
if od:
    os.makedirs(od, exist_ok=True)
np.savez_compressed(a.out, **out)
print('wrote %s  (%s events, %d unreadable files, %.1f min, %.0f MB)'
      % (a.out, format(len(out['trig']), ','), nbad, (time.time() - t0) / 60,
         os.path.getsize(a.out) / 1e6), flush=True)
