#!/usr/bin/env python3
"""
shap_model.py

Usage:
  denv python3 shap_model.py --label <BDT_name> --model <pkl> \
      --sig-dir <dir> --sig-branch EcalVeto_reco_v1 \
      --bkg-dir <dir> --bkg-branch EcalVeto_reco_v1 --n-sig 50000 --n-bkg 50000
"""
import argparse, glob, os, re, sys, time, pickle
import numpy as np
import uproot
import awkward as ak
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--model', default=None)
ap.add_argument('--sig-dir', default=None)
ap.add_argument('--bkg-dir', default=None)
ap.add_argument('--sig-branch', default=None)
ap.add_argument('--bkg-branch', default=None)
ap.add_argument('--n-sig', type=int, default=50000)
ap.add_argument('--n-bkg', type=int, default=50000)
ap.add_argument('--nthread', type=int, default=16)
ap.add_argument('--energy-label', default='8 GeV')
ap.add_argument('--exclude-list', default=None,
                help='text file of basenames to SKIP (e.g. files the model trained on), one per line')
ap.add_argument('--sig-tail', action='store_true',
                help='take signal from the END of each mass\'s sorted file list. bdtMaker reads signal\n'
                     '                files sorted and interleaved by mass, so a model trained with a\n'
                     '                max_evt below the pool size never saw the tail files.')
ap.add_argument('--bkg-tail', action='store_true',
                help='take background from the END of UNSORTED glob order -- the order bdtMaker used')
ap.add_argument('--replot', default=None,
                help='path to an existing <label>_shap.npz: redraw plots only, no SHAP recompute')
ap.add_argument('--out', default='/home/jihoonyoo/LDMX/v4.5.11/ldmx-sw/bdt_analysis/SHAP')
ap.add_argument('--precuts', action='store_true',
                help='keep only events passing the cutflow stages BEFORE the BDT, on BOTH '
                     'signal and background: Trigger + Fiducial + Preselection + Tracker veto')
a = ap.parse_args()

def make_plots(sv, X, names, od, label):
    """mean|SHAP| CSV + bar + beeswarm, LDMX-tagged as in shap_format_correct.ipynb."""
    os.makedirs(od, exist_ok=True)
    # Display-only rename: the booster stores the feature as 'n_tracking_hits',
    # but the name we publish is nHitsInPhotonTerritory. Applied to the CSV and
    # both plots. The saved _shap.npz keeps the booster's original names, so this
    # stays re-runnable and does not mutate any computed SHAP data.
    RENAME = {'n_tracking_hits': 'nHitsInPhotonTerritory'}
    names = [RENAME.get(n, n) for n in names]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)
    with open(os.path.join(od, label + '_mean_abs_shap.csv'), 'w') as f:
        f.write('rank,feature,mean_abs_shap\n')
        for r, i in enumerate(order):
            f.write('%d,%s,%.6f\n' % (r + 1, names[i], imp[i]))

    def ldmx_tag(ax):
        ax.text(0.99, 0.90, 'LDMX', fontsize=25, fontweight='bold', transform=ax.transAxes,
                ha='left', va='bottom', rotation=-90)
        ax.text(0.99, 0.82, 'Simulation', fontsize=18, fontstyle='italic', transform=ax.transAxes,
                ha='left', va='bottom', rotation=-90)
        # directly under 'Simulation' (the notebook's 0.69 was tuned for a longer string)
        ax.text(0.99, 0.775, a.energy_label, fontsize=16, transform=ax.transAxes,
                ha='left', va='bottom', rotation=-90)

    try:
        import shap
        shap.summary_plot(sv, X, feature_names=names, plot_type='bar', max_display=44, show=False)
        ax = plt.gca(); ax.set_xlabel('Mean SHAP value', fontsize=13); plt.title(label, fontsize=16)
        plt.savefig(os.path.join(od, label + '_shap_bar.png'), bbox_inches='tight', dpi=300); plt.close()
        shap.summary_plot(sv, X, feature_names=names, max_display=44, show=False)
        ax = plt.gca(); ax.set_xlabel('SHAP value', fontsize=13); plt.title(label, fontsize=16)
        plt.savefig(os.path.join(od, label + '_shap_beeswarm.png'), bbox_inches='tight', dpi=300); plt.close()
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 14))
        ax.barh(range(len(names))[::-1], imp[order], color='#1E88E5')
        ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels([names[i] for i in order])
        ax.set_xlabel('Mean SHAP value'); ax.set_title(label)
        fig.savefig(os.path.join(od, label + '_shap_bar.png'), bbox_inches='tight', dpi=300)
    return imp, order


if a.replot:
    z = np.load(a.replot, allow_pickle=True)
    od = os.path.dirname(os.path.abspath(a.replot))
    make_plots(z['shap'], z['X'], [str(x) for x in z['feature_names']], od, a.label)
    print('replotted from %s -> %s' % (a.replot, od))
    sys.exit(0)
for _req in ('model', 'sig_dir', 'bkg_dir', 'sig_branch', 'bkg_branch'):
    if getattr(a, _req) is None:
        raise SystemExit('--%s is required unless --replot is given' % _req.replace('_', '-'))


TRIG = 'Trigger_sim'
FLAT = ['n_readout_hits_', 'summed_det_', 'summed_tight_iso_', 'max_cell_dep_', 'shower_rms_',
        'x_std_', 'y_std_', 'avg_layer_hit_', 'std_layer_hit_', 'deepest_layer_hit_',
        'ecal_back_energy_', 'ep_sep_', 'ep_dot_', 'n_tracking_hits_']
IDX = [('energy_seg_', (0,)), ('x_mean_seg_', (0,)), ('y_mean_seg_', (0,)), ('layer_mean_seg_', (0,)),
       ('energy_seg_', (1,)), ('y_mean_seg_', (2,)),
       ('e_cont_energy_', (0, 0)), ('e_cont_energy_', (1, 0)), ('e_cont_y_mean_', (0, 0)),
       ('e_cont_energy_', (0, 1)), ('e_cont_energy_', (1, 1)), ('e_cont_y_mean_', (0, 1)),
       ('g_cont_n_hits_', (0, 0)), ('g_cont_y_mean_', (0, 0)), ('g_cont_n_hits_', (0, 1)),
       ('o_cont_energy_', (0, 0)), ('o_cont_energy_', (1, 0)), ('o_cont_energy_', (2, 0)),
       ('o_cont_n_hits_', (0, 0)), ('o_cont_x_mean_', (0, 0)), ('o_cont_y_mean_', (0, 0)),
       ('o_cont_y_mean_', (1, 0)), ('o_cont_y_std_', (0, 0)),
       ('o_cont_energy_', (0, 1)), ('o_cont_energy_', (1, 1)), ('o_cont_energy_', (2, 1)),
       ('o_cont_layer_mean_', (0, 1)), ('o_cont_layer_std_', (0, 1)),
       ('o_cont_energy_', (0, 2)), ('o_cont_layer_mean_', (0, 2))]


PRECUT_N = [0, 0]   # [read, passed], for the log
PRESEL_MISSING = set()


def read_file(f, br, is_sig, need=None):
    want = [br] + ([TRIG] if (is_sig or a.precuts) else [])
    t = uproot.open(f)['LDMX_Events']
    pre = trk = None
    if a.precuts:
        # pass names differ by sample (presel: _ecal_pres vs _Skim), so resolve per file
        _pc = sorted({k for k in t.keys() if k.startswith('EcalPreselectionDecision')}, key=len)
        # Trigger-level samples (SP / Recoil training sets) never had preselection
        # run, so the branch does not exist. Apply the other three cuts and record
        # it -- never silently treat a missing flag as all-pass.
        pre = _pc[0] if _pc else None
        if pre is None:
            PRESEL_MISSING.add(os.path.basename(os.path.dirname(f)))
        trk = sorted({k.split('/')[0] for k in t.keys() if k.startswith('TrackerVeto')}, key=len)[0] \
            + '/passes_veto_'
        want += ([pre] if pre else []) + [trk]
    # Only read as many entries as still needed (with headroom for the cuts).
    # Signal files here hold ~190k events each, and reading the whole EcalVeto
    # collection when 12.5k are needed dominated the runtime.
    head = 2.5 if a.precuts else 1.6
    stop = None if need is None else min(t.num_entries, int(need * head) + 1000)
    arr = t.arrays(want, library='ak', entry_stop=stop)
    if a.precuts:
        m = arr[TRIG].pass_ & arr[br].fiducial_ & ak.values_astype(arr[trk], bool)
        if pre:
            m = m & ak.values_astype(arr[pre], bool)
        PRECUT_N[0] += len(arr); PRECUT_N[1] += int(ak.sum(m))
        arr = arr[m]
    elif is_sig:
        arr = arr[arr[TRIG].pass_]
    e = arr[br]
    cols = [ak.to_numpy(e[n]).astype(np.float32) for n in FLAT]
    for n, ix in IDX:
        cols.append(ak.to_numpy(e[n][(slice(None),) + ix]).astype(np.float32))
    return np.column_stack(cols)


def mass_of(name):
    m = re.search(r'_(1|10|100|1000)MeV_', name)
    return m.group(1) if m else None


EXCL = set()
if a.exclude_list:
    EXCL = set(l.strip() for l in open(a.exclude_list) if l.strip())


def collect(d, br, is_sig, nmax):
    if is_sig:
        fs = [f for f in sorted(glob.glob(d + '/*.root')) if os.path.getsize(f) > 100000]
    else:
        # UNSORTED glob, matching bdtMaker, so '--bkg-tail' means "last in bdtMaker's order"
        fs = [f for f in glob.glob(d + '/*.root') if os.path.getsize(f) > 100000]
    fs = [f for f in fs if os.path.basename(f) not in EXCL]
    if not fs:
        raise SystemExit('no files in ' + d)
    if is_sig and a.sig_tail:
        pass   # handled per mass below
    if (not is_sig) and a.bkg_tail:
        fs = fs[::-1]
    if is_sig:
        # mass-balanced: round-robin one file per mass, cap each mass at nmax/4
        by = {}
        for f in fs:
            by.setdefault(mass_of(os.path.basename(f)) or 'unknown', []).append(f)
        masses = sorted(by)
        per = nmax // len(masses)
        out, got = [], {}
        for m in masses:
            chunk, n = [], 0
            for f in (by[m][::-1] if a.sig_tail else by[m]):
                try:
                    X = read_file(f, br, True, need=per - n)
                except Exception as ex:
                    print('  [skip] %s: %s' % (os.path.basename(f), str(ex)[:60]), flush=True); continue
                chunk.append(X); n += len(X)
                if n >= per:
                    break
            X = np.vstack(chunk)[:per]
            out.append(X); got[m] = len(X)
        print('  signal per mass: %s' % got, flush=True)
        return np.vstack(out)
    out, n = [], 0
    step = 1 if a.bkg_tail else max(1, len(fs) // 40)   # tail: contiguous from the end
    seen = set()
    for f in fs[::step] + fs:
        if f in seen:
            continue
        seen.add(f)
        if n >= nmax:
            break
        try:
            X = read_file(f, br, False, need=nmax - n)
        except Exception as ex:
            print('  [skip] %s: %s' % (os.path.basename(f), str(ex)[:60]), flush=True); continue
        out.append(X); n += len(X)
    return np.vstack(out)[:nmax]


t0 = time.time()
booster = pickle.load(open(a.model, 'rb'))
if hasattr(booster, 'get_booster'):
    booster = booster.get_booster()
fn = booster.feature_names
if booster.num_features() != 44 or (fn and len(fn) != 44):
    raise SystemExit('model has %d features, expected 44' % booster.num_features())
booster.set_param({'nthread': a.nthread})

# Column order must follow THIS model's training order. FLAT above is the
# original (V15R_base/Humberto) order; new-order models (Humberto2, H3_*) put
# nHitsInPhotonTerritory BEFORE epSep/epDot. Feeding the wrong order is silent:
# the SHAP self-check below only tests internal consistency, not alignment.
_BR = {'epSep': 'ep_sep_', 'epDot': 'ep_dot_',
       'n_tracking_hits': 'n_tracking_hits_', 'nHitsInPhotonTerritory': 'n_tracking_hits_'}
if fn:
    _mid = [_BR.get(x) for x in fn[11:14]]
    if None in _mid or sorted(_mid) != sorted(['ep_sep_', 'ep_dot_', 'n_tracking_hits_']):
        raise SystemExit('unrecognised feature names at f11-f13: %s' % fn[11:14])
    FLAT[11:14] = _mid
    print('[%s] feature order f11-f13 from model: %s -> %s' % (a.label, fn[11:14], _mid), flush=True)
else:
    print('[%s] WARNING: model has no feature_names; assuming original order' % a.label, flush=True)
print('[%s] model %s  trees=%d' % (a.label, os.path.basename(a.model), len(booster.get_dump())), flush=True)

print('reading signal (%s) ...' % a.sig_branch, flush=True)
Xs = collect(a.sig_dir, a.sig_branch, True, a.n_sig)
print('reading background (%s) ...' % a.bkg_branch, flush=True)
Xb = collect(a.bkg_dir, a.bkg_branch, False, a.n_bkg)
X = np.vstack([Xs, Xb]).astype(np.float32)
y = np.concatenate([np.ones(len(Xs)), np.zeros(len(Xb))])
print('events: sig %d  bkg %d  (read %.1f min)' % (len(Xs), len(Xb), (time.time() - t0) / 60), flush=True)
if a.precuts:
    print('precuts (Trig+Fid+Presel+TrkVeto): kept %d of %d events read (%.1f%%)'
          % (PRECUT_N[1], PRECUT_N[0], 100.0 * PRECUT_N[1] / max(PRECUT_N[0], 1)), flush=True)
    if PRESEL_MISSING:
        print('*** PRESELECTION NOT APPLIED for %s: no EcalPreselectionDecision branch '
              '(trigger-level sample). Cuts used there: Trigger + Fiducial + Tracker veto. ***'
              % sorted(PRESEL_MISSING), flush=True)

D = xgb.DMatrix(X, feature_names=fn)
t1 = time.time()
contrib = booster.predict(D, pred_contribs=True)          # (N, 45), last col = bias
margin = booster.predict(D, output_margin=True)
chk = np.abs(contrib.sum(axis=1) - margin).max()
print('TreeSHAP %.1f s   self-check max|sum(SHAP)-margin| = %.2e' % (time.time() - t1, chk), flush=True)
if chk > 1e-3:
    raise SystemExit('SHAP self-check FAILED')
sv = contrib[:, :-1]
names = list(fn) if fn else ['f%d' % i for i in range(44)]

od = os.path.join(a.out, a.label)
os.makedirs(od, exist_ok=True)
np.savez_compressed(os.path.join(od, a.label + '_shap.npz'), shap=sv.astype(np.float32), X=X, y=y,
                    feature_names=np.array(names), model=np.array([a.model]))

imp, order = make_plots(sv, X, names, od, a.label)
print('wrote %s  (total %.1f min)' % (od, (time.time() - t0) / 60), flush=True)
print('top 10:', ', '.join('%s=%.3f' % (names[i], imp[i]) for i in order[:10]), flush=True)
