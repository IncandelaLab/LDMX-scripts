#!/usr/bin/env python3
"""Flatten the tracks from one scan point into a CSV ntuple.

One row per track, with the distortion parameters carried along as columns so
the scan points can simply be concatenated afterwards. Run inside the container
(the LDMX event dictionaries have to be loadable).

  denv python3 make_ntuple.py scan/tracks_*.root -o ntuples/

Momenta are in MeV and positions in mm, both in the LDMX global frame.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import ROOT

# without these the branches come back as opaque ints
for _lib in ("libSimCore_Event", "libTracking_Event"):
    ROOT.gSystem.Load(_lib)

# unique / duplicate / fake follow TrackingRecoDQM::sortTracks so the numbers
# here are comparable to the DQM ones
TRUTH_PROB_CUT = 0.5

COLLECTIONS = {
    "tagger": ("TaggerTracks", "TaggerTruthTracks"),
    "recoil": ("RecoilTracks", "RecoilTruthTracks"),
}

AT_TARGET = 1


def state_at(track, tstype):
    for state in track.getTrackStates():
        if state.ts_type_ == tstype:
            return state
    return None


def track_rows(track, tracker, truth_by_id, point):
    state = state_at(track, AT_TARGET)
    if state is None:
        return None
    px, py, pz = state.mom_[0], state.mom_[1], state.mom_[2]
    x, y, _ = state.pos_[0], state.pos_[1], state.pos_[2]
    ndf = track.getNdf()

    row = {
        "tracker": tracker,
        "p": math.sqrt(px * px + py * py + pz * pz),
        "px": px,
        "py": py,
        "pz": pz,
        "x": x,
        "y": y,
        "dxdz": px / pz if pz else float("nan"),
        "dydz": py / pz if pz else float("nan"),
        "chi2": track.getChi2(),
        "ndf": ndf,
        "chi2ndf": track.getChi2() / ndf if ndf else float("nan"),
        "nhits": track.getNhits(),
        "truth_prob": track.getTruthProb(),
        "track_id": track.getTrackID(),
        "d0": track.getD0(),
        "z0": track.getZ0(),
        "phi": track.getPhi(),
        "theta": track.getTheta(),
        "qop": track.getQoP(),
        "is_fake": track.getTruthProb() < TRUTH_PROB_CUT,
    }

    truth = truth_by_id.get(track.getTrackID())
    if truth is not None:
        row["truth_p"] = truth["p"]
        row["truth_theta"] = truth["theta"]
        row["dp"] = row["p"] - truth["p"]
        row["dp_over_p"] = (row["p"] - truth["p"]) / truth["p"] if truth["p"] else float("nan")

    row.update(point)
    return row


def truth_map(event, name):
    """Truth tracks are ldmx::Track too, so they carry track states rather than
    a bare momentum. Prefer the AtTarget state; fall back to the perigee q/p,
    which is always filled even when no state was stored."""
    out = {}
    for truth in getattr(event, name, []):
        mom = list(truth.getMomentumAtTarget())
        if len(mom) == 3:
            p = math.sqrt(sum(c * c for c in mom))
            theta = math.acos(mom[2] / p) if p else float("nan")
        else:
            qop = truth.getQoP()
            if not qop:
                continue
            p = abs(1.0 / qop)
            theta = float("nan")
        out[truth.getTrackID()] = {"p": p, "theta": theta}
    return out


def convert(path, out_dir):
    sidecar = path.parent / f"point_{path.stem.removeprefix('tracks_')}.json"
    point = {}
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        tx, ty, tz = meta["translation"]
        ax, ay, az = meta["rotation"]
        point = {
            "tx": tx, "ty": ty, "tz": tz,
            "rx": ax, "ry": ay, "rz": az,
            "scale": meta["scale"],
        }

    tree_file = ROOT.TFile.Open(str(path))
    tree = tree_file.Get("LDMX_Events")

    # branches are named <collection>_<passname>, and the pass name is whatever
    # the reco job happened to use, so resolve by prefix rather than guessing
    branches = [b.GetName() for b in tree.GetListOfBranches()]

    def branch(collection):
        for name in branches:
            if name.rsplit("_", 1)[0] == collection:
                return name
        return None

    resolved = {
        tracker: (branch(trk), branch(truth))
        for tracker, (trk, truth) in COLLECTIONS.items()
    }
    for tracker, (trk, truth) in resolved.items():
        if trk is None:
            print(f"  warning: no {COLLECTIONS[tracker][0]} branch in {path.name}")

    rows, n_truth = [], {"tagger": 0, "recoil": 0}
    for event in tree:
        for tracker, (trk_name, truth_name) in resolved.items():
            if trk_name is None:
                continue
            truth_by_id = truth_map(event, truth_name) if truth_name else {}
            # count the collection, not the id map: several truth tracks can
            # share a track id, and the efficiency denominator must not collapse
            n_truth[tracker] += len(getattr(event, truth_name)) if truth_name else 0
            for track in getattr(event, trk_name):
                row = track_rows(track, tracker, truth_by_id, point)
                if row is not None:
                    rows.append(row)
    tree_file.Close()

    df = pd.DataFrame(rows)
    out = Path(out_dir) / f"{path.stem}.csv"
    df.to_csv(out, index=False)

    # efficiency needs the truth denominator, which is not a per-track quantity
    # and so has nowhere to live in the table itself
    summary = dict(point)
    summary.update({f"n_truth_{k}": v for k, v in n_truth.items()})
    summary["n_tracks"] = len(df)
    (Path(out_dir) / f"{path.stem}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{path.name}: {len(df)} tracks -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-o", "--out-dir", default="ntuples")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for path in args.files:
        convert(path, args.out_dir)


if __name__ == "__main__":
    main()
