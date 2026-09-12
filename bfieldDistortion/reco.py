#!/bin/python
"""Re-run tracking over a fixed sim file with a deliberately mis-set B field.

The simulation is never re-run: every scan point reads the same input file, so
the only thing that changes between points is the field the CKF and the GSF
reconstruct with. That is the whole point of the study, and it is also what
makes the scan cheap.

The distortion is in the LDMX global frame -- x bend plane, y vertical, z beam.
A positive translation moves the magnet, so the field at a fixed point becomes
the nominal field from further upstream. Rotations are about bfield_pivot,
which defaults to the field-map origin at z = -400 mm.

Needs an ldmx-sw with the bfield_* knobs on CKFProcessor and GSFProcessor.

Example:

  fire reco.py --input-file events_target_mono_4gev_run0001.root \\
      --translation 0 0 2.0 --out-dir scan/
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--out-dir", default=".")
parser.add_argument("--detector", default="ldmx-det-v15-8gev-no-cals")
parser.add_argument("--input-pass-name", default="")
parser.add_argument("--max-events", type=int, default=-1)
parser.add_argument("--tag", default="", help="extra label for the output name")
parser.add_argument(
    "--translation", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z")
)
parser.add_argument(
    "--rotation",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    metavar=("AX", "AY", "AZ"),
    help="radians about the LDMX x, y, z axes",
)
parser.add_argument(
    "--pivot", type=float, nargs=3, default=[0.0, 0.0, -400.0], metavar=("X", "Y", "Z")
)
parser.add_argument("--scale", type=float, default=1.0)
args = parser.parse_args()


def label():
    """Short, sortable, unambiguous name for this scan point."""
    if all(v == 0.0 for v in args.translation + args.rotation) and args.scale == 1.0:
        point = "nominal"
    else:
        tx, ty, tz = args.translation
        ax, ay, az = args.rotation
        point = (
            f"t{tx:+.3f}_{ty:+.3f}_{tz:+.3f}"
            f"_r{ax:+.5f}_{ay:+.5f}_{az:+.5f}"
            f"_s{args.scale:.5f}"
        )
    return f"{args.tag}_{point}" if args.tag else point


from LDMX.Framework import ldmxcfg

p = ldmxcfg.Process("trkreco")
p.input_files = [args.input_file]
p.max_events = args.max_events

name = label()
p.output_files = [f"{args.out_dir}/tracks_{name}.root"]
p.histogram_file = f"{args.out_dir}/hist_{name}.root"

from LDMX.Tracking.full_tracking_sequence import full_tracking_sequence

trk = full_tracking_sequence(detector=args.detector)

# The CKF finds and fits; the GSF refits what the CKF found. Giving them
# different fields would make the GSF collections meaningless, so both get the
# same distortion.
for proc in (trk.tracking_tagger, trk.tracking_recoil, trk.gsf_tagger, trk.gsf_recoil):
    proc.bfield_translation = list(args.translation)
    proc.bfield_rotation = list(args.rotation)
    proc.bfield_pivot = list(args.pivot)
    proc.bfield_scale = args.scale

# Only the parameters that point at *sim* collections take the input pass name.
# The rest refer to collections this pass produces itself, and forcing the input
# pass on those makes the job fail looking for products that do not exist yet.
# Empty means "resolve it", which is what the reco-to-reco links want.
SIM_PASSNAMES = (
    "tracker_hit_passname",
    "sim_particles_passname",
    "sim_particles_event_passname",
    "sp_pass_name",
)
if args.input_pass_name:
    for proc in trk.sequence + trk.dqm_sequence:
        for key in SIM_PASSNAMES:
            if hasattr(proc, key):
                setattr(proc, key, args.input_pass_name)

p.sequence = trk.sequence + trk.dqm_sequence

# a sidecar so the analysis never has to parse a filename
os.makedirs(args.out_dir, exist_ok=True)
with open(f"{args.out_dir}/point_{name}.json", "w") as f:
    json.dump(
        {
            "translation": args.translation,
            "rotation": args.rotation,
            "pivot": args.pivot,
            "scale": args.scale,
            "input_file": os.path.abspath(args.input_file),
            "detector": args.detector,
            "output_file": p.output_files[0],
        },
        f,
        indent=2,
    )
