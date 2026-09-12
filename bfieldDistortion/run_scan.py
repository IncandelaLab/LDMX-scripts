#!/usr/bin/env python3
"""Drive a B-field distortion scan: one reconstruction job per grid point.

Every point reads the same sim file, so this only re-runs reconstruction. Run
it outside the container; each point is dispatched as its own `denv fire`.

  # see what would run
  ./run_scan.py --input-file events.root --out-dir scan/ --dry-run

  # one-parameter scan along the beam axis, four at a time
  ./run_scan.py --input-file events.root --out-dir scan/ \
      --scan tz --jobs 4

  # everything
  ./run_scan.py --input-file events.root --out-dir scan/ --scan all

Re-running skips points whose output already exists, so an interrupted scan can
just be started again.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# The ranges are wide enough to see a clear trend without leaving the field map,
# which is only +-250 mm in x, +-70 mm in y and +-1500 mm in z about z = -400.
# Beyond the map ACTS returns zero field, which shows up as a fake nonlinearity.
STEPS_MM = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
STEPS_RAD = [0.0, 0.010, 0.025, 0.050, 0.100, 0.150, 0.200]
STEPS_SCALE = [0.98, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02]

SCANS = {
    "tx": ("translation", 0, STEPS_MM),
    "ty": ("translation", 1, STEPS_MM),
    "tz": ("translation", 2, STEPS_MM),
    "rx": ("rotation", 0, STEPS_RAD),
    "ry": ("rotation", 1, STEPS_RAD),
    "rz": ("rotation", 2, STEPS_RAD),
    "scale": ("scale", None, STEPS_SCALE),
}

# Realistic simultaneous mis-placements, to check the single-parameter slopes
# add linearly in the regime we care about.
COMBINED = [
    {"translation": [0.0, 0.0, 1.0], "rotation": [0.0, 0.0, 0.025], "scale": 1.002},
    {"translation": [0.5, 0.0, 1.0], "rotation": [0.0, 0.0, 0.025], "scale": 1.000},
    {"translation": [0.0, 0.5, 2.0], "rotation": [0.025, 0.0, 0.0], "scale": 0.998},
    {"translation": [1.0, 1.0, 1.0], "rotation": [0.0, 0.0, 0.000], "scale": 1.005},
    {"translation": [0.0, 0.0, 5.0], "rotation": [0.0, 0.0, 0.100], "scale": 1.010},
]

NOMINAL = {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": 1.0}


def grid(scans):
    """Points to run. The nominal point is always first -- it is the baseline
    every slope is measured against, and the regression check that an
    unconfigured job is unchanged."""
    points = [dict(NOMINAL)]
    for scan in scans:
        if scan == "combined":
            points.extend(dict(NOMINAL) | c for c in COMBINED)
            continue
        field, index, steps = SCANS[scan]
        for step in steps:
            point = {k: list(v) if isinstance(v, list) else v for k, v in NOMINAL.items()}
            if index is None:
                point[field] = step
            else:
                point[field][index] = step
            points.append(point)
    # drop duplicates (every scan contains its own nominal point)
    seen, unique = set(), []
    for point in points:
        key = json.dumps(point, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def command(point, args):
    cmd = ["denv", "fire", str(HERE / "reco.py")]
    cmd += ["--input-file", str(Path(args.input_file).resolve())]
    cmd += ["--out-dir", str(Path(args.out_dir).resolve())]
    cmd += ["--detector", args.detector]
    cmd += ["--input-pass-name", args.input_pass_name]
    cmd += ["--translation", *(str(v) for v in point["translation"])]
    cmd += ["--rotation", *(str(v) for v in point["rotation"])]
    cmd += ["--scale", str(point["scale"])]
    if args.max_events > 0:
        cmd += ["--max-events", str(args.max_events)]
    if args.tag:
        cmd += ["--tag", args.tag]
    return cmd


def output_for(point, args):
    """Mirror reco.py's naming so we can tell whether a point is already done."""
    if all(v == 0.0 for v in point["translation"] + point["rotation"]) and (
        point["scale"] == 1.0
    ):
        name = "nominal"
    else:
        tx, ty, tz = point["translation"]
        ax, ay, az = point["rotation"]
        name = (
            f"t{tx:+.3f}_{ty:+.3f}_{tz:+.3f}"
            f"_r{ax:+.5f}_{ay:+.5f}_{az:+.5f}"
            f"_s{point['scale']:.5f}"
        )
    if args.tag:
        name = f"{args.tag}_{name}"
    return Path(args.out_dir) / f"tracks_{name}.root"


def run(point, args):
    out = output_for(point, args)
    # a truncated file is worse than a missing one, so require a plausible size
    if not args.overwrite and out.exists() and out.stat().st_size > 10_000:
        return point, 0, f"skip {out.name}"
    cmd = command(point, args)
    if args.dry_run:
        return point, 0, " ".join(cmd)
    log = out.with_suffix(".log")
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return point, rc, f"{'ok  ' if rc == 0 else 'FAIL'} {out.name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="the fixed sim file")
    parser.add_argument("--out-dir", default="scan")
    parser.add_argument("--detector", default="ldmx-det-v15-8gev-no-cals")
    parser.add_argument("--input-pass-name", default="")
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--scan",
        nargs="+",
        default=["tz"],
        choices=[*SCANS, "combined", "all"],
        help="which parameters to scan",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scans = [*SCANS, "combined"] if "all" in args.scan else args.scan
    points = grid(scans)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"{len(points)} points over {', '.join(scans)} -> {args.out_dir}")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _, rc, msg in pool.map(lambda pt: run(pt, args), points):
            print(msg, flush=True)
            failures += rc != 0

    if failures:
        print(f"{failures} of {len(points)} points failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
