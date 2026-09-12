#!/bin/python
"""Reference gun sample for the B-field distortion study.

Simulation only -- no tracking. The output of this is re-reconstructed once per
scan point by reco.py, so it must be generated once and then left alone.

Two gun modes:

  target  electrons fired from the target, standing in for the recoil. Use
          --energy for a mono-energetic sample (the residual p_reco - p_truth
          is then clean) or --min-energy/--max-energy for a uniform one.
  tagger  a single beam-energy electron fired from upstream of the tagger,
          standing in for the beam electron.

Example:

  fire gun_sim.py --mode target --energy 4.0 --n-events 20000 --run 1
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["target", "tagger"], default="target")
parser.add_argument(
    "--energy", type=float, default=None, help="mono-energetic gun energy [GeV]"
)
parser.add_argument("--min-energy", type=float, default=0.0)
parser.add_argument("--max-energy", type=float, default=8.0)
parser.add_argument(
    "--angle", type=float, default=60.0, help="maximum polar angle [degrees]"
)
parser.add_argument("--beam-energy", type=float, default=8.0)
parser.add_argument("--detector", default="ldmx-det-v15-8gev-no-cals")
parser.add_argument("--n-events", type=int, default=20000)
parser.add_argument("--run", type=int, default=1, help="pin this, it is the seed")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

from LDMX.Framework import ldmxcfg

p = ldmxcfg.Process("sim")
p.max_events = args.n_events
p.run = args.run

if args.mode == "tagger":
    tag = f"tagger_{args.beam_energy:g}gev"
elif args.energy is not None:
    tag = f"target_mono_{args.energy:g}gev"
else:
    tag = f"target_uniform_{args.min_energy:g}_{args.max_energy:g}gev"

name = f"{tag}_{args.detector}_run{args.run:04d}_n{args.n_events}"
p.output_files = [f"{args.out_dir}/events_{name}.root"]
p.histogram_file = f"{args.out_dir}/hist_{name}.root"

from LDMX.SimCore import generators, simulator

sim = simulator.Simulator("sim")
sim.set_detector(args.detector, include_scoring_planes_minimal=True)

if args.mode == "tagger":
    sim.description = f"single {args.beam_energy} GeV electron upstream of the tagger"
    gun = {
        8.0: generators.single_8gev_e_upstream_tagger,
        4.0: generators.single_4gev_e_upstream_tagger,
        1.2: generators.single_1pt2gev_e_upstream_tagger,
    }.get(args.beam_energy)
    if gun is None:
        parser.error("--beam-energy must be one of 1.2, 4.0, 8.0 in tagger mode")
    sim.generators = [gun()]
else:
    sim.description = "electrons shot from the target"
    if args.energy is not None:
        energy_cmds = [
            "/gps/ene/type Mono",
            f"/gps/ene/mono {args.energy} GeV",
        ]
    else:
        energy_cmds = [
            "/gps/ene/type Lin",
            f"/gps/ene/min {args.min_energy} GeV",
            f"/gps/ene/max {args.max_energy} GeV",
            "/gps/ene/gradient 0",
            "/gps/ene/intercept 1",
        ]
    sim.generators = [
        generators.Gps(
            instance_name="target_electrons",
            init_commands=[
                "/gps/particle e-",
                "/gps/pos/type Point",
                "/gps/pos/centre 0 0 0 mm",
                # default GPS direction is -z, rotate the angular frame to +z
                "/gps/direction 0 0 1",
                "/gps/ang/rot1 1 0 0",
                "/gps/ang/rot2 0 -1 0",
                "/gps/ang/type cos",
                "/gps/ang/mintheta 0 deg",
                f"/gps/ang/maxtheta {args.angle} deg",
                "/gps/ang/minphi 0 deg",
                "/gps/ang/maxphi 360 deg",
                *energy_cmds,
                "/gps/number 1",
            ],
        )
    ]

p.sequence = [sim]
