# B-field distortion scan

How well do we need to know where the dipole sits before the reconstructed
momentum scale is biased? The CKF reconstructs with a field map object that is
separate from the field Geant4 simulated with, so we can deliberately mis-set
the reconstruction field — shift it, rotate it, rescale it — with the events
held fixed, and read off how the track parameters respond.

The scan therefore simulates **once** and reconstructs **many times**. Nothing
in the simulation changes between scan points.

An earlier version of this study is in section 3.3.5 of the LDMX detector paper
([arXiv:2508.11833](https://arxiv.org/abs/2508.11833)): roughly 50 MeV/mm of
momentum shift per mm of field z-displacement, with rotations only mattering
above about 100 mrad. Those numbers are from the older tracking and a 4 GeV
v14 detector; this reproduces them on CKF tracking.

## Requirements

An ldmx-sw with the `bfield_*` knobs on `CKFProcessor` and `GSFProcessor`
(`bfield_translation`, `bfield_rotation`, `bfield_pivot`, `bfield_scale`).
Without them `reco.py` will fail immediately with a `KeyError` from the config
parameter validation.

## Running it

Generate the reference sample once and leave it alone. Pin the run number: it
is the seed, and the whole method rests on every scan point seeing identical
events.

```sh
# recoil: mono-energetic electrons from the target
denv fire gun_sim.py --mode target --energy 4.0 --n-events 20000 --run 1

# recoil: uniform in energy and angle, for the differential slopes
denv fire gun_sim.py --mode target --min-energy 0 --max-energy 8 --n-events 50000 --run 2

# tagger: the beam-electron stand-in
denv fire gun_sim.py --mode tagger --beam-energy 8.0 --n-events 20000 --run 3
```

Then scan. Start with the beam axis alone — it is the sensitive direction and
the one the paper quotes:

```sh
./run_scan.py --input-file events_target_mono_4gev_*.root --out-dir scan/ \
    --scan tz --jobs 4
```

`--scan all` does all seven parameters plus a handful of combined
mis-placements. `--dry-run` prints the commands without running them, and
re-running skips points that already have output, so an interrupted scan can
just be restarted.

Flatten the results and analyse:

```sh
denv python3 make_ntuple.py scan/tracks_*.root -o ntuples/
```

## Known-good starting point

A 300-event smoke test of the whole chain — 4 GeV mono-energetic gun from the
target, `ldmx-det-v15-8gev-no-cals`, ldmx-sw at `03b0a849` plus the knobs —
gives, for the mean reconstructed recoil momentum at the target:

| beam-axis shift | mean p (MeV) | recoil tracks |
| --- | --- | --- |
| −5 mm | 4003.5 | 119 |
| nominal | 4065.0 | 119 |
| +5 mm | 4128.9 | 119 |

So ≈13 MeV/mm, linear and symmetric about nominal, with the track count
unchanged — the momentum scale moves while the efficiency does not, which is
the asymmetry the study is about. Note this is *not* the paper's ~50 MeV/mm:
different detector, different tracking, and a gun from the target rather than a
scattered beam electron. Reproduce this before trusting a bigger scan; if you
get a different number, something has changed and it is worth knowing what.

That is a mean over all tracks, which the radiative tail pulls around. For real
results fit the core of the residual, as below.

## Conventions

The distortion is in the **LDMX global frame**: x is the bend plane, y is
vertical, z is the beam. A positive translation moves the *magnet*, so the
field at a fixed point becomes the nominal field from further upstream.
Rotations are about `--pivot`, which defaults to the field-map origin at
z = −400 mm, the centre of the dipole.

Be careful with the map edges. The field map only covers ±250 mm in x, ±70 mm
in y and ±1500 mm in z about z = −400 mm, and outside it ACTS returns zero
field rather than an error. A large distortion can walk part of a trajectory
out of the map and show up as a fake nonlinearity — the vertical axis is the
tight one. Before trusting any large-distortion point, dump |B| along a nominal
trajectory and check it never drops to zero inside the tracker.

## What to measure

Because this is a gun study the observable is the residual `p_reco − p_truth`
(column `dp`, or `dp_over_p`), not a peak position on a falling spectrum. Fit
its core — a Gaussian, or a Crystal Ball if the radiative tail matters — and
take the fitted mean as the bias at that scan point.

The deliverable is a table of the slope of each observable against each
distortion parameter, tagger and recoil separately. Things worth knowing while
you do it:

- **Seeds are distortion-independent by construction.** `SeedFinderProcessor`
  uses a constant 1.5 T analytic fit, so seeding efficiency is flat and any
  efficiency loss is CKF-side.
- **Expect the bias not to be flat in momentum.** The recoil sits on a steep
  field gradient (dB/dz ≈ −5 mT/mm at the target) and low-momentum tracks curve
  more. If the slope comes out flat in p and θ, be suspicious and cross-check.
- **The tagger should be far less sensitive** — it sits in the −1.5 T plateau
  rather than on the falling edge. The ratio of field gradients predicts the
  ratio of sensitivities; a mismatch means something in the setup is wrong.
- **Efficiency is not a usable monitor.** A distortion large enough to bias the
  momentum scale may well be invisible in efficiency. That asymmetry is worth
  quantifying, because it says we could not catch a mis-set field in data that
  way.
