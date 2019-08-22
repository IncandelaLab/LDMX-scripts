OVERVIEW OF MIP TRACKING ADDITIONS:

The directory tracking_dev contains all necessary tracking_scripts.  The directory hits contains scripts for generating input files for the tracking routines.


HITS:

sim1000MeV_parent.py:
Modified version of BDT code.  Running this script will generate the hits, momenta, and particleinfo .txt files required as input for the tracking algorithm.  Note that this script creates files for signal only; simbkg_parent.py must be ran if the background files are desired.  Currently set to run on 10k events.
* Note:  Errors may occur if hits_copy_trees is empty.

simbkg_parent.py:
Same as the above, except for background events.  Currently set to 10k events; takes around a half hour to run.

submitHitsSim*.sh:
bash script for submitting the above jobs to slac.
* Note:  By default, submitTreeMaker creates a large number of jobs in submitHits* when ran on the corresponding .conf files.  In practice, only one of these jobs is necessary to generate the text files; the rest may be deleted.  (Also, running multiple jobs in parallel may cause errors in the generated text files.)

Once the text files are generated, they should be moved into tracking_dev so the tracking algorithm can read them.  (Alternatively, one could change the file paths in the various tracks_*.py files to point to the files in the hits directory.)


TRACKING_DEV:

simplot.py, simplotv2.py:
Plotting routines for visualizing individual events without running the full tracking algorithm.  I believe both work; v2 is arguably cleaner.

tracks_inradii.py:
The latest version of the tracking algorithm.  inradii:
- Performs a first pass over the ecal, checking for straight tracks (1 hop) with a minimum length of 4.
- Performs a second pass over the remaining hits, checking for all tracks (1 hop) with a minimum length of 5.
- Considers hits inside the electron radii of containment if the electron and the photon are near-parallel.
- Plots results in 3d one event at a time if the corresponding var is set.  Two color schemes are available:  by unique particle ID, and by hit region.
- Has a different structure from the other versions to improve ease of use.
- Unlike in previous versions, isolatedEnd is ignored, as we've found that this increases accuracy with little signal loss.
* Note:  No input args are required.  Run with 'python tracks_inradii.py'.

tracks_purity.py:
WIP.  Based on an earlier version of tracks_inradii, tracks_purity is meant to check whether the tracks found by the algorithm are "pure"; i.e. whether most of the hits are produced by a single particle.
* Note:  Due to the incompleteness of the parent information, this algorithm may overestimate the actual track puritites.

tracks_ambig.py:
Instead of adding the first valid neighbor it finds to the track, tracks_ambig creates a list of all possible candidate neighbors and then chooses one based on proximity.  (However, neighbors in the same layer are given a lower priority.)

tracks_weighted_avg.py:
First version of the code to use an exponentially weighted moving average of the previous slopes between hits to eliminate crooked "fake" tracks.  tracks_inradii does this as well.

tracks_vertical.py:
Earlier version that searches for hits in the same layer in addition to hits in neighboring layers, in an attempt to find high-angle tracks radiating outwards from the electron and photon trajectories.  May be buggy.
* Note:  High-angle tracks appear in signal.  This version is questionably useful.


