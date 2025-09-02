# Photophobic ALPs
Outlines both the production of Photophobic ALPs undergoing an electron-positron conversion within the LDMX detector and scripts to calculate jet (substructure) observables utilized in discriminating between ALP signals and background. I have opted into utilizing non-standard python libraries such as `pylhe` and `fastjet`. Make sure these are installed using your favorite package manager or just simply by:

```
pip3 install pylhe
pip3 install fastjet
```

Further, make sure you have `denv` up and running as this will be the primary way I have written my code to interface with `ldmx-sw`.

## Signal Production
All of the production scripts, both signal and background events, were written to be ran on POD, but many of the necessary components can be altered to be ran locally. The primary difference will be done by interfacing everything through docker. Besides `ldmx-sw`,the primary tool necessary for event production is `MadGraph`. To run `MadGraph` on POD, we will utilize a docker image produced by Tamas. For people on POD, execute the following command to copy over the singularity image to your current working directory.

```
cp /home/vamitamas/madpydel_latest.sif .
```
Once we have the `MadGraph` singularity image in your current working directory, we can start the process of generating the necessary LHE files for the ALP to e+e- conversion. This is done by running the `alp_gen_config.py` configuration file or more specifically running:

```
python3 alp_gen_config.py --mg_config_tpl mg_config.txt.tpl --parser lhe_parser_uniform_z.py --n_events 20000 --seed 1 -min 200 -max 2000
```

which will submit sbatch jobs while:

1. Createing a `alpLHE` directory which will save the LHE files produced from `MadGraph`. 
2. Runing `MadGraph`, creating a single LHE file with all the necessary kinematic information of both the initial and final particle. We have configured the configuration files such that it deletes all the unnecessary junk produced by `MadGraph` and leaves the single LHE file which is then saved into the `mgLHE` directory.
3. Parsing the LHE file into two different files, one responsible for the production of the alp at the target and the other responsible for the decay of the alp. The position of the decay is uniformly distributed between `-min` and `-max`.

Once the LHE files are created, we can run `ldmx-sw` by:

```
python3 alp_ldmx_config.py --config_tpl ldmx_config.py.tpl --seed 1 -min 200 -max 2000
```

which will create a `alpLDMX` directory, storing the resulting root files. Again, this production will be done by submitting jobs to sbatch.  

## Background Production
For producing background events, we will be using the `ecal.deep_photo_nuclear` module from `ldmx-sw`. We however note that event production is VERY inefficient where `ldmx-sw` has to generate on the order of $10^5$ events just to generate a photonuclear event which decays deep in the Ecal. As such, it is recommended to split up the background event production into many jobs. Jobs can be created by the following command. 

```
python3 deep_photon_ldmx_config.py --config_tpl deep_photon_ldmx_config.py.tpl --n_events 1000 --decay_z 350 --seed 1 --n_jobs 10
```

Please adjust both `--n_events` and `--n_jobs` depending on how busy POD is. Further, trying to push `--decay_z` to larger values will lead to longer simulation time. It is extremely important to test these three values before submitting a large production job to POD as it can really take a long time (and timeout).

## Computing Jet Observables
Before computing the necessary jet observables, we will skim the root files obtained from `ldmx-sw` to simplify our work flow, which is achieved by:

```
denv python3 skim_root.py -min 200 -max 2000 --bkg_decay_z 350
```

This will create a `skimmed.root` file with the least amount of variables necessary for calculating jet observables. NOTE: in the code snippet above, we are using an arbitrary decay minimum of 200 and 2000. As such there are certain events which leave almost no energy in the Ecal. However, such events will be vetoed by requiring a minimum energy deposition in the Ecal which is necessary to calculate any jet observables using EcalRecHits. As such, we have placed a temporary `nReadoutHits < 10` cut in the `skim_root.py` file (please adjust depending on use). Finally, jet observables are calculated by:

```
denv python3 jet_observables.py
```












