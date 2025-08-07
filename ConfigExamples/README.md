# Example config files

## How to get ldmx-sw

See details in https://ldmx-software.github.io/developing/getting-started.html

Do the setup once:
```
curl -s https://raw.githubusercontent.com/tomeichlersmith/denv/main/install | sh 
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh |\
  bash -s -- --to ~/.local/bin
```

Add the following to the `.bashrc`
```
module load apptainer
eval "$(just --completions bash)"
```

Make a project every time you do something new
```
mkdir Project1
cd Project1
git clone --recursive https://github.com/LDMX-Software/ldmx-sw.git
cd ldmx-sw
just init
just compile
```

Run a given config file
```
just fire config.py
```

# Example config files

This repo has a few example config files to run

## Hard brem process

The `hard_brem_only_config.py` has a trigger skim with settings of
```
recoil_max_p = 3000.,brem_min_e = 5000.
```
This has an efficiency of 1/10k (0.01%) events and timing of 1 event/100min 


## Electron gun

The `electron_gun_config.py` shoots an electron in front of the ECAL into the ECAL. The main settings are in
```
mpgGen = generators.multi( "mgpGen" )                                                                           
mpgGen.vertex = [ 0., 0., 200. ] # mm                                                                                                                              
mpgGen.nParticles = 1
mpgGen.pdgID = 11
mpgGen.enablePoisson = False #True                                                                                     

# import math
# import numpy as np
# theta = math.radians(5.65)
# beamEnergyMeV=1000*beamEnergy
# px = beamEnergyMeV*math.sin(theta)
# py = 0.;
# pz= beamEnergyMeV*math.cos(theta)
px = 0.
py = 0.
pz= 3000.
mpgGen.momentum = [ px, py, pz ]
```

