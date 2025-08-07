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

HardBremOnly.py has a trigger skim with settings of
```
recoil_max_p = 3000.,brem_min_e = 5000.
```
This has an efficiency of 1/10k (0.01%) events and timing of 1 event/100min 
