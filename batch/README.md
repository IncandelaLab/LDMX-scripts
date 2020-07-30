# Instructions for batch submission on the UCSB POD cluster

This repository contains the run scripts (`run_prod.py` and `run_ana.py`) and batch submission scripts (`sbatch_prod.py` and `sbatch_ana.py`) that can be used to run simulation or analysis jobs interactively and then submit them to the slurm batch system on the POD cluster. The configuration of jobs is set up using yaml files.

## Simulation jobs
The following instructions apply to the submission of simulation jobs.

### Setting up
Go into your working directory. Load singularity and point to the image file to be used:
```
module load singularity
export LDMX_PRODUCTION_IMG=$PWD/ldmx_pro_<tag>.sif
```

Create the singularity image (if it doesn't exist already):

`singularity build $LDMX_PRODUCTION_IMG docker://ldmx/pro:<tag>`

If you are using your own custom image, just modify the docker repository to pull from as appropriate, i.e. `docker://<your_user_name>/<your_repo_name>:<tag_name>`.

### Configuring, testing, and submitting jobs
Use the following command to submit jobs to the batch system (add the `-t` option if you just want to test the submission script first):

`python sbatch_prod.py -c <yaml_file> -s <starting_seed>`

Always run first with the `-t` option to make sure everything is set up correctly.

The `<yaml_file>` is used to define the paths, run script, template config file, and number of jobs to be submitted. An example yaml file is provided in `ecal_prod_config.yml`, please modify the parameters as needed. A template configuration file that can be used to produce ECAL photo-nuclear events is provided in `ecal_pn.py.tpl`. You may want to change the `maxEvents` parameter in this file to run fewer events per job, e.g. for testing purposes. Configuration files to produce other types of samples of interest should be added to this repository.

The `<starting_seed>` value should be incremented with every batch submission for the same type of job.

Before submitting jobs to the batch system, you should first run the production script (`run_prod.py`) interactively to make sure there are no issues and check the output file in `root`. You can run an example command from one of the slurm submit scripts created by running the above command with the `-t` option to create a test file.

If everything looks fine, after submitting batch jobs to the slurm system you can check their status with the command `squeue -u <username>`. You can find a list of common slurm commands [here](https://slurm.schedmd.com/quickstart.html). Once jobs are completed successfully, the output files should show up in the output directory specified in the yaml file.

## Analysis jobs
The following instructions apply to the submission of analysis jobs.

### Setting up
Go into your working directory. Load singularity and point to the image file to be used:
```
module load singularity
export LDMX_PRODUCTION_IMG=$PWD/ldmx_ana_<tag>.sif
```

Create the singularity image (if it doesn't exist already) from a docker image that includes the `ldmx-sw` and `ldmx-analysis` modules:

`singularity build $LDMX_PRODUCTION_IMG docker://<your_user_name>/<your_repo_name>:<tag>`

An example image can be found at `docker://valentinadutta/ldmx-ana:latest`. You can build your own custom image, containing your latest developments, following the instructions [here](https://hub.docker.com/r/valentinadutta/ldmx-ana).

### Configuring, testing, and submitting jobs
Use the following command to submit jobs to the batch system (add the `-t` option if you just want to test the submission script first):

`python sbatch_ana.py -c <yaml_file>`

Always run first with the `-t` option to make sure everything is set up correctly.

The `<yaml_file>` is used to define the paths, run script, template config file, and input directory containing the files you want to analyze. An example yaml file is provided in `sample.yml`, please modify the parameters as needed. A template configuration file that can be used to run the `ECalVetoAnalyzer` processor and produce `root` trees with the standard ECal veto variables is provided in `ecal_ana.py.tpl`.

Before submitting jobs to the batch system, you should first run the analysis script (`run_ana.py`) interactively to make sure there are no issues and check the output file in `root`. You can run an example command like the test command printed out by running the above command with the `-t` option to create a test file (remove `--test` from the list of arguments for `run_ana.py` to actually run the configuration).

If everything looks fine, you can go ahead and submit batch jobs to the slurm system. Once jobs are completed successfully, the output files should show up in the output directory specified in the yaml file.
