# ParticleNet for the LDMX ECal veto
------

## Introduction

ParticleNet is a convolutional neural network architecture designed for jet tagging.  This repository contains an implementation of ParticleNet designed to serve as a powerful veto for the LDMX electromagnetic calorimeter (ECal).

ParticleNet's workflow is straightforward.  First, ROOT files produced by the ldmx-sw software framework--typically for signal events at various masses and photonuclear background events--must be passed to a processing script that skims out most of the background and produces output ROOT files formatted specifically for easy use in ParticleNet.  Second, these output files are passed to ParticleNet, which trains on 80% of the events and uses the remaining 20% for validation.  Once finished, the trained ParticleNet model may then be evaluated on more events to check its performance.  This is faciliated by a jupyter notebook containing a number of plotting routines for checking ROC curves, pT bias, and more.

## Training the ParticleNet-based model

### Setup the environment

You will need to set up Miniconda and install the needed packages if you start from scratch. This needs to be done only once.

#### Install miniconda (if you don't already have it)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the insturctions to finish the installation
```

Verify the installation is successful by running `conda info`.

If you cannot run the `conda` command, check if the you added the conda path to your `PATH` variable in your bashrc/zshrc file, e.g., 

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

#### Set up the conda environment and install necessary packages

The following instructions are for training on Nvidia GPU w/ [CUDA](https://developer.nvidia.com/cuda-downloads) installed.

To set up the conda environment, use:

```bash
# create a new conda environment
# OLD COMMAND:  conda create -n pytorch python=3.7
# ROOT is now required for efficiency lazy data loading
conda create -c conda-forge --name torchroot root

# activate the environment
conda activate torchroot

# install the necessary python packages
# psutil is also recommended if monitoring GPU usage becomes necessary
pip install numpy pandas scikit-learn scipy matplotlib tqdm pyarrow

# we (sometimes) use uproot to access ROOT files
pip install uproot

# install pytorch
# pip install torch torchvision
#! install pytorch nightly version for now -- the current stable version (1.4.0) has a bug
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

```

### Preprocess the input files

The training code requires skimmed and partially-processed root files as input.  [file\_processor.py](file_processor.py) is responsible for generating these files from ldmx-sw simulation output.  In addition to performing a simple preselection on all input events that removes ~95% of all PN background events and <5% of signal events, the processing script only writes information to the processed root files that's necessary for ParticleNet and the plotting notebook.

To generate input files, edit the filepaths in [file\_processor.py](file_processor.py) to point to your signal and background ldmx-sw root files and run the script.  If you're working on POD, you can submit this to the batch system with `sbatch file_processor.py`.  With 20 processes and O(200k) v3.0.0 events for each signal and background category, it should take about an hour to run.

(If possible, it's easier to use exising input files, such as those in `/home/pmasterson/GraphNet_input/v12/processed`, rather than generating your own.)


### Run the training

Before running the code, make sure you have activated the training environment:

```bash
conda activate pytorch
```

The [train.py](train.py) is the main script to run the training. If you're running ParticleNet directly on a machine with GPUs, such as pod-gpu, you can run a demo version in the command line with:

```bash
python -u train.py --coord-ref none --optimizer ranger --start-lr 5e-3 --focal-loss-gamma 2 --network particle-net-lite --batch-size 128 --save-model-path models/ecal_coord-ref-none_particlenet-lite_focal2_ranger_lr5e-3/model --test-output-path test-output/ecal_coord-ref-none_particlenet-lite_focal2_ranger_lr5e-3/output --num-epochs 20 --num-workers 16 --device 'cuda:0' --demo | tee ecal_coord-ref-none_particlenet-lite_focal2_ranger_lr5e-3.log
```

Note that the `--demo` option runs the training on a very small sample of events instead of the full ~400k per category.  Since it takes much longer to run the full training, full training jobs should be submitted to the batch system instead.  To do this, modify the model names/paths in [run\_training.job](run_training.job) to something descriptive and run:

```bash
sbatch run_training.job
```

The meaning of each command line argument in the base command can be found w/ `python train.py -h` or inside the [train.py](train.py) file. The input signal and background files are set in the beginning of the [train.py](train.py) file, together w/ the number of events that will be taken from each process. We use the same number of events from each signal points (was 200k, now 400k), and the same number of background events as the sum of all signal points (400k\*4 = 1600k) for the training, to avoid bias to a specific signal point. By default, we only use 80% of all available events for the training -- the rest ("validation sample") will be used for evaluating the performance of the trained model. 

The training is performed for 20 epochs (set by `--num-epochs`), w/ each epoch going over all the signal and background events. At the end of each epoch, a model snapshot is saved to the path set by `--save-model-path`. At the end of the training, the model snapshot w/ the best accuracy is used for evaluation -- the output will be saved to `--test-output-path`, and a number of performance metrics will be printed to the screen, e.g., the signal efficiencies at background efficiencies of 1e-3, 1e-4, 1e-5, and 1e-6 (the signal eff. at bkg=1e-6 is typically not very accurate due to low stats in the validation sample).


### Run the prediction/evaluation

After training, the next step for generating ROC curves and other plots is to run [train.py](train.py) in prediction mode.  This is most easily done by using the corresponding slurm script ([run\_prediction.job](run_prediction.job)).

The [eval.py](eval.py) script can also be used to apply the trained network to the input files. Unlike the [train.py](train.py) file, `eval.py` will not load all signal and background files in the same data set together, but will run over each file separately (and write a separate output for each input file). The command line options are very similar as those for the `train.py` script.  On POD, it's once again easiest to use one of the slurm job scripts, [run\_eval.job](run_eval.py).

## Plotting with Jupyter (on POD)

Once all of your training and evaluation is done, it's time to plot the results!  If you're using ParticleNet on a computing cluster that you've ssh'ed into, like POD, you'll need to start up a Jupyter notebook server first, then set up an ssh tunnel that lets you access that notebook in your web browser.  Starting the server is straightforward:

```jupyter notebook --no-browser```

The output of this command should include something like the following lines near the end:

```
...
To access the notebook, open this file in a browser:
    file:///run/user/1000/jupyter/nbserver-21524-open.html
Or copy and paste one of these URLs:
    http://localhost:8888/?token=702b7225978109ddc0c833098a19364bc3b0da73ffb3b6eb
...
```

Take note of the port number after localhost, 8888 in the above example.  (It's almost always 8888 unless you have another tunnel/server set up somewhere.)  You'll then need to set up an ssh tunnel on your local machine that connects to this port.  Open up a terminal and run the following command:

```
ssh -fNT -L 8888:localhost:8888 [username]@pod-login1.cnsi.ucsb.edu
```

Or, if you fired up the server on pod-gpu instead of pod-login for whatever reason,

```bash
ssh -fNT -L 8888:localhost:8888 [username]@pod-gpu -J [username]@pod-login1.cnsi.ucsb.edu
```

At this point, you should be able to paste one of the above URLs into a browser on your local machine and open up the notebook.


## References: 
  - ParticleNet: https://arxiv.org/abs/1902.08570
