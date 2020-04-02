# Training the ParticleNet-based model
------

## Setup the environment

You will need to set up Miniconda and install the needed packages if you start from scratch. This needs to be done only once.

#### Install miniconda (if you don't already have it)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the insturctions to finish the installation
```

Verify the installation is successful by running `conda info`.

If you cannot run `conda` command, check if the you added the conda path to your `PATH` variable in your bashrc/zshrc file, e.g., 

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

#### Set up the conda environment and install necessary packages

The following instruction is for training on Nvidia GPU w/ [CUDA](https://developer.nvidia.com/cuda-downloads) installed.

```bash
# create a new conda environment
conda create -n pytorch python=3.7

# activate the environment
conda activate pytorch

# install the necessary python packages
pip install numpy pandas scikit-learn scipy matplotlib tqdm

# we use uproot to access ROOT files
pip install uproot

# install pytorch
# pip install torch torchvision
#! install pytorch nightly version for now -- the current stable version (1.4.0) has a bug
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

```


## Run the training

Before running the code, make sure you have activated the training environment:

```bash
conda activate pytorch
```

The [train.py](train.py) is the main script to run the training. You can use it like:

```bash
python -u train.py --coord-ref none --optimizer ranger --start-lr 5e-3 --focal-loss-gamma 2 --network particle-net-lite --batch-size 128 --save-model-path models/ecal_coord-ref-none_particlenet-lite_focal2_ranger_lr5e-3/model --test-output-path test-output/ecal_coord-ref-none_particlenet-lite_focal2_ranger_lr5e-3/output --num-epochs 20 --num-workers 2 --device 'cuda:0' | tee ecal_coord-ref-none_particlenet-lite_focal2_ranger_lr5e-3.log
```

The meaning of each command line argument can be found w/ `python train.py -h` or inside the [train.py](train.py) file. The input signal and background files are set in the beginning of the [train.py](train.py) file, together w/ the number of events that will be taken from each process. We use the same number of events from each signal points (200k), and the same number of background events as the sum of all signal points (200k*4 = 800k) for the training, to avoid bias to a specific signal point. By default, we only use 80% of all available events for the training -- the rest ("validation sample") will be used for evaluating the performance of the trained model. 

The training is performed for 20 epochs (set by `--num-epochs`), w/ each epoch going over all the signal and background events. At the end of each epoch, a model snapshot is saved to the path set by `--save-model-path`. At the end of the training, the model snapshot w/ the best accuracy is used for evaluation -- the output will be saved to `--test-output-path`, and a number of performance metrics will be printed to the screen, e.g., the signal efficiencies at background efficiencies of 1e-3, 1e-4, 1e-5, and 1e-6 (the signal eff. at bkg=1e-6 is typically not very accurate due to low stats in the validation sample). The trained model will also be exported to ONNX format for integration into the ldmx-sw framework at the end of the training.



## Run the prediction/evaluation

The [eval.py](eval.py) script can also be used to apply the trained network to the input files. Unlike the [train.py](train.py) file, `eval.py` will not load all signal and background files into memory together, but will run over each file separately (and write a separate output for each input file). The command line options are very similar as those for the `train.py` script.

## References: 
  - ParticleNet: https://arxiv.org/abs/1902.08570