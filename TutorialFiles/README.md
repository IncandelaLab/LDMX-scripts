# Quick Introduction to LDMX Software
Before you dive into the more in depth tutorials located in the sidebar on the right hand side of the [wiki](https://github.com/IncandelaLab/LDMX-scripts/wiki), here is a very quick guide on how to get some LDMX software working out-of-box!

## Installing pre-requisite software
* Install [XQuartz](https://www.xquartz.org/) (mac) or [Xming](http://www.straightrunning.com/XmingNotes/) (windows) 
* Install [Docker](https://docs.docker.com/engine/install/) and make an account 
* Ensure that you have a terminal app if you are on Mac or Linux, or Ubuntu (and WSL 2) downloaded if you are on Windows, then open up a terminal (shell/bash session)

## Installing LDMX Software Locally (i.e., on your own computer, no ssh or POD needed)
* Run the following commands (one line at a time) to install LDMX software
```
cd ~
git clone --recursive https://github.com/LDMX-Software/ldmx-sw.git 
source ldmx-sw/scripts/ldmx-env.sh
cd ldmx-sw; mkdir build; cd build;
ldmx cmake ..
ldmx make install
```
NOTE: You will need to run the source command every time you start a new terminal and want to use ldmx-sw, or you can add it to your ```.bashrc``` file (runs on startup) using ```vim```. You will also need your Docker desktop app running. 
* Now download LDMX-scripts
```
cd ~
# replace this next command with this repo for now (until I do a pull request)
# i.e., git clone -b v14_tutorial_files --recursive https://github.com/DuncanWilmot/LDMX-scripts LDMX-scripts-temp
git clone --recursive https://github.com/IncandelaLab/LDMX-scripts
```

## Generating Photonuclear (PN) Background samples
* We will use our ```v3_v14_pn_config.py``` file in the folder marked ```TutorialFiles``` in ```LDMX-scripts``` to generate some simulated samples using ldmx-sw v3 with v14 detector geometry. This will take a few minutes.
```
cd ~/LDMX-scripts/TutorialFiles
ldmx fire v3_v14_pn_config.py  
```
* The new samples will be saved as ```100k_v14_pn_testprod.root``` in ```~/LDMX-scripts/TutorialFiles```. Load ```ROOT``` and open a ```TBrowser``` to view them. You will need to first run the XQuartz or Xming app for this to work
```
ldmx root
# wait until you see root[0] in your terminal
new TBrowser()
```
This will open a graphical browser much like a file explorer but with plotting capabilities. Double click your root file to expand it and begin navigating through the branches/directories by double clicking them. When you get to LDMX variables with a leaf icon next to them, double clicking these will plot a histogram with their values

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/5ec4788f-31c7-43d5-ac0f-00cc32ec89ae)

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/9057a31e-2682-4c3a-85c7-543dfc588212)

## Flattening Trees for BDT
* You may need to download a couple extra ldmx-sw files before the next step. Go to the following directory and look for the files ```CaloTrigPrim.h``` and ```CaloCluster.h``` with the ```ls``` command. If you don't see them, execute the two ```wget``` commands
```
cd ~/ldmx-sw/Recon/include/Recon/Event
ls | grep -i '^calo[ct].*.h'
# if you don't see the two files execute the following two wget comands to download them
wget https://raw.githubusercontent.com/LDMX-Software/ldmx-sw/trunk/Recon/include/Recon/Event/CaloTrigPrim.h
wget https://raw.githubusercontent.com/LDMX-Software/ldmx-sw/trunk/Recon/include/Recon/Event/CaloCluster.h
```
* Now we will flatten the tree of the PN samples we just generated. This essentially just means picking all the leaves we want in the root file (e.g., nReadoutHits) and placing them in a single branch. This would be a bit like placing all the files you want to keep in your home directory, and deleting all subdirectories. In this case, we want all the ECal Veto variables needed for the BDT. We will accomplish this by using ```gabrielle_treeMaker.py``` on our samples ```100k_v14_pn_testprod.root```. The output with the flattened tree will be saved as ```100k_pn_v14_flatout.root```
```
cd ~/LDMX-scripts/TutorialFiles
# the following three lines are one single command!
ldmx python3 gabrielle_treeMaker.py \
-i $PWD/100k_v14_pn_testprod.root \
-o $PWD -g 100k_pn_v14_flatout
```

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/e0869bf0-bf22-47b9-bb7e-0c51357f0c14)

## Evaluate Gabrielle BDT
* We will now evaluate the pretrained v3 Gabrielle BDT on ```100k_pn_v14_flatout.root``` using a pickled set of weights in ```gabrielle_train_out_v3_weights.pkl``` obtained during training. The output will be saved as ```100k_pn_evalout.root```...Note: again, the three lines are one command, not three
```
ldmx python3 gabrielle_bdtEval.py \
-i $PWD/100k_pn_v14_flatout.root \
-o $PWD -g 100k_pn_v14_evalout
```
* Inspect the output with a ```TBrowser``` as before. You should see a new leaf called ```discValue_ECalVeto```. This discriminator value is a number between 0 and 1, which can more or less be interpreted as a measure of probability that an event is signal (so this number should be quite small for typical PN background events if your model is working well). We can set a threshold for this value and cut events accordingly to get rid of background while (ideally) preserving as much signal as possible.

## Skimming ROOT Files with Uproot

Much of the time we would like apply various cuts to our data. For example, we may want to get rid of all events containing 50 or more ECal readout hits. Or we may want to apply the trigger, which (for a 4 GeV electron beam) cuts all events depositing more than 1500 MeV (1.5 GeV) in the first 20 layers of the ECal. Some studies may require you to look only at fiducial (or non-fiducial) events...the list goes on. This is also called skimming a tree (or TTree, the main structure in a root file). We may want to save our skimmed data to a new set of root files, which must be done with ROOT. Alternatively, we may just want to store the skimmed data for particular variables in arrays to perform calculations with and/or plot. This can be done in python with the library ```uproot```.
* The basic syntax for opening a root file with uproot (within a .py script) is as follows
  ```python
  import uproot

  # create emply list for skimmed energy data (or some other variable of interest)
  totalE = []

  # location of root file
  # (when you have many files, instead iterate through directories with glob and for loops)
  filename = '/path/to/file.root'

  # list of branches to look at (here I just have number of ECal readout hits and total energy deposited in ECal using v14 naming conventions)
  branchList = ['EcalVeto_signal/nReadoutHits_', 'EcalVeto_signal/summedDet_'] 

  # open root file and generate arrays with variable data
  with uproot.open(filename)['LDMX_Events'] as t:
      raw_data = t.arrays(branchList)
      # this next line creates a 1D array of length equal to the # of events in the file
      # each element is a scalar number indicating the number of ECal readout hits in that event
      # when dealing with many variables it is useful to define dictionaries and naming functions
      # that way you can use for loops and easily point to the data using meaningful strings like 'EcalVeto' and 'nReadoutHits_'
      # instead of the index 0, which may change if I alter my original list
      nReadoutHits = data[branchList[0]]
  
      # now make a similar array where each element is the total energy desposited in the ECal (for each event in file)
      summedDet = data[branchList[1]]

      # now I'll make a logical comparison that evaluates True when I have less than 50 ECal hits in an event, and False otherwise
      MAX_NUM_ECAL_HITS = 50
      cut = (nReadoutHits < MAX_NUM_ECAL_HITS)

      # now apply the cut
      skimmed_data = {}
      for branch in branchList:
          skimmed_data[branch] = raw_data[branch][cut]

      # put the skimmed summedDet data into the empty list totalE
      summedDet_skimmed = skimmed_data[branchList[1]]
      totalE.extend(summedDet_skimmed)

  # Now you can plot totalE in a 1D histogram, or keep performing other operations

  ```

## Plotting LDMX Variables
* For official LDMX plots, check out the [plotting scripts and README](https://github.com/IncandelaLab/LDMX-scripts/tree/master/plotting). You can also keep reading to use the custom plotting code in ```TutorialFiles```

* Plot 1D histograms of LDMX variables (value vs. event density), using EcalVeto variables in this case, with ```v14_plotvars.py```. Plots will be saved as ```png``` image files in ```v14_4gev_plots``` (or some other specified directory with the default being the current working directory). Currently, ```v14_plotvars.py``` uses root files stored on POD (go read the code and comments). You can alter the file templates to use the root files you generated locally, or ssh into POD and run as is.
```
# ssh into POD
ssh -Y [username]@pod-login1.cnsi.ucsb.edu  # don't forget the -Y flag for visuals
```
You will need python 3 installed for this. This can be achieved by installing miniconda (if you dont already have it)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the insturctions to finish the installation
```
Verify the installation is successful by running ```conda info```
If you cannot run the conda command, check if you added the conda path to your PATH variable in your bashrc/zshrc file, e.g.,
```
export PATH="$HOME/miniconda3/bin:$PATH"
```
Now you can run the plotting script
```
cd LDMX-scripts/TutorialFiles
python3 v14_plotvars.py --save -o v14_4gev_plots
```
* You can view your plots by starting a ```jupyter notebook``` server
```
which jupyter # check that you have it
jupyter notebook --no-browser
```
The output of the last command should look something like this
```
To access the notebook, open this file in a browser:
        file:///home/duncansw/.local/share/jupyter/runtime/nbserver-207241-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=7ed46b4cda4b3265b75f68d569f2a722123b5fddc4b87f3f
     or http://127.0.0.1:8888/?token=7ed46b4cda4b3265b75f68d569f2a722123b5fddc4b87f3f
```
Note the port number (8888 in this case) and open a NEW terminal window then run the following command
```
ssh -fNT -L 8888:localhost:8888 [username]@pod-login1.cnsi.ucsb.edu
```
At this point, copy and paste one of the URL's above into a browser and navigate to your plots.

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/93bd56c6-4c6c-47ab-a8d5-8d498928c033)

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/8bb58c3d-2407-41d7-80b9-1640aa0117ab)




  
  





