# LDMX-scripts

# Description of analysis workflow

ECal veto analysis based on `ldmx-sw` v1.7 or earlier. Compatible with v9 LDMX samples.

## Creating flat analysis ROOT trees from LDMX recon files
Start by checking out this repository:

```
cd <your_work_dir>
git clone https://github.com/IncandelaLab/LDMX-scripts.git .
cd EcalVeto
```

The script <code>bdtTreeMaker.py</code> (under EcalVeto) can be used to run over the LDMX files which contain both sim and recon information, and produce flat ROOT trees which contain branches corresponding to the variables of interest for our analysis. In order to run this script interactively over a single input file, run the following command: </br>
<code>python bdtTreeMaker.py --interactive -i <path_to_input_file> -o <desired_path_to_output_directory> -f <desired_name_of_output_file> </code>
A list of locations in which to find various versions of the LDMX recon files can usually be found here: https://confluence.slac.stanford.edu/display/MME/Monte+Carlo+Production.
If you are running over signal (rather than photonuclear background) files, add the option <code>--signal</code> to the command above. If you want to test with just a few events rather than a full file, uncomment the lines that say:</br>
  <code>            #if self.event_count>1000:</code></br>
  <code>            #    return</code>
  
  ### Batch submission
  After testing the tree maker script interactively, usually you will want to run over a large number of signal and background recon files in order to produce analysis trees with a large number of events. The <code>submitTreeMaker.py</code> script can be used to set up the submission of parallel jobs to the LSF-based batch system for this purpose. The configuration of the jobs is set up in the <code>process.conf</code> file. Various parameters are defined in this conf file which are then picked up by <code>submitTreeMaker.py</code> to configure the batch jobs. Edit this conf file as appropriate, updating the locations of the tree maker script, input and output directories, etc. Then, run the following command: </br>
<code>python submitTreeMaker.py process.conf</code></br>
This will create a shell script (by default this is named <code>submittree.sh</code>), check this to make sure that the jobs you wish to submit have been correctly configured, and then submit them to the batch system using </br>
<code>source submittree.sh</code>

## BDT training
### Separation of training and test samples
The BDT training and testing needs to be done on separate events in order to get an unbiased evaluation of the BDT performance. Once you have produced ROOT trees as described above for both signal and background, go to the output directory and merge the output files for each process together separately, e.g.:</br>
<code> hadd bkg_tree.root bkg_tree_\*.root </code></br>
<code> hadd 1.0_tree.root 1.0_tree_\*.root </code></br>
<code> hadd 0.1_tree.root 0.1_tree_\*.root </code></br>
<code> hadd 0.01_tree.root 0.01_tree_\*.root </code></br>
<code> hadd 0.001_tree.root 0.001_tree_\*.root </code></br>
Then, the macro <code>makeSubTrees.C</code> can be used (commenting and uncommenting various lines to perform the desired operation) to separate out the background and signal training and test samples (labeled as subtrees in the macro). For the training, we use 1.25M events from the background tree, and 312.5k events from each of the signal mass point trees (for 1.0, 0.1, 0.01, 0.001 GeV A' masses). The signal training subtrees should be merged into a signal training tree:</br>
<code> hadd signal_bdttrain_tree.root 1.0_bdttrain_subtree.root 0.1_bdttrain_subtree.root 0.01_bdttrain_subtree.root 0.001_bdttrain_subtree.root </code></br>

### Training
The BDT can now be trained using the <code>bdtMaker.py</code> script. To train using the training trees you have just created, use the following command:</br>
<code> python bdtMaker.py --bkg_file <your_path>/bkg_bdttrain_subtree.root --sig_file <your_path>/signal_bdttrain_subtree.root --out_name <bdt_name> </code></br>
Note that the BDT training may take some time which increases with the number of variables. The output pkl file from xgboost should get saved to <code><bdt_name_somenumber>/<bdt_name_somenumber>_weights.pkl</code>.
  
### Testing
Now that you have trained your BDT, you will want to evaluate the BDT scores for your test events in order to see how well the BDT performs. This can be done using the <code>bdtEval.py</code> script. To run it interactively over one of your testing trees, you can do the following, for example, to add the BDT predictions per event to the flat ROOT trees for the 1.0 GeV A' mass test sample:</br>
<code>python bdtEval.py -o <desired_name_of_output_root_file.root> --outdir <desired_path_to_output_directory> -p <path_to_bdt_pkl_file> -i <your_path>/1.0_bdttest_subtree.root</code></br>
  
If you have a large number of files to run over for the BDT evaluation (e.g., if you want to evaluate the BDT for the full 1e14 EoT equivalent ECAL PN sample), you can also ust batch submission for this step. Simply edit <code>addbdt.conf</code> as needed to specify the locations of your evaluation script, input and output directories and any other settings that need to be changed, and submit the batch jobs as follows:</br>
<code> python submitTreeMaker.py addbdt.conf</code></br>
<code> source submitbdt.sh </code>

## Analysis on flat trees
Coming soon!
