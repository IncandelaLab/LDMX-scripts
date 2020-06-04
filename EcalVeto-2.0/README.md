# EcalVeto-2.0

ECal veto analysis using `ldmx-sw` `v2.0` or later. Compatible with `v12` LDMX samples.

## Getting started
Start by setting up your environment and `ldmx-sw` following the instructions [here](https://docs.google.com/presentation/d/1J58bSqR5rQvenrmgmnGBAYgQ4WQo8Wftb8GhfYRwB_w/edit#slide=id.g659a216f71_0_77u). If you have already set up `ldmx-sw`, you just need to source the environment script you created following those instructions.

## Setting up the analyzer
Check out the `ldmx-analysis` directory from github in your work area:

```
cd <your_work_dir>
git clone https://github.com/LDMX-Software/ldmx-analysis.git
cd ldmx-analysis
```

Then, build and install the package following the instructions in the README [here](https://github.com/LDMX-Software/ldmx-analysis). You can start your analysis from the `ECalVetoAnalyzer`. This processor analyzes LDMX event files and produces ntuples with the variables we typically look at for an ECal veto analysis. The header can be found under the `include` directory and the implementation under the `src` directory. As you develop your own analysis looking at different variables and information, you can set up your own analyzer in a similar way. Any time you make a change to the processor, remember to re-build.

## Creating flat analysis ROOT trees from LDMX recon files
To run an analysis, check out this repository, which contains the scripts you will need to run:

```
cd <your_work_dir>
git clone https://github.com/IncandelaLab/LDMX-scripts.git .
cd EcalVeto-2.0
```

The template configuration file, `ecal_ana_tpl.py`, is used to specify the necessary parameters to run the analyzer (through `ldmx-app`). The input and output files are substitutable and are filled in by the `run_ldmx_app.py` script.

To test the configuration and analyzer on a single LDMX recon file, run the following command:

```python run_ldmx_app.py --prefix <some_prefix> --inputFile <full_path_to_input_file> --histOut <desired_full_path_to_output_directory> $PWD/ecal_ana_tpl.py ```

If this runs successfully, you should see an output file with root trees (and/or histograms) in the output directory you specified, which you should check interactively with `root` to make sure the content looks as expected. Remember to specify full paths in the arguments above.

### Batch submission
After you have successfully tested `run_ldmx_app.py` on a single file, you will usually want to run over a large number of recon files in order to produce analysis trees with a large number of events. The `ldmx_bsub.py` script can be used to submit jobs analyzing many input files in parallel to the LSF-based batch system. The configuration of the jobs is set up in the `sample.yml` file. Various parameters are defined in this file which specify the template configuration file, location of the input recon files you want to run over, location to which you want to save the output root files, etc. Please update these parameters as needed.

Once you are ready to submit, and confident that everything is configured correctly, you can submit jobs as follows:

```
python ldmx_bsub.py sample.yml
```

You can check the progress of your jobs with the `bjobs` command. Once the jobs are completed, you can merge the output root files together using the `hadd` command, and then you are ready to start producing some plots!
