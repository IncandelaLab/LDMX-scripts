# Python Implimentation of BDT

## Purpose: Faster development and eliminates need for ldmx-analysis and other dependencies
## Requirments: Working install of `ldmx-sw-v2.3.0` or greater and `v12` samples.
##       +     Only tested with container including numpy, xgboost, and matplotlib packages.
             
Currently set to to run seg-mip BDT.

Example TreeMaker command to make flat trees from event samples:
```
ldmx python3 treeMaker.py -i <absolute_path_to_inputs> -g <labels_for_input_eg_PN> --out <absolute_outdirs> -m <max_events>
```
`--indirs` can be used to run over all files from given directories. More information can be found in `mods/ROOTmanager.py`

Example bdtMaker command to train BDT:
```
ldmx python3 bdtMaker.py -s <path_to_combined_signal_training_file> -b <path_to_bkg_file>
```
There's more options for this too but the command gets long anough as is and I usually just change a few numbers in the script rather than using any parsing. You'll get a warning from XGBoost but it's fine, it's working. It just takes a while. I'd suggest training and evaluate on 100 event background and signal samples first just too see how it works.

Example bdtEval command to evaluate trained BDT on test samples:
```
ldmx python3 bdtEval.py -i <absolute_path_to_testing> -g <labels> --out <absolute_path_output_file_name>
```
