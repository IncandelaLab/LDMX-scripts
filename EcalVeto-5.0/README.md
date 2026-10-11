# EcalVeto BDT — training and analysis procedure

A complete procedure for training, deploying and evaluating an EcalVeto BDT:
prepare a training sample, train, score a large sample, plot, choose an operating
point, and understand what the model uses.

    training  prepare a training sample, train a BDT, convert it for ldmx-sw
    scoring   score the trained model over a large sample
    plotting  ROC curves, discriminator distributions, summary panels
    cutflow   the cutflow and the Punzi figure of merit -> the operating point
    shap      which features the model uses

Everything runs inside the container: `denv python3 <script>`.
Exception: `smart_multifire.py` runs with plain `python3`, because it calls `sbatch`.

Throughout, `<MODEL>` is your model's name, `<PASS>` a label you choose for a
processing pass, and `<...dir>` a directory of ROOT files.

--- 

# 1. Training — `training/`

| file | use |
|---|---|
| `digi_tracking_ecalveto_config.py` | Sample prep: digi + tracking + EcalVeto + HCal. Use when the sample has no digitized/tracking information. |
| `bdtMaker.py` | XGBoost training to create .pkl model |
| `train_v15R.sh` | Example of training command to train mutliple BDTs with different hyperparameters |
| `pickle_to_onnx.py` | `.pkl` -> `.onnx` for ldmx-sw. |


**The training sample must be separate from the sample you evaluate on.** A model
scored on events it trained on will look far better than it is.

Training does not need the huge sample. Something like **4M background and 4M
signal** is enough and trains in minutes to an hour. 



    denv python3 bdtMaker.py -s <sig_dir> -b <bkg_dir> \
      --max_evt <N> --bkg_max_evt <N> \
      --bkg_branch EcalVeto_<PASS> --sig_branch EcalVeto_<PASS> \
      --train_frac 0.9 --subsample 0.8 --colsample_bytree 0.75 \
      --depth 6 --min_child_weight 60 --gamma 0.5 --reg_lambda 1.5 \
      --eta 0.05 --tree_number 4000 --early_stopping_rounds 100 \
       -o <MODEL>

The hyperparameters above are a reasonable starting point.

Then, conver to `.onnx` 

    denv python3 pickle_to_onnx.py -i <MODEL>_0_weights.pkl -o <dir>

Copy the `.onnx` into `ldmx-sw/Ecal/data/`, point `vetos.py` at it, and `just compile`. You won't need to change the cut yet disc cut value yet, as this is for the final BDTs `pass` or `fail` cut.

---

# 2. Scoring — `scoring/`

The first scoring run is for plotting purposes. It will need to be done on roughly 1-2 Million samples to start. If the tail on the ROC log curve (around 1e-6) needs more statistics, a scoring on more samples may be needed.

A second scoring run will be needed for the cutflow. This will be done when certain models are chosen to be good after plotting, and want a cutflow done for futher analysis. This scoring run will be on the full data set (excluding training).

| file | use |
|---|---|
| `ecalveto_config.py` | Runs EcalVetoProcessor to add ONE BDT's `EcalVeto` branch to existing files. |
| `veto_mip_config.py` | EcalVeto + MIP |
| `smart_multifire.py` | Submits an entire directory to slurm using either config |

    denv fire ecalveto_config.py -i <in.root> -o <outdir> -m -1 -g <PASS> --bdt <MODEL>




Each pass **adds** a branch rather than replacing one, so scoring repeatedly chains
several models into the same files and lets you compare them event by event.

---

# 3. Plotting — `plotting/`

There are two ways to plot. You can either extract all the disc values from the files with the `extract_plottiing_npz.py`, and use the npz arrays for `roc_disc_ana.py` and `panel_ana.py`. 

Another way is to directly use  `plot_roc_curves.py` and `plot_disc_comparison.py`. These read directly from the root files, and 
The ROC work is cached in the metrics file, so re-panelling is instant.

---

# 4. Cutflow and operating point — `cutflow/`

This is the final evaluation of a bdt model, specifically using the largest sample set we have (still excluding training samples).
| file | use |
|---|---|
| `prerun_config.py` | Produces everything needed for the cutflow: Trigger, EcalVeto (incl. `fiducial_`), MIP tracking, Preselection, Tracker veto. |
| `extract_cutflow_npz.py` | ROOT -> `.npz` (slice + merge for big samples). |
| `fom_ana.py` | Punzi FOM; solves for the cut leaving exactly N background events. |
| `cutflow_table_ana.py` | Produces cutflow tables in csv. Can analyze multiple different models at once |


---

# 5. SHAP — `shap/`


    denv python3 shap_model.py --label <MODEL> \
      --model <MODEL>_0_weights.pkl \
      --sig-dir <sig_dir> --sig-branch EcalVeto_<PASS> \
      --bkg-dir <bkg_dir> --bkg-branch EcalVeto_<PASS>

Writes a bar plot, a beeswarm, a ranked CSV and the raw values. 
`--precuts` restricts to events passing Trigger + Fiducial + Preselection + Tracker
veto, i.e. what the BDT actually sees in the analysis. `--replot <npz>` redraws
without recomputing.

Use data the model did **not** train on, or the plot describes memorized events.
`README_shap_provenance.md` is where to record what each published plot used.

---
