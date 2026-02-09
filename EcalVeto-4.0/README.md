# EcalVeto-3.0
In order to achieve better time-efficiency, we develop a new workflow to replace the Python-based processor in [`pyEcalveto`](https://github.com/IncandelaLab/LDMX-scripts/tree/master/pyEcalVeto). This new version of EcalVeto makes use of the `EcalVetoProcessor` in `ldmx-sw` to process the sample, which will significantly reduce the processing time.

## Prerequisites
- Install the latest `ldmx-sw` on your machine. See the document [here](https://ldmx-software.github.io/building/intro.html).
- Install necessary python packages for BDT training.
- In order to convert the BDT model in pickle file to onnx file, one must upgrade the `onnxmltools` and `onnxconverter-common` to the `HEAD` on github:
    - `pip install git+https://github.com/microsoft/onnxconverter-common`
    - `pip install git+https://github.com/onnx/onnxmltools`

## EcalVeto Branch Re-reco
The config file `ecalveto_reco.py` re-processes existing LDMX samples, i.e., it will create a new branch `EcalVeto_SegmipBDTReco` (the passname `SegmipBDTReco` can be changed [here](https://github.com/danyi211/LDMX-scripts/blob/master/EcalVeto-3.0/ecalveto_reco.py#L13)) that contains the new Seg-MIP variables for BDT training. It can be used to both calculate the BDT variables and evaluate a new BDT model. The latter task will require one to create a new BDT onnx file in [`Ecal/data`](https://github.com/LDMX-Software/ldmx-sw/tree/trunk/Ecal/data) and change the corresponding BDT path in the [python config file](https://github.com/LDMX-Software/ldmx-sw/blob/trunk/Ecal/python/vetos.py). Run `ldmx python3 ecalveto_reco.py -h` to get the help meassages for the arguments.

The script `bdtMaker.py` reads in LDMX event files and train the Seg-MIP BDT. Please make sure the learning objective of `xgboost.train` is `binary:logistic` (currently `multi:softmax` is incompatible with the onnx converter). The output will be a pickle file containing the trained BDT model.

The script `pickle_to_onnx.py` converts the BDT model in pickle file to onnx file. Please make sure the `onnxmltools` and `onnxconverter-common` are updated to the `HEAD` on github. (See [Prerequisites](https://github.com/danyi211/LDMX-scripts/tree/master/EcalVeto-3.0#prerequisites))

