import uproot
import glob
from tqdm import tqdm
import json
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(20)

# file dictionary {mass: filepath}
file_templates = {
    0.001: '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*0.001*.root',
    0.01:  '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*0.01*.root',
    0.1:   '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*0.1*.root',
    1.0:   '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*1GeV*.root',
    0:     '/home/aminali/production/v14_prod/v3.2.0_ecalPN_tskim_sizeskim/*.root'
}

# dictionaires for total events, selected events, and preselection efficiency
# for each mass point (4 sig + PN bkg)
nTotal = {}
nPass = {}
effs = {}

# for each mass point
for mass in file_templates.keys():
    
    print(f"==== m = {mass} ====", flush=True)
    
    # different branch name syntax for signal vs. bkg
    if mass: # signal
        branchList = ['EcalVeto_signal/nReadoutHits_', 'EcalVeto_signal/summedTightIso_'] 
    else: # PN bkg
        branchList = ['EcalVeto_sim/nReadoutHits_', 'EcalVeto_sim/summedTightIso_']
    
    # preselection parameters
    MAX_NUM_ECAL_HITS = 50
    MAX_ISO_ENERGY = 700
    
    nTotal[mass] = 0 # total event count
    nPass[mass] = 0 # selected event count
    
    file_list = glob.glob(file_templates[mass])
    nFiles = len(file_list)
    
    # for each file (with progress bar)
    for i, filename in tqdm(enumerate(file_list), total=nFiles):
        with uproot.open(filename) as file:
            if not file.keys(): # if no keys in file
                print(f"FOUND ZOMBIE: {filename}  SKIPPING...", flush=True)
                continue
        with uproot.open(filename)['LDMX_Events'] as t:
            if not t.keys(): # if no keys in 'LDMX_Events'
                print(f"FOUND ZOMBIE: {filename}  SKIPPING...", flush=True)
                continue
            data = t.arrays(branchList, interpretation_executor=executor)
            
            nReadoutHits = data[branchList[0]] # array with number of ecal hits for each event
            IsoEnergy = data[branchList[1]] # array with iso energy for each event
                
            # preselection
            presel = (nReadoutHits < MAX_NUM_ECAL_HITS) * (IsoEnergy < MAX_ISO_ENERGY)
                
            # update event counts
            nTotal[mass] += len(nReadoutHits)
            nPass[mass] += len(nReadoutHits[presel])
            
        # stop after i + 1 files (optional)
        #if i == 100:
            #break
    
    # update preselection efficiencies dictionary
    effs[mass] = float(nPass[mass]) / nTotal[mass] if nTotal[mass] else 0
    
    print(f"\nRead {nTotal[mass]} events ... Passed {nPass[mass]} events\n", flush=True)
    
json_effs = json.dumps(effs, indent=4)
print(f"Preselection efficiencies:\n{json_effs}", flush=True)

print("\nPreselection parameters:\n")
print(f"Maximum number of ecal hits (nReadoutHits): {MAX_NUM_ECAL_HITS}")
print(f"Maximum isolated energy (summedTightIso): {MAX_ISO_ENERGY}")

# output ...
# v14 preselection efficiencies for MAX_NUM_ECAL_HITS = 50 and MAX_ISO_ENERGY = 700
# effs = {0.001: 0.9815366636237547, 0.01: 0.990946862016182, 0.1: 0.9926712171901053, 1.0: 0.9888390477198093, 0: 0.06929824265618538}