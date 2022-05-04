import numpy as np
import uproot
import awkward
import glob
import math
from multiprocessing import Pool

file_templates = {
    0.001: '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*0.001*.root',
    0.01:  '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*0.01*.root',
    0.1:   '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*0.1*.root',
    1.0:   '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*1.0*.root',
    0:     '/home/aminali/production/rotation_prod/v300_tskim/bkg_output/*.root'
}

# preselection values
MAX_NUM_ECAL_HITS = 50 #60  #110 for ldmx-sw v2.3.0
MAX_ISO_ENERGY = 500  #650 for ldmx-sw v2.3.0
# Results:  >0.994 vs 0.055
MAX_NUM_HCAL_HITS = 30

# data[branch_name][scalars/vectors][leaf_name]
data_to_save = {
    'EcalScoringPlaneHits_v3_v13': {
        'scalars':[],
        'vectors':['pdgID_', 'trackID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    # NEW:  Added to correct photon trajectory calculation
    'TargetScoringPlaneHits_v3_v13': {
        'scalars':[],
        'vectors':['pdgID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    'EcalVeto_v3_v13': {
        'scalars':['passesVeto_', 'nReadoutHits_', 'summedDet_',
                   'summedTightIso_', 'discValue_',
                   'recoilX_', 'recoilY_',
                   'recoilPx_', 'recoilPy_', 'recoilPz_'],
        'vectors':[]
    },
    'EcalRecHits_v3_v13': {
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_']  # OLD: ['id_', 'energy_']
    },
    'HcalRecHits_v3_v13':{
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_']
    },
    'HcalVeto_v3_v13':{
        'scalars':['passesVeto_'],
        'vectors':[]
    }
}

def blname(branch, leaf):
    if branch.startswith('EcalVeto') or branch.startswith('HcalVeto'):
        return '{}/{}'.format(branch, leaf)
    else:
        return '{}/{}.{}'.format(branch, branch, leaf)

def processFile(input_vars):
    # input_vars is a list: [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0] 
    mass = input_vars[1]
    filenum = input_vars[2]

    branchList = []
    for branchname, leafdict in data.items():
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            if branchname == "EcalVeto_v3_v13" or branchname == "HcalVeto_v3_v13":
                branchList.append(branchname + '/' + leaf)
            else:
                branchList.append(branchname + '.' + leaf)

    # count total events
    file = uproot.open(filename)
    if len(file.keys()) == 0:
        print("FOUND ZOMBIE: {} SKIPPING...".format(filename))
        return 0, 0
    t = uproot.open(filename)['LDMX_Events']
    raw_data = t.arrays(branchList)
    nTotalEvents = len(raw_data[blname('EcalRecHits_v3_v13', 'xpos_')])

    # preselection 
    el = (raw_data[blname('EcalVeto_v3_v13', 'nReadoutHits_')] < MAX_NUM_ECAL_HITS) * (raw_data[blname('EcalVeto_v3_v13', 'summedTightIso_')] < MAX_ISO_ENERGY)

    preselected_data = {}
    for branch in branchList:
        preselected_data[branch] = raw_data[branch][el]
    nEvents = len(preselected_data[blname('EcalVeto_v3_v13', 'summedTightIso_')])

    # hcal veto 
    hc1 = preselected_data[blname('HcalVeto_v3_v13', 'passesVeto_')] == 1

    selected_data = {}
    for branch in branchList:
        selected_data[branch] = preselected_data[branch][hc1]
    nPassesVeto = len(selected_data[blname('HcalVeto_v3_v13', 'passesVeto_')])

    # hcal hits cut (boosted preselection)
    HE_data = preselected_data[blname('HcalRecHits_v3_v13', 'energy_')]
    nHRecHits = np.zeros(nEvents)
    for i in range(nEvents):
        nHRecHits[i] = sum(HE_data[i] > 0)
        if len(HE_data[i]) == 0:
            nHRecHits[i] = 0
    preselected_data['nHRecHits'] = np.array(nHRecHits)

    hc2 = preselected_data['nHRecHits'] < MAX_NUM_ECAL_HITS

    preselected_data_2 = {}
    for branch in branchList:
        preselected_data_2[branch] = preselected_data[branch][hc2]
    nEvents2 = len(preselected_data_2['nHRecHits'])


    return (nTotalEvents, nEvents, nPassesVeto, nEvents2)

if __name__ == '__main__':
    presel_eff = {}
    hcalveto_eff = {}
    presel_eff_2 = {}
    print("Enter detector version:")
    version = input()
    _load_cellMap(version)
    for mass, filepath in file_templates.items():
        print("======  m={}  ======".format(mass))
        params = []
        for filenum, f in enumerate(glob.glob(filepath)):
            params.append([f, mass, filenum])  # list will be passed to ProcessFile:  processFile([filepath, mass, file_number])
        with Pool(20) as pool:
            results = pool.map(processFile, params)
        nTotal  = sum([r[0] for r in results])
        nEvents = sum([r[1] for r in results])
        nPassesVeto = sum([r[2] for r in results])
        nEvents2 = sum([r[3] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection, {} passed preselection + hcal veto, {} passed double preselection".format(int(mass*1000), nTotal, nEvents, nPassesVeto, nEvents2))
        presel_eff[int(mass * 1000)] = float(nEvents) / nTotal if nTotal != 0 else 'no events'
        hcalveto_eff[int(mass * 1000)] = float(nPassesVeto) / nEvents if nEvents != 0 else 'no events'
        presel_eff_2[int(mass * 1000)] = float(nEvents2) / nTotal if nTotal != 0 else 'no events'
    print("Done.  Preselection efficiency: {}, Hcal Veto efficiency: {}, Boosted preselection efficiency: {}".format(presel_eff, hcalveto_eff, presel_eff_2))