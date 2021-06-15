import numpy as np
import uproot
import awkward
import glob
import os
import re
print("Importing ROOT")
import ROOT as r
print("Imported root.  Starting...")
from multiprocessing import Pool

# NEW:  Introduced in tandem w/ lazy loading to lower increasing GPU usage
# NOTE:  Must run this with a working ldmx-sw installation
# Goal:
# - Loop through each input root file (no difference in treatment except filename)
# - Perform preselection
# - Compute pT from TargetScoringPlaneHits
# - If event passes presel, keep it; otherwise, drop it
# - Write necessary branches ONLY to new output root trees

# Standard preselection values (-> 95% sig/5% bkg)
MAX_NUM_ECAL_HITS = 110
MAX_ISO_ENERGY = 650

# Branches to save:
# (Everything else can be safely ignored)
# data[branch_name][scalars/vectors][leaf_name]
data_to_save = {
    'EcalScoringPlaneHits_v12': {
        'scalars':[],
        'vectors':['pdgID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    'EcalVeto_v12': {
        'scalars':['passesVeto_', 'nReadoutHits_', 'summedDet_',
                   'summedTightIso_', 'discValue_',
                   'recoilX_', 'recoilY_',
                   'recoilPx_', 'recoilPy_', 'recoilPz_'],
        'vectors':[]
    },
    'EcalRecHits_v12': {
        'scalars':[],
        'vectors':['id_', 'energy_']
    }
}

# Base:

#output_dir = '/home/pmasterson/GraphNet_input/v12/processed'
output_dir = 'test_output_files'
file_templates = {
    #0.001: '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.001*.root',
    #0.01:  '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.01*.root',
    #0.1:   '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.1*.root',
    #1.0:   '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*1.0*.root',  # was sig_extended_tracking
    0:     '/home/pmasterson/GraphNet_input/v12/background_230_trunk/*.root'  # was bkg_12M
}
"""
# For eval:
output_dir = '/home/pmasterson/GraphNet_input/v12/processed_eval'
file_templates = {
    #0.001: '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.001*.root',
    #0.01:  '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.01*.root',
    #0.1:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.1*.root',
    #1.0:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*1.0*.root',
    0:     '/home/pmasterson/GraphNet_input/v12/bkg_12M/evaluation/*.root'
}
"""

def processFile(input_vars):

    filename = input_vars[0]  # Apparently this is the easiest approach to multiple args...
    mass = input_vars[1]
    filenum = input_vars[2]

    print("Processing file {}".format(filename))
    if mass == 0:
        outfile_name = "v12_pn_mipless_{}.root".format(filenum)
    else:
        outfile_name = "v12_{}_mipless_{}.root".format(mass, filenum)
    outfile_path = os.sep.join([output_dir, outfile_name])

    # NOTE:  Added this to ...
    if os.path.exists(outfile_path):
        print("FILE {} ALREADY EXISTS.  SKIPPING...".format(outfile_name))
        return 0, 0

    branchList = []
    for branchname, leafdict in data_to_save.items():
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            # EcalVeto needs slightly different syntax:   . -> /
            if branchname == "EcalVeto_v12":
                branchList.append(branchname + '/' + leaf)
            else:
                branchList.append(branchname + '.' + leaf)
    # Load only events passing the preselection
    #print("branchList is:")
    #print(branchList)
    #alias_dict = {'EcalVeto_v12/nReadoutHits_': 'nHits',
    #              'EcalVeto_v12/summedTightIso_': 'isoEnergy'}
    #preselection = '(EcalVeto_v12/nReadoutHits_ < {})'.format(MAX_NUM_ECAL_HITS) # & (EcalVeto_v12/summedTightIso_ < {})'.format(MAX_NUM_ECAL_HITS, MAX_ISO_ENERGY)

    t = uproot.open(filename)['LDMX_Events']
    # This is just for printing the # of pre-preselection events
    tmp = t.arrays(['EcalVeto_v12/nReadoutHits_'])
    nTotalEvents = len(tmp)
    print("Before preselection:  found {} events".format(nTotalEvents))

    raw_data = t.arrays(branchList) #, preselection)  #, aliases=alias_dict)
    el = (raw_data['EcalVeto_v12/nReadoutHits_'] < MAX_NUM_ECAL_HITS) * (raw_data['EcalVeto_v12/summedTightIso_'] < MAX_ISO_ENERGY)
    preselected_data = {}
    for branch in branchList:
        preselected_data[branch] = raw_data[branch][el]
    #print("Preselected data")
    nEvents = len(preselected_data['EcalVeto_v12/summedTightIso_'])
    print("Skimming from {} events".format(nEvents))

    # ADDITIONALLY:  Have to compute TargetSPRecoilE_pt here instead of in train.py.  (No access to TSP data)
    # For each event, find the recoil electron:
    pdgID_ = t['TargetScoringPlaneHits_v12.pdgID_'].array()[el]
    z_     = t['TargetScoringPlaneHits_v12.z_'].array()[el]
    px_    = t['TargetScoringPlaneHits_v12.px_'].array()[el]
    py_    = t['TargetScoringPlaneHits_v12.py_'].array()[el]
    pz_    = t['TargetScoringPlaneHits_v12.pz_'].array()[el]
    tspRecoil = []
    # Currently trying the slow approach...
    for i in range(nEvents):
        #if i % 1000 == 0:  print("pT, evt {}".format(i))
        max_pz = 0
        recoil_index = 0  # Find the recoil electron
        for j in range(len(pdgID_[i])):
            if pdgID_[i][j] == 11 and z_[i][j] > 0.176 and z_[i][j] < 0.178 and pz_[i][j] > max_pz:
                max_pz = pz_[i][j]
                recoil_index = j
        # Calculate the recoil SP
        tspRecoil.append(np.sqrt(px_[i][recoil_index]**2 + py_[i][recoil_index]**2))
    #print("SP recoil pT, sample values:", tspRecoil[:5], ". total values:", len(tspRecoil))
    preselected_data['TargetSPRecoilE_pt'] = np.array(tspRecoil)

    # Now take the loaded + preselected data, iterate through each event, and write them to a new root file:
    outfile = r.TFile(outfile_path, "RECREATE")
    tree = r.TTree("skimmed_events", "skimmed ldmx event data")
    # Everything in EcalSPHits is a vector; everything in EcalVetoProcessor is a scalar


    # Additionally, add a new branch storing the length for vector data (nSPHits)
    nSPHits = []
    x_data = preselected_data['EcalScoringPlaneHits_v12.x_']
    for i in range(nEvents):
        # NOTE:  max num hits may exceed MAX_NUM...this is okay.
        nSPHits.append(len(x_data[i]))
    # This assignment may fail...
    preselected_data['nSPHits'] = np.array(nSPHits)
    #print("nSPHits created, sample:", x_data[:10])
    nRecHits = []
    E_data = preselected_data['EcalRecHits_v12.energy_']
    for i in range(nEvents):
        #print("appending len", len(E_data[i]))
        nRecHits.append(len(E_data[i]))
    preselected_data['nRecHits'] = np.array(nRecHits)


    #print("Preparing vars to hold leaf data")
    # For each branch, prepare a var to hold the corresp information
    scalar_holders = {}  # Hold ecalVeto (scalar) information
    vector_holders = {}
    for branch in branchList:
        # Search for item in data_to_save
        # If found, add branch: scalar or branch: vector to dict
        #print(branch)
        #print(re.split(r'[./]', branch))
        leaf = re.split(r'[./]', branch)[1]  #Split at / or .
        datatype = None
        for br, brdict in data_to_save.items():
            #print(leaf)
            #print(brdict['scalars'], brdict['vectors'])
            if leaf in brdict['scalars']:
                datatype = 'scalar'
                continue
            elif leaf in brdict['vectors']:
                datatype = 'vector'
                continue
        if datatype == 'scalar':
            scalar_holders[branch] = np.zeros((1), dtype='float32')
        elif datatype == 'vector':
            # NOTE: EcalSPHits may contain MORE than MAX_NUM... hits.
            vector_holders[branch] = np.zeros((2000), dtype='float32')
        else:  print("ERROR:  datatype is neither a scalar nor a vector!")
    #scalar_holders['nSPHits'] = np.zeros((1), 'i')
    #scalar_holders['TargetSPRecoilE_pt'] = np.zeros((1), 'i')
    #scalar_holders['nRecHits'] = np.zeros((1), 'i')

    #print("Scalar, vector_holders:")
    #print(scalar_holders)
    #print(vector_holders)

    """
    # TESTING:  This code works
    tree.Branch('nSPHits', scalar_holders['nSPHits'], 'nSPHits/I')
    tree.Branch('x_', vector_holders['EcalScoringPlaneHits_v12.x_'], 'x_[nSPHits]/F')

    scalar_holders['nSPHits'][0] = 3
    vector_holders['EcalScoringPlaneHits_v12.x_'][0] = 1
    vector_holders['EcalScoringPlaneHits_v12.x_'][1] = 2
    vector_holders['EcalScoringPlaneHits_v12.x_'][2] = 4
    tree.Fill()
    scalar_holders['nSPHits'][0] = 2
    vector_holders['EcalScoringPlaneHits_v12.x_'][0] = 6
    vector_holders['EcalScoringPlaneHits_v12.x_'][1] = 7
    tree.Fill()
    """


    # Create new branches to store nSPHits, pT (necessary for tree creation)...
    scalar_holders['nSPHits'] = np.array([0], 'i')
    scalar_holders['TargetSPRecoilE_pt'] = np.array([0], dtype='float32')
    scalar_holders['nRecHits'] = np.array([0], 'i')
    branchList.append('nSPHits')
    branchList.append('nRecHits')
    branchList.append('TargetSPRecoilE_pt')
    # Create a new branch for each event
    # For all other branches:
    for branch, var in scalar_holders.items():
        #print("ADDING BRANCH", branch)
        if branch == 'nSPHits' or branch == 'nRecHits':
            branchname = branch
            dtype = 'I'
        elif branch == 'TargetSPRecoilE_pt':
            branchname = branch
            dtype = 'F'
        else:
            branchname = re.split(r'[./]', branch)[1]
            dtype = 'F'
        #print("Adding scalar branch:", branchname, branchname+'/'+dtype)
        tree.Branch(branchname, var, branchname+"/"+dtype)
    for branch, var in vector_holders.items():
        # NOTE:  Can't currently handle EcalVeto branches that store vectors
        parent = re.split(r'[./]', branch)[0]
        branchname = re.split(r'[./]', branch)[1]
        #print("Adding vector branch:", branchname)
        #print(branchname)
        #print(data_to_save['EcalScoringPlaneHits_v12']['vectors'])
        #print(branch)
        if parent == 'EcalScoringPlaneHits_v12':
            tree.Branch(branchname, var, "{}[nSPHits]/F".format(branchname))
            #print("Creating branch", branchname)
        else:  # else in EcalRecHits
            #print("rec parent is", parent)
            tree.Branch(branchname+'rec_', var, "{}[nRecHits]/F".format(branchname+'rec_'))
            #print("Creating branch", branchname+'rec_')

    print("All branches added.  Filling...")

    for i in range(nEvents):
        #if i % 1000 == 0:  print("  Filling event {}".format(i))
        for branch in branchList:
            # Contains both vector and scalar data.  Treat them differently:
            #print("  branch:", branch)
            #print("  ", preselected_data[branch][i])
            
            if branch in scalar_holders.keys():  # Scalar
                # fill scalar data
                scalar_holders[branch][0] = preselected_data[branch][i]
            elif branch in vector_holders.keys():  # Vector
                # fill vector data
                #print("Confirm equal lengths:  data={}, nHits={}".format(len(preselected_data[branch][i]), preselected_data['nSPHits'][i]))
                for j in range(len(preselected_data[branch][i])):
                    vector_holders[branch][j] = preselected_data[branch][i][j]
            else:
                print("ERROR:  {} not found in _holders".format(branch))
        tree.Fill()


    outfile.Write()
    print("FINISHED.  File written to {}.".format(outfile_path))
    return (nTotalEvents, nEvents)


if __name__ == '__main__':
    # New approach:  Use multiprocessing
    #pool = Pool(8)  # 8 processors, factor of 8 speedup in theory...
    presel_eff = {}
    for mass, filepath in file_templates.items():
        print("======  m={}  ======".format(mass))
        # Assemble list of function params
        params = []
        for filenum, f in enumerate(glob.glob(filepath)[:4]):
            params.append([f, mass, filenum])
        print("num params:", len(params))
        with Pool(2) as pool:
            results = pool.map(processFile, params)
        print("Finished.  Result len:", len(results))
        print(results)
        nTotal  = sum([r[0] for r in results])
        nEvents = sum([r[1] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        presel_eff[int(mass * 1000)] = nEvents / nTotal
    print("Done.  Presel_eff:")
    print(presel_eff)

    """
    presel_eff = {}
    for mass, filepath in file_templates.items():
        #if mass != 0:  continue
        filenum = 0
        nTotal = 0  # pre-preselection
        nEvents = 0 # post-preselection
        print("======  m={}  ======".format(mass))
        for f in glob.glob(filepath):
            # Process each file separately
            nT, nE = processFile(f, mass, filenum)
            nTotal += nT
            nEvents += nE
            filenum += 1
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        presel_eff[int(mass * 1000)] = nEvents / nTotal

    print("DONE.  presel_eff: ", presel_eff)
    """



