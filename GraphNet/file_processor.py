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

"""
file_processor.py

Purpose:  Read through ROOT files containing LDMX events that ParrticleNet will be trained on, drop every event that
doesn't pass the ParticleNet preselection, and save the remainder to output ROOT files that ParticleNet can read.
This was introduced because both the preselection and the pT calculation involve loading information from the ROOT
files that ParticleNet itself doesn't need, and would substantially increase runtime if ParticleNet performed the
calculation for every event individually.

Outline:
- For every input file, do the following:
   - Read all necessary data from the file into arrays using uproot.
   - Drop all events that fail the preselection condition.
   - Compute the pT of each event from the TargetScoringPlaneHit information (needed for pT bias plots, and not
     present in ROOT files), and keep track of it alongside the other arrays/branches loaded for the file.
   - Use ROOT to create new output files and to fill them with the contents of the loaded arrays.

"""


# Standard preselection values (-> 95% sig/5% bkg)
MAX_NUM_ECAL_HITS = 110
MAX_ISO_ENERGY = 650

# Branches to save:
# Quantities labeled with 'scalars' have a single value per event.  Quantities labeled with 'vectors' have
# one value for every hit (e.g. number of ecal hits vs x position of each hit).
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

# Directory to write output files to:
output_dir = 'test_output_files'
# Locations of the ldmx-sw ROOT files to process+train on:
file_templates = {
    0.001: '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.001*.root',  # 0.001 GeV, etc.
    0.01:  '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.01*.root',
    0.1:   '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.1*.root',
    1.0:   '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*1.0*.root',
    # Note:  m=0 here refers to PN background events
    0:     '/home/pmasterson/GraphNet_input/v12/background_230_trunk/*.root'
}

"""
# Additional sample for evaluation:
output_dir = '/home/pmasterson/GraphNet_input/v12/processed_eval'
file_templates = {
    0.001: '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.001*.root',
    0.01:  '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.01*.root',
    0.1:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.1*.root',
    1.0:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*1.0*.root',
    0:     '/home/pmasterson/GraphNet_input/v12/bkg_12M/evaluation/*.root'
}
"""


def processFile(input_vars):
    # input_vars is a list:
    # [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0]  # Apparently this is the easiest approach to multiple args...
    mass = input_vars[1]
    filenum = input_vars[2]

    print("Processing file {}".format(filename))
    if mass == 0:
        outfile_name = "v12_pn_upkaon_{}.root".format(filenum)
    else:
        outfile_name = "v12_{}_upkaon_{}.root".format(mass, filenum)
    outfile_path = os.sep.join([output_dir, outfile_name])

    # NOTE:  Added this to ...
    if os.path.exists(outfile_path):
        print("FILE {} ALREADY EXISTS.  SKIPPING...".format(outfile_name))
        return 0, 0

    # Fix branch names:  uproot refers to EcalVeto branches with a / ('EcalVeto_v12/nReadoutHits_', etc), while
    # all other branches are referred to with a . ('EcalRecHits_v12.energy_', etc).  This is because ldmx-sw
    # writes EcalVeto information to the ROOT files in a somewhat unusual way; this may change in future updates
    # to ldmx-sw.
    branchList = []
    for branchname, leafdict in data_to_save.items():
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            # EcalVeto needs slightly different syntax:   . -> /
            if branchname == "EcalVeto_v12":
                branchList.append(branchname + '/' + leaf)
            else:
                branchList.append(branchname + '.' + leaf)


    # Open the file and read all necessary data from it:
    t = uproot.open(filename)['LDMX_Events']
    # (This part is just for printing the # of pre-preselection events:)
    tmp = t.arrays(['EcalVeto_v12/nReadoutHits_'])
    nTotalEvents = len(tmp)
    print("Before preselection:  found {} events".format(nTotalEvents))

    # t.arrays() returns a dict-like object:
    #    raw_data['EcalVeto_v12/nReadoutHits_'] == awkward array containing the value of 
    #    nReadoutHits_ for each event, and so on.
    raw_data = t.arrays(branchList) #, preselection)  #, aliases=alias_dict)

    # Perform the preselection:  Drop all events with more than MAX_NUM_ECAL_HITS in the ecal, 
    # and all events with an isolated energy that exceeds MAXX_ISO_ENERGY
    el = (raw_data['EcalVeto_v12/nReadoutHits_'] < MAX_NUM_ECAL_HITS) * (raw_data['EcalVeto_v12/summedTightIso_'] < MAX_ISO_ENERGY)
    preselected_data = {}
    for branch in branchList:
        preselected_data[branch] = raw_data[branch][el]
    #print("Preselected data")
    nEvents = len(preselected_data['EcalVeto_v12/summedTightIso_'])
    print("Skimming from {} events".format(nEvents))

    # Next, we have to compute TargetSPRecoilE_pt here instead of in train.py.  (This involves TargetScoringPlane
    # information that ParticleNet doesn't need, and that would take a long time to load with the lazy-loading
    # approach.)
    # For each event, find the recoil electron (maximal recoil pz):
    pdgID_ = t['TargetScoringPlaneHits_v12.pdgID_'].array()[el]
    z_     = t['TargetScoringPlaneHits_v12.z_'].array()[el]
    px_    = t['TargetScoringPlaneHits_v12.px_'].array()[el]
    py_    = t['TargetScoringPlaneHits_v12.py_'].array()[el]
    pz_    = t['TargetScoringPlaneHits_v12.pz_'].array()[el]
    tspRecoil = []
    for i in range(nEvents):
        max_pz = 0
        recoil_index = 0  # Store the index of the recoil electron
        for j in range(len(pdgID_[i])):
            # Constraint on z ensures that the SP downstream of the target is used
            if pdgID_[i][j] == 11 and z_[i][j] > 0.176 and z_[i][j] < 0.178 and pz_[i][j] > max_pz:
                max_pz = pz_[i][j]
                recoil_index = j
        # Calculate the recoil SP
        tspRecoil.append(np.sqrt(px_[i][recoil_index]**2 + py_[i][recoil_index]**2))
    # Put it in the preselected_data and treat it as an ordinary branch from here on out
    preselected_data['TargetSPRecoilE_pt'] = np.array(tspRecoil)

    # Additionally, add new branches storing the length for vector data (number of SP hits, number of ecal hits):
    nSPHits = []
    nRecHits = []
    x_data = preselected_data['EcalScoringPlaneHits_v12.x_']
    E_data = preselected_data['EcalRecHits_v12.energy_']
    for i in range(nEvents):
        # NOTE:  max num hits may exceed MAX_NUM...this is okay.
        nSPHits.append(len(x_data[i]))
        nRecHits.append(len(E_data[i]))
    preselected_data['nSPHits'] = np.array(nSPHits)
    preselected_data['nRecHits'] = np.array(nRecHits)


    # Prepare the output tree+file:
    outfile = r.TFile(outfile_path, "RECREATE")
    tree = r.TTree("skimmed_events", "skimmed ldmx event data")
    # Everything in EcalSPHits is a vector; everything in EcalVetoProcessor is a scalar

    # For each branch, create an array to temporarily hold the data for each event:
    scalar_holders = {}  # Hold ecalVeto (scalar) information
    vector_holders = {}
    for branch in branchList:
        leaf = re.split(r'[./]', branch)[1]  #Split at / or .
        # Find whether the branch stores scalar or vector data:
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
        assert(datatype == 'scalar' or datatype == 'vector')
        if datatype == 'scalar':  # If scalar, temp array has a length of 1
            scalar_holders[branch] = np.zeros((1), dtype='float32')
        else:  # If vector, temp array must have at least one element per hit
            # (liberally picked 2k)
            vector_holders[branch] = np.zeros((2000), dtype='float32')
    # Create new branches to store nSPHits, pT (necessary for tree creation)...
    scalar_holders['nSPHits'] = np.array([0], 'i')
    scalar_holders['nRecHits'] = np.array([0], 'i')
    scalar_holders['TargetSPRecoilE_pt'] = np.array([0], dtype='float32')
    branchList.append('nSPHits')
    branchList.append('nRecHits')
    branchList.append('TargetSPRecoilE_pt')
    # Now, go through each branch name and a corresponding branch to the tree:
    for branch, var in scalar_holders.items():
        # Need to make sure that each var is stored as the correct type (floats, ints, etc):
        if branch == 'nSPHits' or branch == 'nRecHits':
            branchname = branch
            dtype = 'I'
        elif branch == 'TargetSPRecoilE_pt':
            branchname = branch
            dtype = 'F'
        else:
            branchname = re.split(r'[./]', branch)[1]
            dtype = 'F'
        tree.Branch(branchname, var, branchname+"/"+dtype)
    for branch, var in vector_holders.items():
        # NOTE:  Can't currently handle EcalVeto branches that store vectors.  Not necessary for PN, though.
        parent = re.split(r'[./]', branch)[0]
        branchname = re.split(r'[./]', branch)[1]
        if parent == 'EcalScoringPlaneHits_v12':
            tree.Branch(branchname, var, "{}[nSPHits]/F".format(branchname))
        else:  # else in EcalRecHits
            tree.Branch(branchname+'rec_', var, "{}[nRecHits]/F".format(branchname+'rec_'))

    print("All branches added.  Filling...")

    for i in range(nEvents):
        # For each event, fill the temporary arrays with data, then write them to the tree with Fill()
        for branch in branchList:
            # Contains both vector and scalar data.  Treat them differently:
            if branch in scalar_holders.keys():  # Scalar
                # fill scalar data
                scalar_holders[branch][0] = preselected_data[branch][i]
            elif branch in vector_holders.keys():  # Vector
                # fill vector data
                for j in range(len(preselected_data[branch][i])):
                    vector_holders[branch][j] = preselected_data[branch][i][j]
            else:
                print("FATAL ERROR:  {} not found in *_holders".format(branch))
                assert(False)
        tree.Fill()

    # Finally, write the filled tree to the ouput file:
    outfile.Write()
    print("FINISHED.  File written to {}.".format(outfile_path))

    return (nTotalEvents, nEvents)


if __name__ == '__main__':
    # New approach:  Use multiprocessing
    #pool = Pool(16) -> Run 16 threads/process 16 files in parallel
    
    presel_eff = {}
    # For each signal mass and for PN background:
    for mass, filepath in file_templates.items():
        print("======  m={}  ======".format(mass))
        # Assemble list of function params
        # These get passed to processFile() when Pool requests them
        params = []
        for filenum, f in enumerate(glob.glob(filepath)):
            params.append([f, mass, filenum])  # list will be passed to ProcessFile:  processFile([filepath, mass, file_number])
        with Pool(16) as pool:  # Can increase this number if desired, although this depends on how many threads POD will let you run at once and on how much GPU is free.
            results = pool.map(processFile, params)
        print("Finished.  Result len:", len(results))
        print(results)
        nTotal  = sum([r[0] for r in results])
        nEvents = sum([r[1] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        presel_eff[int(mass * 1000)] = nEvents / nTotal
    print("Done.  Presel_eff: {}".format(presel_eff))

    # For running without multithreading (note:  will be extremely slow and is impractical unless you want to test/use 1-2 files at a time):
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
            nT, nE = processFile([f, mass, filenum])
            nTotal += nT
            nEvents += nE
            filenum += 1
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        presel_eff[int(mass * 1000)] = nEvents / nTotal

    print("DONE.  presel_eff: ", presel_eff)
    """



