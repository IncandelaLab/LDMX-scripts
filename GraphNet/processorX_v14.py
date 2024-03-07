#!/home/pmasterson/miniconda3/envs/torchroot/bin/python

#SBATCH -n 20
#SBATCH --output=slurm_file_processor.out

# NOTE:  was --nodes=1, --ntasks-per-node 2

import numpy as np
import uproot
import awkward
import glob
import os
import sys
import re
import math
print("Importing ROOT")
import ROOT as r
print("Imported ROOT.  Starting...")
# this will allow you to control how Pool creates processes (spawn instead of fork) to avoid deadlock!
# nice blog post about this here: https://pythonspeed.com/articles/python-multiprocessing/
from multiprocessing import get_context # WAS: from multiprocessing import Pool

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

# Directory to write output files to:
output_dir = '/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total'
# Locations of the 2.3.0 ldmx-sw ROOT files to process+train on:
"""
file_templates = {
    0.001: '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.001*.root',
    0.01:  '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.01*.root',
    0.1:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.1*.root',
    1.0:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*1.0*.root',
    0:     '/home/dgj1118/LDMX-scripts/GraphNet/background_230_trunk/*.root'
}
"""
"""
file_templates = {
    0.001: '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.001*.root',  # 0.001 GeV, etc.
    0.01:  '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.01*.root',
    0.1:   '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*0.1*.root',
    1.0:   '/home/pmasterson/GraphNet_input/v12/signal_230_trunk/*1.0*.root',
    # Note:  m=0 here refers to PN background events
    0:     '/home/pmasterson/GraphNet_input/v12/background_230_trunk/*.root'
}
"""

# 3.0.0:
"""
file_templates = {
    0.001: '/home/pmasterson/events/v3.0.0_trigger/signal/*0.001*.root',  # 0.001 GeV, etc.
    0.01:  '/home/pmasterson/events/v3.0.0_trigger/signal/*0.01*.root',
    0.1:   '/home/pmasterson/events/v3.0.0_trigger/signal/*0.1*.root',
    1.0:   '/home/pmasterson/events/v3.0.0_trigger/signal/*1.0*.root',
    # Note:  m=0 here refers to PN background events
    0:     '/home/pmasterson/events/v3.0.0_trigger/background/*.root'
}
"""
"""
file_templates = {
    0.001: '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*0.001*.root',  # 0.001 GeV, etc.
    0.01:  '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*0.01*.root',
    0.1:   '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*0.1*.root',
    1.0:   '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*1.0*.root',
    # Note:  m=0 here refers to PN background events
    0:     '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/photonuclear/*.root'
}
"""
'''
# v13 geometry:
file_templates = {
    0.001: '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*0.001*.root',
    0.01:  '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*0.01*.root',
    0.1:   '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*0.1*.root',
    1.0:   '/home/aminali/production/rotation_prod/v300_tskim/sig_output/*1.0*.root',
    0:     '/home/aminali/production/rotation_prod/v300_tskim/bkg_output/*.root'
}
'''
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
# v14 geometry
'''
file_templates = {
    0.001: '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*0.001*.root',
    0.01:  '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*0.01*.root',
    0.1:   '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*0.1*.root',
    1.0:   '/home/aminali/production/v14_prod/Ap_sizeskim_1e_v3.2.1_v14_tskim/*Ap1GeV*.root',
    0:     '/home/aminali/production/v14_prod/v3.2.0_ecalPN_tskim_sizeskim/*.root'
}
'''
'''
file_templates = {
    0.001: '/home/aminali/production/v14_prod/Ap0.001GeV_1e_v3.2.2_v14_tskim/*.root',
    0.01:  '/home/aminali/production/v14_prod/Ap0.01GeV_1e_v3.2.2_v14_tskim/*.root',
    0.1:   '/home/aminali/production/v14_prod/Ap0.1GeV_1e_v3.2.2_v14_tskim/*.root',
    1.0:   '/home/aminali/production/v14_prod/Ap1GeV_1e_v3.2.3_v14_tskim/*.root',
    0:     '/home/aminali/production/v14_prod/v3.2.0_ecalPN_tskim_sizeskim/*.root'
}
'''
file_templates = {
    0.001:  '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root',
    0.01:  '/home/vamitamas/Samples8GeV/Ap0.01GeV_sim/*.root',
    0.1:   '/home/vamitamas/Samples8GeV/Ap0.1GeV_sim/*.root',
    1.0:   '/home/vamitamas/Samples8GeV/Ap1GeV_sim/*.root',
    0:     '/home/vamitamas/Samples8GeV/v3.3.3_ecalPN*/*.root'
}

# Standard preselection values (-> 95% sig/5% bkg)
MAX_NUM_ECAL_HITS = 90 #50 #60  #110  #Now MUCH lower!  >99% of 1 MeV sig should pass this. (and >10% of bkg)
MAX_ISO_ENERGY = 1100 #700 #500  # NOTE:  650 passes 99.99% sig, ~13% bkg for 3.0.0!  Lowering...
# Results:  >0.994 vs 0.055
# UPDATED FOR v14 ... Results: ~0.98-0.99 vs 0.069
#MAX_NUM_HCAL_HITS = 30

# Branches to save:
# Quantities labeled with 'scalars' have a single value per event.  Quantities labeled with 'vectors' have
# one value for every hit (e.g. number of ecal hits vs x position of each hit).
# (Everything else can be safely ignored)
# data[branch_name][scalars/vectors][leaf_name]
data_to_save = {
    'EcalScoringPlaneHits': {
        'scalars':[],
        'vectors':['pdgID_', 'trackID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    # NEW:  Added to correct photon trajectory calculation
    'TargetScoringPlaneHits': {
        'scalars':[],
        'vectors':['pdgID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    'EcalVeto': {
        'scalars':['passesVeto_', 'nReadoutHits_', 'summedDet_',
                   'summedTightIso_', 'discValue_',
                   'recoilX_', 'recoilY_',
                   'recoilPx_', 'recoilPy_', 'recoilPz_'],
        'vectors':[]
    },
    'EcalRecHits': {
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_']  # OLD: ['id_', 'energy_']
    },
    'HcalRecHits':{
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_', 'id_', 'pe_']
    }
}

def blname(branch, leaf, sig):
    if sig:
        if branch.startswith('EcalVeto') or branch.startswith('Trigger'):
            return '{}/{}'.format(f'{branch}_signal', leaf)
        else:
            return '{}/{}.{}'.format(f'{branch}_signal', f'{branch}_signal', leaf)

    else: # bkg (different syntax)
        if branch.startswith('EcalVeto'):
            return '{}/{}'.format(f'{branch}_sim', leaf)
        else:
            return '{}/{}.{}'.format(f'{branch}_sim', f'{branch}_sim', leaf)

def processFile(input_vars):
    # input_vars is a list:
    # [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0]  # Apparently this is the easiest approach to multiple args...
    mass = input_vars[1]
    filenum = input_vars[2]

    sig = True
    if not mass:
        sig = False

    print("Processing file {}".format(filename))
    if not mass:
        outfile_name = "v14_8gev_pn_XCal_total_{}.root".format(filenum)
    else:
        outfile_name = "v14_8gev_{}_XCal_total_{}.root".format(mass, filenum)
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
        if mass:
            branchname_ = f'{branchname}_signal'
        else:
            branchname_ = f'{branchname}_sim'
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            # EcalVeto needs slightly different syntax:   . -> /
            if branchname == "EcalVeto":
                branchList.append(branchname_ + '/' + leaf)
            else:
                branchList.append(branchname_ + '/' + branchname_ + '.' + leaf)
    if sig:
        branchList.append(blname('TriggerSums20Layers', 'pass_', sig))

    #print("Branches to load:")
    #print(branchList)

    # Open the file and read all necessary data from it:
    with uproot.open(filename) as file:
        if not file.keys():
            print("FOUND ZOMBIE: {} SKIPPING...".format(filename))
            return 0, 0
    with uproot.open(filename)['LDMX_Events'] as t:
        if not t.keys():
            print("FOUND ZOMBIE: {} SKIPPING...".format(filename))
            return 0, 0
        key_miss = False
        for branch in branchList:
            if not re.split('/', branch)[0] in t.keys():
                key_miss = True
                break
        if key_miss:
            print(f"MISSING KEYS IN: {filename}  SKIPPING...")
            return 0,0

        # (This part is just for printing the # of pre-preselection events:)
        # Must trigger skim first (if signal)
        if sig:
            raw_data = t.arrays(branchList)
            trig_pass = raw_data[blname('TriggerSums20Layers', 'pass_', sig)]
            tskimmed_data = {}
            for branch in branchList:
                tskimmed_data[branch] = raw_data[branch][trig_pass]
        else: # bkg, already trigger skimmed
            #tmp = t.arrays(['EcalVeto_v12/nReadoutHits_'])
            #nTotalEvents = len(tmp)
            tskimmed_data = t.arrays(branchList) #, preselection)  #, aliases=alias_dict)
        
        #print("Check tskimmed_data:")
        #print(tskimmed_data[blname('EcalScoringPlaneHits_v3_v13','pdgID_')])
        nTotalEvents = len(tskimmed_data[blname('EcalRecHits', 'xpos_', sig)])
        if nTotalEvents == 0:
            print("FILE {} CONTAINS ZERO EVENTS. SKIPPING...".format(filename))
            return 0, 0

        print("Before preselection: found {} events".format(nTotalEvents))

        # t.arrays() returns a dict-like object:
        #    raw_data['EcalVeto_v12/nReadoutHits_'] == awkward array containing the value of 
        #    nReadoutHits_ for each event, and so on.
        #raw_data = t.arrays(branchList) #, preselection)  #, aliases=alias_dict)

        # Perform the preselection:  Drop all events with more than MAX_NUM_ECAL_HITS in the ecal, 
        # and all events with an isolated energy that exceeds MAXX_ISO_ENERGY
        el = (tskimmed_data[blname('EcalVeto', 'nReadoutHits_', sig)] < MAX_NUM_ECAL_HITS) * (tskimmed_data[blname('EcalVeto', 'summedTightIso_', sig)] < MAX_ISO_ENERGY) 

        preselected_data = {}
        for branch in branchList:
            preselected_data[branch] = tskimmed_data[branch][el]
        #print("Preselected data")
        nEvents = len(preselected_data[blname('EcalVeto', 'summedTightIso_', sig)])
        print("After preselection: skimming from {} events".format(nEvents))

        # Next, we have to compute TargetSPRecoilE_pt here instead of in train.py.  (This involves TargetScoringPlane
        # information that ParticleNet doesn't need, and that would take a long time to load with the lazy-loading
        # approach.)
        # For each event, find the recoil electron (maximal recoil pz):
        # pdgID_ = t[blname('TargetScoringPlaneHits', 'pdgID_', sig)].array()[el] (this type of array used before, no trig skim)
        pdgID_ = preselected_data[blname('TargetScoringPlaneHits', 'pdgID_', sig)]
        z_     = preselected_data[blname('TargetScoringPlaneHits', 'z_', sig)]
        px_    = preselected_data[blname('TargetScoringPlaneHits', 'px_', sig)]
        py_    = preselected_data[blname('TargetScoringPlaneHits', 'py_', sig)]
        pz_    = preselected_data[blname('TargetScoringPlaneHits', 'pz_', sig)]
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
            if max_pz > 0: #if len(px_[i]) > 0:   # if max_pz > 0:
                tspRecoil.append(np.sqrt(px_[i][recoil_index]**2 + py_[i][recoil_index]**2))
            else:
                tspRecoil.append(-999)
        # Put it in the preselected_data and treat it as an ordinary branch from here on out
        preselected_data['TargetSPRecoilE_pt'] = np.array(tspRecoil)

        # Additionally, add new branches storing the length for vector data (number of SP hits, number of ecal hits):
        nSPHits = np.zeros(nEvents) # was: []
        nTSPHits = np.zeros(nEvents)
        nRecHits = np.zeros(nEvents) # was: []
        nHRecHits = np.zeros(nEvents)
        maxPE = np.zeros(nEvents)
        x_data = preselected_data[blname('EcalScoringPlaneHits','x_', sig)]
        xsp_data = preselected_data[blname('TargetScoringPlaneHits','x_', sig)]
        E_data = preselected_data[blname('EcalRecHits','energy_', sig)]
        HE_data = preselected_data[blname('HcalRecHits', 'energy_', sig)]
        pe_data = preselected_data[blname('HcalRecHits', 'pe_', sig)]
        for i in range(nEvents):
            # NOTE:  max num hits may exceed MAX_NUM...this is okay.
            nSPHits[i] = len(x_data[i])      # was: nSPHits.append(len(x_data[i])) 
            nTSPHits[i] = len(xsp_data[i]) 
            nRecHits[i] = sum(E_data[i] > 0) # was: nRecHits.append(len(E_data[i]))
            if len(E_data[i]) == 0:
                nRecHits[i] = 0
            nHRecHits[i] = sum(HE_data[i] > 0)
            if len(HE_data[i]) == 0:
                nHRecHits[i] = 0
            if len(pe_data[i]) != 0:
                maxPE[i] = max(pe_data[i])
            if len(pe_data[i]) == 0:
                maxPE[i] = 0
        preselected_data['nSPHits'] = np.array(nSPHits)
        preselected_data['nTSPHits'] = np.array(nTSPHits)
        preselected_data['nRecHits'] = np.array(nRecHits)
        preselected_data['nHRecHits'] = np.array(nHRecHits)
        preselected_data['maxPE'] = np.array(maxPE)

        '''
        # Apply cut on nHRecHits for improved background rejection 

        hc = preselected_data['nHRecHits'] < MAX_NUM_HCAL_HITS

        for branch in branchList:
            preselected_data[branch] = preselected_data[branch][hc]
        nEvents = len(preselected_data['nRecHits'])
        print("After preselection: skimming from {} events".format(nEvents))
        '''

        # Prepare the output tree+file:
        outfile = r.TFile(outfile_path, "RECREATE")
        tree = r.TTree("skimmed_events", "skimmed ldmx event data")
        # Everything in EcalSPHits is a vector; everything in EcalVetoProcessor is a scalar

        # For each branch, create an array to temporarily hold the data for each event:
        scalar_holders = {}  # Hold ecalVeto (scalar) information
        vector_holders = {}
        for branch in branchList:
            leaf = re.split(r'[./]', branch)[-1]  #Split at / or .
            # Find whether the branch stores scalar or vector data:
            datatype = None
            for br, brdict in data_to_save.items():
                #print(leaf)
                #print(brdict['scalars'], brdict['vectors'])
                if leaf in brdict['scalars'] or leaf:
                    datatype = 'scalar'
                    continue
                elif leaf in brdict['vectors']:
                    datatype = 'vector'
                    continue
            if leaf == 'pass_':
                datatype = 'scalar'
            assert(datatype == 'scalar' or datatype == 'vector')
            if datatype == 'scalar':  # If scalar, temp array has a length of 1
                scalar_holders[branch] = np.zeros((1), dtype='float32')
            else:  # If vector, temp array must have at least one element per hit
                # (liberally picked 2k)
                vector_holders[branch] = np.zeros((200000), dtype='float32')
        #print("TEMP:  Scalar, vector holders keys:")
        #print(scalar_holders.keys())
        #print(vector_holders.keys())
        # Create new branches to store nSPHits, pT (necessary for tree creation)...
        scalar_holders['nSPHits'] = np.array([0], 'i')
        scalar_holders['nTSPHits'] = np.array([0], 'i')
        scalar_holders['nRecHits'] = np.array([0], 'i')
        scalar_holders['nHRecHits'] = np.array([0], 'i')
        scalar_holders['maxPE'] = np.array([0], 'i')
        scalar_holders['TargetSPRecoilE_pt'] = np.array([0], dtype='float32')
        branchList.append('nSPHits')
        branchList.append('nTSPHits')
        branchList.append('nRecHits')
        branchList.append('nHRecHits')
        branchList.append('maxPE')
        branchList.append('TargetSPRecoilE_pt')
        # Now, go through each branch name and a corresponding branch to the tree:
        for branch, var in scalar_holders.items():
            # Need to make sure that each var is stored as the correct type (floats, ints, etc):
            if branch == 'nSPHits' or branch == 'nTSPHits' or branch == 'nRecHits' or branch == 'nHRecHits' or branch == 'maxPE':
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
            branchname = re.split(r'[./]', branch)[-1]
            #print("Found parent={}, branchname={}".format(parent, branchname))
            if parent == 'EcalScoringPlaneHits_signal' or parent == 'EcalScoringPlaneHits_sim':
                tree.Branch(branchname, var, "{}[nSPHits]/F".format(branchname))
            elif parent == 'TargetScoringPlaneHits_signal' or parent == 'TargetScoringPlaneHits_sim':
                tree.Branch(branchname+'tsp_', var, "{}[nTSPHits]/F".format(branchname+'tsp_'))
            elif parent == 'EcalRecHits_signal' or parent == 'EcalRecHits_sim': 
                tree.Branch(branchname+'rec_', var, "{}[nRecHits]/F".format(branchname+'rec_'))
            else: # else in HcalRecHits
                tree.Branch(branchname+'hrec_', var, "{}[nHRecHits]/F".format(branchname+'hrec_'))
        #print("TEMP:  Branches added to tree:")
        #for b in tree.GetListOfBranches():  print(b.GetFullName())
        #print("TEMP:  Leaves added ot tree:")
        #for b in tree.GetListOfLeaves():   print(b.GetFullName())

        print("All branches added.  Filling...")

        for i in range(nEvents):
            # For each event, fill the temporary arrays with data, then write them to the tree with Fill()
            # Also: ignore events with zero ecal hits 
            #if preselected_data['nRecHits'][i] == 0:  
                #continue                           
            for branch in branchList:
                # Contains both vector and scalar data.  Treat them differently:
                if branch in scalar_holders.keys():  # Scalar
                    # fill scalar data
                    #if i==0:  print("filling scalar", branch)
                    try:
                        scalar_holders[branch][0] = preselected_data[branch][i]
                    except IndexError:
                        print("Encountered index error filling scalar branches.")
                        print(f"SKIPPING FILE: {filename}")
                        return 0,0
                elif branch in vector_holders.keys():  # Vector
                    # fill vector data
                    #if i==0:  print("filling vector", branch)
                    for j in range(len(preselected_data[branch][i])):
                        try:
                            vector_holders[branch][j] = preselected_data[branch][i][j]
                        except IndexError:
                            print("INDEX ERROR FILLING VECTOR BRANCHES...")
                            print(f"Offending file: {filename}")
                            print(f"Offending branch: {branch}")
                            print("EXITING PROGRAM ...")
                            sys.exit(1)
                else:
                    print("FATAL ERROR:  {} not found in *_holders".format(branch))
                    assert(False)
            tree.Fill()

        # Finally, write the filled tree to the ouput file:
        outfile.Write()
        outfile.Close()
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
        with get_context("spawn").Pool(20, maxtasksperchild=1) as pool:  # Can increase this number if desired, although this depends on how many threads POD will let you run at once...
            # this number is unclear, but 20 seems right judging from the POD webpage
            # changed job file to specify 20 tasks and max 1 task per core
            # may try increasing to 40 (entire node), and changing Pool arg to 40
            # maxtasksperchild=1 makes sure processes don't linger after task completion (in case there is a memory leak somewhere in the script)
            results = pool.map(processFile, params)
        print("Finished.  Result len:", len(results))
        print(results)
        nTotal  = sum([r[0] for r in results])
        nEvents = sum([r[1] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        presel_eff[int(mass * 1000)] = float(nEvents) / nTotal if nTotal else 'no events!'
    print("Done.  Preselection efficiency: {}".format(presel_eff))

    # For running without multithreading (note:  will be extremely slow and is impractical unless you want to test/use 1-2 files at a time):
    """
    presel_eff = {}
    fiducial_ratio = {}
    for mass, filepath in file_templates.items():
        #if mass != 0:  continue
        filenum = 0
        nTotal = 0  # pre-preselection
        nEvents = 0 # post-preselection
        nPostTrigger = 0 # post-trigger
        nFiducial = 0 # post-fiducial-cut
        print("======  m={}  ======".format(mass))
        for f in glob.glob(filepath):
            # Process each file separately
            nT, nE, nP, nF = processFile([f, mass, filenum])
            nTotal += nT
            nEvents += nE
            nPostTrigger += nP
            nFiducial += nF
            filenum += 1
        print("m = {} MeV:  Read {} events, {} passed preselection, {} passed trigger, {} passed fiducial cut".format(int(mass*1000), nTotal, nEvents, nPostTrigger, nFiducial))
        presel_eff[int(mass * 1000)] = nEvents / nTotal
        fiducial_ratio[int(mass * 1000)] = nFiducial / nPostTrigger

    print("DONE.  presel_eff: ", presel_eff)
    print("DONE. fiducial ratio: ", fiducial_ratio)
    """



