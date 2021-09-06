#!/home/pmasterson/miniconda3/envs/torchroot/bin/python

#SBATCH -n 20
#SBATCH --output=slurm_file_processor.out

# NOTE:  was --nodes=1, --ntasks-per-node 2

import numpy as np
import uproot
import awkward
import glob
import os
import re
import math
print("Importing ROOT")
import ROOT as r
print("Imported ROOT.  Starting...")
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

# Directory to write output files to:
output_dir = '/home/duncansw/GraphNet_input/v12/processed'
# Locations of the 2.3.0 ldmx-sw ROOT files to process+train on:
file_templates = {
    0.001: '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.001*.root',
    0.01:  '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.01*.root',
    0.1:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*0.1*.root',
    1.0:   '/home/pmasterson/GraphNet_input/v12/sig_extended_extra/*1.0*.root',
    0:     '/home/dgj1118/LDMX-scripts/GraphNet/background_230_trunk/*.root'
}
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
"""
# 3.0.0:
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

# Standard preselection values (-> 95% sig/5% bkg)
MAX_NUM_ECAL_HITS = 60  #110  #Now MUCH lower!  >99% of 1 MeV sig should pass this. (and >10% of bkg)
MAX_ISO_ENERGY = 500  # NOTE:  650 passes 99.99% sig, ~13% bkg for 3.0.0!  Lowering...
# Results:  >0.994 vs 0.055

# Branches to save:
# Quantities labeled with 'scalars' have a single value per event.  Quantities labeled with 'vectors' have
# one value for every hit (e.g. number of ecal hits vs x position of each hit).
# (Everything else can be safely ignored)
# data[branch_name][scalars/vectors][leaf_name]
data_to_save = {
    'EcalScoringPlaneHits_v12': {
        'scalars':[],
        'vectors':['pdgID_', 'trackID_', 'layerID_', 'x_', 'y_', 'z_',
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

scoringPlaneZ = 240.5005
ecalFaceZ = 248.35
cell_radius = 5.0

def dist(p1, p2):
    return math.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz*(HitZ - Recoilz) if RPz != 0 else 0
    y_final = Recoily + RPy/RPz*(HitZ - Recoilz) if RPy != 0 else 0
    return (x_final, y_final)

def _load_cellMap(version='v12'):
    cellMap = {}
    for i, x, y in np.loadtxt('data/%s/cellmodule.txt' % version):
        cellMap[i] = (x, y)
    global cells 
    cells = np.array(list(cellMap.values()))
    print("Loaded detector info")

def get_layer_id(cid):
    layer = (awkward.to_numpy(awkward.flatten(cid)) >> 17) & 0x3F

    def unflatten_array(x, base_array):
        return awkward.Array(awkward.layout.ListOffsetArray32(awkward.layout.Index32(base_array.layout.offsets),awkward.layout.NumpyArray(np.array(x, dtype='float32'))))

    layer_id = unflatten_array(layer,cid)
    return layer_id

def pad_array(arr):
    arr = awkward.pad_none(arr, 1, clip=True)
    arr = awkward.fill_none(arr, 0)
    return awkward.flatten(arr)

def processFile(input_vars):
    # input_vars is a list:
    # [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0]  # Apparently this is the easiest approach to multiple args...
    mass = input_vars[1]
    filenum = input_vars[2]

    print("Processing file {}".format(filename))
    if mass == 0:
        outfile_name = "v300_pn_trigger_{}.root".format(filenum)
    else:
        outfile_name = "v300_{}_trigger_{}.root".format(mass, filenum)
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
    print("Before preselection: found {} events".format(nTotalEvents))

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
    print("After preselection: skimming from {} events".format(nEvents))

    # Trigger cut
    eid = preselected_data['EcalRecHits_v12.id_']
    energy = preselected_data['EcalRecHits_v12.energy_']
    pos = energy > 0
    eid = eid[pos]
    energy = energy[pos]
    layer_id = get_layer_id(eid)

    t_cut = np.zeros(len(eid), dtype = bool)

    for event in range(len(eid)):
        en = 0.0
        for hit in range(len(eid[event])):
            if layer_id[event][hit] < 20.0:
                en += energy[event][hit]
        if en < 1500.0:
            t_cut[event] = 1

    selected_data = {}
    for branch in branchList:
        selected_data[branch] = preselected_data[branch][t_cut]

    nPostTrigger = len(selected_data['EcalScoringPlaneHits_v12.x_'])
    print("After trigger: skimming from {} events".format(nPostTrigger))

    #Fiducial cut 

    # Find Ecal SP recoil electron (with maximum momentum)
    recoilZ = selected_data['EcalScoringPlaneHits_v12.z_']
    px = selected_data['EcalScoringPlaneHits_v12.px_']
    py = selected_data['EcalScoringPlaneHits_v12.py_']
    pz = selected_data['EcalScoringPlaneHits_v12.pz_']
    pdgID = selected_data['EcalScoringPlaneHits_v12.pdgID_']
    trackID = selected_data['EcalScoringPlaneHits_v12.trackID_']
    
    e_cut = []
    for i in range(len(pz)):
        e_cut[i].append([])
        for j in range(len(pz[i])):
            e_cut[i][j].append(False)
    
    for i in range(len(pz)):
        maxP = 0
        e_index = 0
        for j in range(len(pz[i])):
            P = np.sqrt(px[i][j]**2 + py[i][j]**2 + pz[i][j]**2)
            if (pdgID[i][j] == 11 and trackID[i][j] == 1 and recoilZ[i][j] > 240 and recoilZ[i][j] < 241 and P > maxP):
                maxP = P
                e_index = j
        if maxP > 0:
            e_cut[i][e_index] = True

    recoilX = pad_array(selected_data['EcalScoringPlaneHits_v12.x_'][e_cut])
    recoilY = pad_array(selected_data['EcalScoringPlaneHits_v12.y_'][e_cut])
    recoilPx = pad_array(pX[e_cut])
    recoilPy = pad_array(pY[e_cut])
    recoilPz = pad_array(pZ[e_cut])

    # Apply fiducial test to recoil electron
    events = len(recoilX)
    f_cut = np.zeros(N, dtype = bool)
    
    for i in range(events):
        fiducial = False
        fXY = projection(recoilX[i], recoilY[i], scoringPlaneZ, recoilPx[i], recoilPy[i], recoilPz[i], ecalFaceZ)

        if not recoilX[i] == 0 and not recoilY[i] == 0 and not recoilPx[i] == 0 and not recoilPy[i] == 0 and not recoilPz[i] == 0:
            for j in range(len(cells)):
                celldis = dist(cells[j], fXY)
                if celldis <= cell_radius:
                    fiducial = True
                    break
        
        if recoilX[i] == 0 and recoilY[i] == 0 and recoilPx[i] == 0 and recoilPy[i] == 0 and recoilPz[i] == 0:
            fiducial = False

        if fiducial == True:
            f_cut[i] = 1

    for branch in branchList:
        selected_data[branch] = selected_data[branch][f_cut]

    nFiducial = len(selected_data['EcalScoringPlaneHits_v12.x_'])
    print("After fiducial cut: found {} events".format(nFiducial))

    # Next, we have to compute TargetSPRecoilE_pt here instead of in train.py.  (This involves TargetScoringPlane
    # information that ParticleNet doesn't need, and that would take a long time to load with the lazy-loading
    # approach.)
    # For each event, find the recoil electron (maximal recoil pz):
    pdgID_ = t['TargetScoringPlaneHits_v12.pdgID_'].array()[el][t_cut][f_cut]
    z_     = t['TargetScoringPlaneHits_v12.z_'].array()[el][t_cut][f_cut]
    px_    = t['TargetScoringPlaneHits_v12.px_'].array()[el][t_cut][f_cut]
    py_    = t['TargetScoringPlaneHits_v12.py_'].array()[el][t_cut][f_cut]
    pz_    = t['TargetScoringPlaneHits_v12.pz_'].array()[el][t_cut][f_cut]
    tspRecoil = []
    for i in range(nFiducial):
        max_pz = 0
        recoil_index = 0  # Store the index of the recoil electron
        for j in range(len(pdgID_[i])):
            # Constraint on z ensures that the SP downstream of the target is used
            if pdgID_[i][j] == 11 and z_[i][j] > 0.176 and z_[i][j] < 0.178 and pz_[i][j] > max_pz:
                max_pz = pz_[i][j]
                recoil_index = j
        # Calculate the recoil SP
        tspRecoil.append(np.sqrt(px_[i][recoil_index]**2 + py_[i][recoil_index]**2))
    # Put it in the selected_data and treat it as an ordinary branch from here on out
    selected_data['TargetSPRecoilE_pt'] = np.array(tspRecoil)

    # Additionally, add new branches storing the length for vector data (number of SP hits, number of ecal hits):
    nSPHits = []
    nRecHits = []
    x_data = selected_data['EcalScoringPlaneHits_v12.x_']
    E_data = selected_data['EcalRecHits_v12.energy_']
    for i in range(nFiducial):
        # NOTE:  max num hits may exceed MAX_NUM...this is okay.
        nSPHits.append(len(x_data[i]))
        nRecHits.append(len(E_data[i]))
    selected_data['nSPHits'] = np.array(nSPHits)
    selected_data['nRecHits'] = np.array(nRecHits)


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

    for i in range(nFiducial):
        # For each event, fill the temporary arrays with data, then write them to the tree with Fill()
        for branch in branchList:
            # Contains both vector and scalar data.  Treat them differently:
            if branch in scalar_holders.keys():  # Scalar
                # fill scalar data
                scalar_holders[branch][0] = selected_data[branch][i]
            elif branch in vector_holders.keys():  # Vector
                # fill vector data
                for j in range(len(selected_data[branch][i])):
                    vector_holders[branch][j] = selected_data[branch][i][j]
            else:
                print("FATAL ERROR:  {} not found in *_holders".format(branch))
                assert(False)
        tree.Fill()

    # Finally, write the filled tree to the ouput file:
    outfile.Write()
    print("FINISHED.  File written to {}.".format(outfile_path))

    return (nTotalEvents, nEvents, nPostTrigger, nFiducial)


if __name__ == '__main__':
    # New approach:  Use multiprocessing
    #pool = Pool(16) -> Run 16 threads/process 16 files in parallel
    
    presel_eff = {}
    fiducial_ratio = {}
    # For each signal mass and for PN background:
    for mass, filepath in file_templates.items():
        print("======  m={}  ======".format(mass))
        # Assemble list of function params
        # These get passed to processFile() when Pool requests them
        params = []
        for filenum, f in enumerate(glob.glob(filepath)):
            params.append([f, mass, filenum])  # list will be passed to ProcessFile:  processFile([filepath, mass, file_number])
        with Pool(20) as pool:  # Can increase this number if desired, although this depends on how many threads POD will let you run at once...
            # this number is unclear, but 20 seems right judging from the POD webpage
            results = pool.map(processFile, params)
        print("Finished.  Result len:", len(results))
        print(results)
        nTotal  = sum([r[0] for r in results])
        nEvents = sum([r[1] for r in results])
        nPostTrigger = sum([r[2] for r in results])
        nFiducial = sum([r[3] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection, {} passed trigger, {} passed fiducial cut".format(int(mass*1000), nTotal, nEvents, nPostTrigger, nFiducial))
        presel_eff[int(mass * 1000)] = float(nEvents) / nTotal
        fiducial_ratio[int(mass * 1000)] = float(nFiducial) / nPostTrigger
    print("Done.  Presel_eff: {}, Fiducial Ratio: {}".format(presel_eff, fiducial_ratio))

    # For running without multithreading (note:  will be extremely slow and is impractical unless you want to test/use 1-2 files at a time):
    """
    presel_eff = {}
    fiducial_ratio = {}
    for mass, filepath in file_templates.items():
        #if mass != 0:  continue
        filenum = 0
        nTotal = 0  # pre-preselection
        nEvents = 0 # post-preselection
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



