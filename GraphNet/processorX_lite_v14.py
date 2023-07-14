import numpy as np
import uproot
import awkward
import glob
import math
from multiprocessing import Pool

file_templates = {
    0.001: '/home/aminali/production/v14_prod/Ap0.001GeV_1e_v3.2.2_v14_tskim/*.root',
    0.01:  '/home/aminali/production/v14_prod/Ap0.01GeV_1e_v3.2.2_v14_tskim/*.root',
    0.1:   '/home/aminali/production/v14_prod/Ap0.1GeV_1e_v3.2.2_v14_tskim/*.root',
    1.0:   '/home/aminali/production/v14_prod/Ap1GeV_1e_v3.2.3_v14_tskim/*.root',
    0:     '/home/aminali/production/v14_prod/v3.2.0_ecalPN_tskim_sizeskim/*.root'
}

# preselection values
MAX_NUM_ECAL_HITS = 50 #60  #110 for ldmx-sw v2.3.0
MAX_ISO_ENERGY = 500  #650 for ldmx-sw v2.3.0
# Results:  >0.994 vs 0.055
MAX_NUM_HCAL_HITS = 30

ECAL_SP_Z = 240.5005
ECAL_FACE_Z = 248.35
SIDE_HCAL_DZ = 600

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
        'scalars':['passesVeto_', 'nReadoutHits_', 'deepestLayerHit_',
                   'summedDet_', 'summedTightIso_', 'discValue_',
                   'recoilX_', 'recoilY_',
                   'recoilPx_', 'recoilPy_', 'recoilPz_'],
        'vectors':[]
    },
    'EcalRecHits': {
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_']  # OLD: ['id_', 'energy_']
    },
    'HcalRecHits': {
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_', 'id_', 'pe_']
    },
    'HcalVeto': {
        'scalars':['passesVeto_'],
        'vectors':[]
    }
}

def blname(branch, leaf, sig):
    if sig:
        if branch.startswith('EcalVeto') or branch.startswith('HcalVeto'):
            return '{}/{}'.format(f'{branch}_signal', leaf)
        else:
            return '{}/{}.{}'.format(f'{branch}_signal', f'{branch}_signal', leaf)

    else: # background
        if branch.startswith('EcalVeto') or branch.startswith('HcalVeto'):
            return '{}/{}'.format(f'{branch}_sim', leaf)
        else:
            return '{}/{}.{}'.format(f'{branch}_sim', f'{branch}_sim', leaf)

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz*(HitZ - Recoilz) if RPz != 0 else 0
    y_final = Recoily + RPy/RPz*(HitZ - Recoilz) if RPy != 0 else 0
    return (x_final, y_final)

def pad_array(arr):
    arr = awkward.pad_none(arr, 1, clip=True)
    arr = awkward.fill_none(arr, 0)
    return awkward.flatten(arr)

def getSection(hid):
    SECTION_MASK = 0x7 # space for up to 7 sections                                 
    SECTION_SHIFT = 18 
    # Hcal Section: BACK = 0, TOP = 1, BOTTOM = 2, RIGHT = 3, LEFT = 4
    return (hid >> SECTION_SHIFT) & SECTION_MASK

def processFile(input_vars):
    # input_vars is a list: [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0] 
    mass = input_vars[1]
    filenum = input_vars[2]

    sig = True
    if not mass:
        sig = False

    branchList = []
    for branchname, leafdict in data_to_save.items():
        if mass:
            branchname_ = f'{branchname}_signal'
        else:
            branchname_ = f'{branchname}_sim'
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            if branchname == "EcalVeto" or branchname == "HcalVeto":
                branchList.append(branchname_ + '/' + leaf)
            else:
                branchList.append(branchname_ + '/' + branchname_ + '.' + leaf)

    # count total events
    #file = uproot.open(filename)
    with uproot.open(filename) as file:
        if len(file.keys()) == 0:
            print("FOUND ZOMBIE: {} SKIPPING...".format(filename))
            return 0, 0, 0 ,0, 0
    with uproot.open(filename)['LDMX_Events'] as t:
        raw_data = t.arrays(branchList)
        nTotalEvents = len(raw_data[blname('EcalRecHits', 'xpos_', sig)])

        # preselection #
        el = (raw_data[blname('EcalVeto', 'nReadoutHits_', sig)] < MAX_NUM_ECAL_HITS) * (raw_data[blname('EcalVeto', 'summedTightIso_', sig)] < MAX_ISO_ENERGY)

        preselected_data = {}
        for branch in branchList:
            preselected_data[branch] = raw_data[branch][el]
        nEvents = len(preselected_data[blname('EcalVeto', 'summedTightIso_', sig)])

        # simple hcal veto #
        hv1 = preselected_data[blname('HcalVeto', 'passesVeto_', sig)] == 1

        selected_data = {}
        for branch in branchList:
            selected_data[branch] = preselected_data[branch][hv1]
        nPassesVeto = len(selected_data[blname('HcalVeto', 'passesVeto_', sig)])

        # hcal hits cut (boosted preselection) #
        HE_data = preselected_data[blname('HcalRecHits', 'energy_', sig)]
        nHRecHits = np.zeros(nEvents)
        for i in range(nEvents):
            nHRecHits[i] = sum(HE_data[i] > 0)
            if len(HE_data[i]) == 0:
                nHRecHits[i] = 0
        preselected_data['nHRecHits'] = np.array(nHRecHits)
        branchList.append('nHRecHits')

        hc = preselected_data['nHRecHits'] < MAX_NUM_ECAL_HITS

        preselected_data_2 = {}
        for branch in branchList:
            preselected_data_2[branch] = preselected_data[branch][hc]
        nEvents2 = len(preselected_data_2['nHRecHits'])

        # crude modified hcal veto #

        # Find Ecal SP recoil electron (with maximum momentum)
        recoilZ = preselected_data[blname('EcalScoringPlaneHits','z_', sig)]
        px = preselected_data[blname('EcalScoringPlaneHits','px_', sig)]
        py = preselected_data[blname('EcalScoringPlaneHits','py_', sig)]
        pz = preselected_data[blname('EcalScoringPlaneHits','pz_', sig)]
        pdgID = preselected_data[blname('EcalScoringPlaneHits','pdgID_', sig)]
        trackID = preselected_data[blname('EcalScoringPlaneHits','trackID_', sig)]
        
        e_cut = []
        for i in range(len(px)):
            e_cut.append([])
            for j in range(len(px[i])):
                e_cut[i].append(False)
        for i in range(len(px)):
            maxP = 0
            e_index = 0
            for j in range(len(px[i])):
                P = np.sqrt(px[i][j]**2 + py[i][j]**2 + pz[i][j]**2)
                if (pdgID[i][j] == 11 and trackID[i][j] == 1 and recoilZ[i][j] > 240 and recoilZ[i][j] < 241 and P > maxP):
                    maxP = P
                    e_index = j
            if maxP > 0:
                e_cut[i][e_index] = True

        recoilX = pad_array(preselected_data[blname('EcalScoringPlaneHits','x_', sig)][e_cut])
        recoilY = pad_array(preselected_data[blname('EcalScoringPlaneHits','y_', sig)][e_cut])
        recoilPx = pad_array(px[e_cut])
        recoilPy = pad_array(py[e_cut])
        recoilPz = pad_array(pz[e_cut])

        # Build recoil e trajectory and follow in steps
        # At each step, check to see if in side hcal
        # If so, which one? (mark 1,2,3,4 for top,bot,right,left...-1 for none)
        # "turn off" the side hcal with the most counts (i.e., treat any pe value from a hit in that side as 0 when computing maxPE)
        # Will likely switch to a "sweeping cone" method later
        N = len(recoilX)
        n_steps = 101
        steps = np.linspace(ECAL_FACE_Z, ECAL_FACE_Z + SIDE_HCAL_DZ, n_steps)
        section_id = np.zeros((N, n_steps))

        for j in range(n_steps):
            for i in range(N):
                fXY = projection(recoilX[i], recoilY[i], ECAL_SP_Z, recoilPx[i], recoilPy[i], recoilPz[i], steps[j]) 
                x = fXY[0] 
                y = fXY[1]

                if x <= -400 and y >= -300:
                    section_id[i][j] = 4             # LEFT HCAL
                elif x >= 400 and y <= 300:
                    section_id[i][j] = 3             # RIGHT HCAL
                elif x <= 400 and y <= -300:
                    section_id[i][j] = 2             # BOTTOM HCAL
                elif x >= -400 and y >= 300:
                    section_id[i][j] = 1             # TOP HCAL
                else:
                    section_id[i][j] = -1            # nonphysical placeholder


        counts = np.array([np.sum(section_id == n, axis=1) for n in [1, 2, 3, 4]])  # sum 4 bool arrays shape (N, n_steps) and place in counts --> result is (4, N) array 

        sectionOff = np.zeros(N)
        for i in range(N):
            counts_col = counts[:,i]  # Rank 1 view of i'th column of counts (length 4)
            idx = np.argmax(counts_col) # index of max value 
            # Record side hcal with section_id given by idx + 1...unless the max value in counts_col is zero then set to -1
            sectionOff[i] = idx + 1 if counts_col[idx] != 0 else -1

        # Omit the side hcal given by sectionOff when calculating maxPE...then apply maxPE < 5 hcal veto 
        PE = preselected_data[blname('HcalRecHits', 'pe_', sig)]
        hcal_id = preselected_data[blname('HcalRecHits', 'id_', sig)]

        modmaxPE = np.zeros(len(PE))
        for i in range(len(PE)):
            PE_temp = np.zeros(2000)
            for j in range(len(PE[i])):
                if getSection(hcal_id[i][j]) != sectionOff[i]:
                    PE_temp[j] = PE[i][j]
            modmaxPE[i] = np.max(PE_temp)

        hv2 = modmaxPE < 5
        
        selected_data_2 = {}
        for branch in branchList:
            selected_data_2[branch] = preselected_data[branch][hv2]
        nPassesModVeto1 = len(selected_data_2[blname('HcalVeto', 'passesVeto_', sig)])
   

    return (nTotalEvents, nEvents, nPassesVeto, nEvents2, nPassesModVeto1)

if __name__ == '__main__':
    presel_eff = {}
    hcalveto_eff = {}
    presel_eff_2 = {}
    modhcalveto1_eff = {}
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
        nPassesModVeto1 = sum([r[4] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection, {} passed preselection + hcal veto, {} passed double preselection, {} passed preselection + mod hcal veto 1".format(int(mass*1000), nTotal, nEvents, nPassesVeto, nEvents2, nPassesModVeto1))
        presel_eff[int(mass * 1000)] = float(nEvents) / nTotal if nTotal != 0 else 'no events'
        hcalveto_eff[int(mass * 1000)] = float(nPassesVeto) / nEvents if nEvents != 0 else 'no events'
        presel_eff_2[int(mass * 1000)] = float(nEvents2) / nTotal if nTotal != 0 else 'no events'
        modhcalveto1_eff[int(mass * 1000)] = float(nPassesModVeto1) / nEvents if nEvents != 0 else 'no events'
    print("Done.  Preselection efficiency: {}, Hcal Veto efficiency: {}, Boosted preselection efficiency: {}, Mod Hcal Veto 1 efficiency: {}".format(presel_eff, hcalveto_eff, presel_eff_2, modhcalveto1_eff))