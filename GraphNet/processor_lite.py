import numpy as np
import uproot
import awkward
import glob
import math
from multiprocessing import Pool

file_templates = {
    0.001: '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*0.001*.root',  
    0.01:  '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*0.01*.root',
    0.1:   '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*0.1*.root',
    1.0:   '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/signal/*1.0*.root',
    0:     '/home/pmasterson/events/v3.0.0_tag_standard_skimmed/photonuclear/*.root'
}

# preselection values
MAX_NUM_ECAL_HITS = 50 #60  #110 for ldmx-sw v2.3.0
MAX_ISO_ENERGY = 500  #650 for ldmx-sw v2.3.0
# Results:  >0.994 vs 0.055

# data[branch_name][scalars/vectors][leaf_name]
data = {
    'EcalScoringPlaneHits_v12': {
        'scalars':[],
        'vectors':['pdgID_', 'trackID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    'EcalVeto_v12': {
        'scalars':['nReadoutHits_', 'summedTightIso_'],
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

def _load_cellMap(version):
    cellMap = {}
    for i, x, y in np.loadtxt('data/%s/cellmodule.txt' % version):
        cellMap[i] = (x, y)
    global cells 
    cells = np.array(list(cellMap.values()))
    print("Loaded {} detector info".format(version))

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
    # input_vars is a list: [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0] 
    mass = input_vars[1]
    filenum = input_vars[2]

    branchList = []
    for branchname, leafdict in data.items():
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            if branchname == "EcalVeto_v12":
                branchList.append(branchname + '/' + leaf)
            else:
                branchList.append(branchname + '.' + leaf)

    file = uproot.open(filename)
    if len(file.keys()) == 0:
        print("FOUND ZOMBIE: {} SKIPPING...".format(filename))
        return 0, 0, 0, 0
    t = uproot.open(filename)['LDMX_Events']
    tmp = t.arrays(['EcalVeto_v12/nReadoutHits_'])
    nTotalEvents = len(tmp)

    raw_data = t.arrays(branchList) # raw_data['EcalVeto_v12/nReadoutHits_'] = awkward array containing the value of nReadoutHits_ for each event, and so on.

    # Preselection
    el = (raw_data['EcalVeto_v12/nReadoutHits_'] < MAX_NUM_ECAL_HITS) * (raw_data['EcalVeto_v12/summedTightIso_'] < MAX_ISO_ENERGY)
    selected_data = {}
    for branch in branchList:
        selected_data[branch] = raw_data[branch][el]
    nEvents = len(selected_data['EcalVeto_v12/summedTightIso_'])

    # Trigger
    eid = selected_data['EcalRecHits_v12.id_']
    energy = selected_data['EcalRecHits_v12.energy_']
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

    for branch in branchList:
        selected_data[branch] = selected_data[branch][t_cut]
    nPostTrigger = len(selected_data['EcalScoringPlaneHits_v12.x_'])

    # Fiducial cut 
    # First find Ecal SP recoil electron (with maximum momentum)
    recoilZ = selected_data['EcalScoringPlaneHits_v12.z_']
    px = selected_data['EcalScoringPlaneHits_v12.px_']
    py = selected_data['EcalScoringPlaneHits_v12.py_']
    pz = selected_data['EcalScoringPlaneHits_v12.pz_']
    pdgID = selected_data['EcalScoringPlaneHits_v12.pdgID_']
    trackID = selected_data['EcalScoringPlaneHits_v12.trackID_']
    
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

    recoilX = pad_array(selected_data['EcalScoringPlaneHits_v12.x_'][e_cut])
    recoilY = pad_array(selected_data['EcalScoringPlaneHits_v12.y_'][e_cut])
    recoilPx = pad_array(px[e_cut])
    recoilPy = pad_array(py[e_cut])
    recoilPz = pad_array(pz[e_cut])

    # Apply fiducial test to recoil electron
    N = len(recoilX)
    f_cut = np.zeros(N, dtype = bool) 
    for i in range(N):
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

    return (nTotalEvents, nEvents, nPostTrigger, nFiducial)

if __name__ == '__main__':
    presel_eff = {}
    fiducial_ratio = {}
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
        nPostTrigger = sum([r[2] for r in results])
        nFiducial = sum([r[3] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection, {} passed trigger, {} passed fiducial cut".format(int(mass*1000), nTotal, nEvents, nPostTrigger, nFiducial))
        presel_eff[int(mass * 1000)] = float(nEvents) / nTotal
        fiducial_ratio[int(mass * 1000)] = float(nFiducial) / nPostTrigger
    print("Done.  Preselection efficiency: {}, Fiducial ratio: {}".format(presel_eff, fiducial_ratio))