from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import tqdm
import uproot
import awkward
import concurrent.futures

# NEW:
import ROOT as r

import psutil
import gc  # May reduce RAM usage

import time # Can remove whenever

executor = concurrent.futures.ThreadPoolExecutor(12)

torch.set_default_dtype(torch.float32)

MAX_NUM_ECAL_HITS = 110  # Still need this to determine array sizes...110 gives >98% efficiency, so using that.
MAX_ISO_ENERGY = 650  # 600 gives slightly under 95% sig eff; shoot for a bit over

# NEW: Radius of containment data
#from 2e (currently not used)
radius_beam_68 = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]

#from 1e
radius_recoil_68_p_0_500_theta_0_10 = [4.045666158618167, 4.086393662224346, 4.359141107602775, 4.666549994726691, 5.8569181911416015, 6.559716356124256, 8.686967529043072, 10.063482736354674, 13.053528344041274, 14.883496407943747, 18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406, 28.329169392203216, 29.468032123356345, 34.03271241527079, 35.03747443690781, 38.50748727211848, 39.41576583301171, 42.63622296033334, 45.41123601592071, 48.618139095742876, 48.11801717451056, 53.220539860213655, 58.87753380915155, 66.31550881539764, 72.94685877928593, 85.95506228335348, 89.20607201266672, 93.34370253818409, 96.59471226749734, 100.7323427930147, 103.98335252232795]

radius_recoil_68_p_500_1500_theta_0_10 = [4.081926458777424, 4.099431732299409, 4.262428482867968, 4.362017581473145, 4.831341579961153, 4.998346041276382, 6.2633736512415705, 6.588371889265881, 8.359969947444522, 9.015085558044309, 11.262722588206483, 12.250305471269183, 15.00547660437276, 16.187264014640103, 19.573764900578503, 20.68072032434797, 24.13797140783321, 25.62942209291236, 29.027596514735617, 30.215039667389316, 33.929540248019585, 36.12911729771914, 39.184563500620946, 42.02062468386282, 46.972125628650204, 47.78214816041894, 55.88428562462974, 59.15520134927332, 63.31816666637158, 66.58908239101515, 70.75204770811342, 74.022963432757, 78.18592874985525, 81.45684447449884]

radius_recoil_68_theta_10_20 = [4.0251896715647115, 4.071661598616328, 4.357690094817289, 4.760224640141712, 6.002480766325418, 6.667318981016246, 8.652513285172342, 9.72379373302137, 12.479492693251478, 14.058548828317289, 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302, 31.768946746958296, 35.79247330197688, 37.27810357669942, 41.657281051476545, 42.628141392692626, 47.94208483539388, 49.9289473559796, 54.604030254423975, 53.958762417361655, 53.03339560920388, 57.026277390001425, 62.10810455035879, 66.10098633115634, 71.1828134915137, 75.17569527231124, 80.25752243266861, 84.25040421346615, 89.33223137382352, 93.32511315462106]

radius_recoil_68_theta_20_end = [4.0754238481177705, 4.193693485630508, 5.14209420056253, 6.114996249971468, 7.7376807326481645, 8.551663213602291, 11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604, 25.04088272634407, 28.853696411302344, 34.72538105333071, 40.21218694947545, 46.07344239520299, 50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047, 74.82378572939959, 89.4528387422834, 93.18228303096758, 92.51751129204555, 98.80228884380018, 111.17537347472128, 120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185, 177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696]

radius_68 = [radius_beam_68,radius_recoil_68_p_0_500_theta_0_10, radius_recoil_68_p_500_1500_theta_0_10,radius_recoil_68_theta_10_20,radius_recoil_68_theta_20_end]



def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array()
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)



class ECalHitsDataset(Dataset):

    def __init__(self, siglist, bkglist, load_range=(0, 1), obs_branches=[], coord_ref=None, detector_version='v12', nRegions=1):
        super(ECalHitsDataset, self).__init__()
        print("Initializing EcalHitsDataset")
        #print("GPU usage: {}%".format(psutil.virtual_memory().percent))
        #print("GPU available: {} GB".format(psutil.virtual_memory().available/1e9))
        # first load cell map
        self._load_cellMap(version=detector_version)
        # NOTE:  the new skimmed input doesn't have sub-branches anymore!
        self._id_branch = 'id_rec_'  #'EcalRecHits_v12.id_'
        self._energy_branch = 'energy_rec_'  #'EcalRecHits_v12.energy_'
        self._branches = [self._id_branch, self._energy_branch]

        self.obs_branches = obs_branches
        # NOTE:  Need to explicitly keep track of and save all obs_dict data!  Fortunately, order doesn't matter.
        self.obs_dict = {br:[] for br in self.obs_branches}
        # Also need to keep track of events that have been loaded into obs_dict, to ensure no duplicates
        # Just store a list of the event numbers
        self.loaded_events = []

        self.coord_ref = coord_ref
        assert(detector_version == 'v12')  # v9 compatibility would be nontrivial to add, and is probably unnecessary

        self.nRegions = nRegions

        # General approach to init():
        # - Input events have all been preselected.
        # - All event data is stored in a "simple" root tree with no sub-branches
        # - Need to create a mapping:  event number -> returns sig/bkg, root file, and evt number within that file
        #    - [[mass, filename, i_file], ...]
        #    - Get element i of the list whenever PN requests an event, and find the event location based on that info

        # event_list:  [[mass, filename, i_file], ...]
        self.event_list = []
        self.extra_labels = []  # mass in MeV if sig, 0 if bkg
        print("Filling event_list")
        # fill event_list to make event access easy
        filelist = {}
        for label, fname in bkglist.items():
            filelist[label] = fname
        for label, fname in siglist.items():
            filelist[label] = fname

        for extra_label in filelist:  # For each mass:
            filepath, max_events = filelist[extra_label]
            if max_events == -1:
                max_events = 1e8  # Unrealistically large so it never constrains the results
            num_loaded_events = 0  # Number of events so far for this mass
            print("   Filling for m={}".format(extra_label))
            for fp in glob.glob(filepath):
                # For each file, check the number of events, then add to event_list accordingly
                tfile = r.TFile(fp)
                ttree = tfile.Get('skimmed_events')
                f_events = ttree.GetEntries()  # Num events in file
                # load_range specifies fraction of file to load from.
                start, stop = [int(x * f_events) for x in load_range]

                f_event = start
                while num_loaded_events < max_events and f_event < stop:
                    self.event_list.append([extra_label if extra_label <= 1 else 1, fp, f_event])
                    self.extra_labels.append(extra_label)
                    num_loaded_events += 1
                    f_event += 1
                print("      {} events in file, {} total for current mass".format(f_event, num_loaded_events))
            print("   Finished m={}:  using {} events".format(extra_label, num_loaded_events))

        self.extra_labels = np.array(self.extra_labels)
        self.label = np.array([1 if l > 0 else 0 for l in self.extra_labels])  # 1 if sig, 0 if bkg

        print("Initialization finished.")
        #print("GPU usage: {}".format(psutil.virtual_memory().percent))

        #print("Initialized")



    @property
    def num_features(self):
        # When would this *not* be 5?  Not worried about generalizing atm
        return 5

    def __len__(self):
        return len(self.event_list)


    #def load_event(self, label, filename, file_index):   # name of file containing event, index (in file) of event
    def __getitem__(self, i):
        # On-demand, read event file_index from filename and process it
        # By assumption, events have already been preselected!
        # returns:  label, coords, features

        label, filename, file_index = self.event_list[i]

        self.obs_data = {k:[] for k in self.obs_branches}
        # Also, keep track of all event nums that have been added to obs_data

        # Load all necessary info into var_data and obs_data:

        # NOW HEAVILY REDONE.
        # Goal:  load data from selected event (1 event!) into var_dict.
        # Use ROOT, not uproot.
        # *NOTE*:  Will still load data before processing, but now only storing data for ONE event.

        tfile = r.TFile(filename) # NOTE:  This is the bottleneck!
        self.ttree = tfile.Get('skimmed_events')
        # Prepare to load data from event [file_index]:
        self.ttree.GetEntry(file_index)
        self._load_sp_data()
        # Fill var_dict and obs_dict:
        # obs_dict contains obs_branches info loaded from train.py; saved for plotting
        # var_dict contains info necessary for PN:  x, y, z, layer, log(E); more if other regions included
        var_data, obs_data = self._read_event()
        # var_data is used by _load_...().  o_d data must be saved.  First, check whether event has already been recorded:
        if not i in self.loaded_events:
            for branch in self.obs_branches:
                self.obs_dict[branch].append(obs_data[branch])
            self.loaded_events.append(i)



        # training features

        # NOTE:  Now needs to have an outer index anyway!  [[a, b...]] for 1-region PN

        coordinates = np.stack((var_data['x_'], var_data['y_'], var_data['z_']), axis=1)
        features    = np.stack((var_data['x_'], var_data['y_'], var_data['z_'],
                                var_data['layer_id_'], var_data['log_energy_']), axis=1)
        #print("Coordinates shape is:", coordinates.shape)
        return coordinates, features, label


    def _load_sp_data(self):
        #print("    Usage before coord ref: {}".format(psutil.virtual_memory().percent))
        pdgID_leaf = self.ttree.GetLeaf('pdgID_')
        z_leaf     = self.ttree.GetLeaf('z_')
        pz_leaf    = self.ttree.GetLeaf('pz_')
        pdgID_ = [int(pdgID_leaf.GetValue(i)) for i in range(pdgID_leaf.GetLen())]
        z_     = [z_leaf.GetValue(i)          for i in range(z_leaf.GetLen())    ]
        pz_    = [pz_leaf.GetValue(i)         for i in range(pz_leaf.GetLen())   ]
        el_ = 0  # SP index of recoil electron
        pmax = 0  # Max pz
        max_index = 0

        # Find the SP electron (first layer, maximum pZ)
        for j in range(pdgID_leaf.GetLen()):
            if pdgID_[j] == 11 and z_[j] > 240 and z_[j] < 241 and pz_[j] > pmax:
                pmax = pz_[j]
                el_ = j
        has_e = pmax != 0  # Check whether event has a SP electron
        
        etraj_x_sp  = self.ttree.GetLeaf('x_').GetValue(el_)
        etraj_y_sp  = self.ttree.GetLeaf('y_').GetValue(el_)
        etraj_z_sp  = self.ttree.GetLeaf('z_').GetValue(el_)
        etraj_px_sp = self.ttree.GetLeaf('px_').GetValue(el_)
        etraj_py_sp = self.ttree.GetLeaf('py_').GetValue(el_)
        etraj_pz_sp = self.ttree.GetLeaf('pz_').GetValue(el_)

        self.etraj_sp = np.array((etraj_x_sp, etraj_y_sp, etraj_z_sp))

        # Create vectors holding the electron/photon momenta so the trajectory projections can be found later
        # Set xtraj_p_norm relative to z=1 to make projecting easier:
        E_beam = 4000.0  # In MeV
        target_dist = 241.5 # distance from ecal to target, mm
        
        if etraj_pz_sp != 0 and has_e:
            self.enorm_sp = np.array((etraj_px_sp/etraj_pz_sp, etraj_py_sp/etraj_pz_sp, 1.0))
            self.pnorm_sp = np.array((-etraj_px_sp/(E_beam - etraj_pz_sp), -etraj_py_sp/(E_beam - etraj_pz_sp), 1.0))
            self.ptraj_sp = np.array((etraj_x_sp + target_dist*(self.pnorm_sp[0] - self.enorm_sp[0]),
                                      etraj_y_sp + target_dist*(self.pnorm_sp[1] - self.enorm_sp[1]),
                                      etraj_z_sp))
        else:
            self.enorm_sp = np.array((0,0,0))
            self.pnorm_sp = np.array((0,0,0))
            self.ptraj_sp = np.array((0,0,0))



    def _read_event(self):
        # Fill var_dict and obs_dict:
        # obs_dict contains obs_branches info loaded from train.py; saved for plotting
        # var_dict contains info necessary for PN:  x, y, z, layer, log(E); more if other regions included

        eid_leaf    = self.ttree.GetLeaf(self._id_branch)
        energy_leaf = self.ttree.GetLeaf(self._energy_branch)
        eid    = np.array([eid_leaf.GetValue(i)    for i in range(eid_leaf.GetLen())   ], dtype='int')  #table[self._id_branch]
        energy = np.array([energy_leaf.GetValue(i) for i in range(energy_leaf.GetLen())], dtype='float32')  #table[self._energy_branch]
        pos = (energy > 0)
        eid = eid[pos]  # Gets rid of all (AND ONLY) hits with 0 energy
        energy = energy[pos]
        (x, y, z), layer_id = self._parse_cid(eid)  # layer_id > 0, so can use layer_id-1 to index e/ptraj_ref

        # Now, work with table['etraj_ref'] and table['ptraj_ref'].
        # Create lists:  x/y/z_e, p
        # For each event, look through all hits.
        # - Determine whether hit falls inside either the e or p RoCs
        # - If so, fill corresp xyzlayer, energy, eid lists...
        x_          = np.zeros((self.nRegions, MAX_NUM_ECAL_HITS), dtype='float32')
        y_          = np.zeros((self.nRegions, MAX_NUM_ECAL_HITS), dtype='float32')
        z_          = np.zeros((self.nRegions, MAX_NUM_ECAL_HITS), dtype='float32')
        log_energy_ = np.zeros((self.nRegions, MAX_NUM_ECAL_HITS), dtype='float32')
        layer_id_   = np.zeros((self.nRegions, MAX_NUM_ECAL_HITS), dtype='float32')

        #print("    Usage after array creation: {}".format(psutil.virtual_memory().percent))
        

        for j in range(eid_leaf.GetLen()):  #range(MAX_NUM_ECAL_HITS):  # For every hit...

            layer_index = int(layer_id[j])
            # Calculate xy coord of point on projected trajectory in same layer
            delta_z = self._layerZs[layer_index] - self.etraj_sp[2]
            etraj_point = (self.etraj_sp[0] + self.enorm_sp[0]*delta_z, self.etraj_sp[1] + self.enorm_sp[1]*delta_z)
            ptraj_point = (self.ptraj_sp[0] + self.pnorm_sp[0]*delta_z, self.ptraj_sp[1] + self.pnorm_sp[1]*delta_z)
            # Additionally, calculate recoil angle (angle of pnorm_sp):
            recoilangle = self.enorm_sp[2] / np.sqrt(self.enorm_sp[0]**2 + self.enorm_sp[1]**2 + self.enorm_sp[2]**2) if self.enorm_sp[2] > 0 else 0
            recoil_p = np.sqrt(self.enorm_sp[0]**2 + self.enorm_sp[1]**2 + self.enorm_sp[2]**2)
            ir = -1
            #if recoilangle==-1 or recoil_p==-1:  ir = 1  # Not used for now
            if recoilangle < 10 and recoil_p < 500:
                ir = 1
            elif recoilangle < 10 and recoil_p >= 500:
                ir = 2
            elif recoilangle <= 20:
                ir = 3
            else:
                ir = 4
            # Determine what regions the hit falls into:
            insideElectronRadius = np.sqrt((etraj_point[0] - x[j])**2 + \
                    (etraj_point[1] - y[j])**2) < 1.0 * radius_68[ir][layer_index]
            insidePhotonRadius   = np.sqrt((ptraj_point[0] - x[j])**2 + \
                    (ptraj_point[1] - y[j])**2) < 1.0 * radius_68[ir][layer_index]
            # NEW:  If an SP electron hit is missing, place all hits in the event into the "other" region
            # 3-region:
            if self.enorm_sp[0] == 0 and self.enorm_sp[1] == 0:
                insideElectronRadius = False
                insidePhotonRadius   = False
            
            regions = []  # Regions hit falls inside
            if self.nRegions == 1:
                regions.append(0)
            elif self.nRegions == 2:
                if insideElectronRadius:
                    regions.append(0)
                else:
                    regions.append(1)
            elif self.nRegions == 3:
                if insideElectronRadius:
                    regions.append(0)
                if insidePhotonRadius:
                    regions.append(1)
                if not insideElectronRadius and not insidePhotonRadius:
                    regions.append(2)
            # Add to each region (need multiple in case inside electron and photon in 3-region)
            for r in range(self.nRegions):
                if r in regions:
                    x_[r][j] = x[j] - etraj_point[0]  # Store relative ot xy distance from trajectory
                    y_[r][j] = y[j] - etraj_point[1]
                    z_[r][j] = z[j] - self._layerZs[0]  # Defined relative to the ecal face
                    layer_id_[r][j] = layer_id[j]
                    log_energy_[r][j] = np.log(energy[j]) if energy[j] > 0 else -1  # Note:  E<1 is very uncommon, so -1 is okay to round to.


        var_dict = {'x_':x_, 'y_':y_, 'z_':z_,
                    'layer_id_':layer_id_,
                    'log_energy_':log_energy_,
                   }

        # Lastly, load obs_dict:
        o_dict = {}
        for branch in self.obs_branches:
            o_leaf = self.ttree.GetLeaf(branch)
            o_arr = np.array([o_leaf.GetValue(i) for i in range(o_leaf.GetLen())], dtype='float32')
            o_dict[branch] = o_arr

        return var_dict, o_dict



        
    # NOTE/WARNING:  After use, obs_dict will consist of np arrays, not lists, and cannot be appended to.
    # Shouldn't be an issue--should never need to call get() again after calling get_obs_data.
    # Could just return a new array instead, ofc, but unnecessary+eats up ram.
    def get_obs_data(self):
        for branch in self.obs_branches:
            if self.obs_dict[branch] is list:
                self.obs_dict[branch] = np.concatenate(self.obs_dict[branch])
        return self.obs_dict


    def _load_cellMap(self, version='v12'):
        self._cellMap = {}
        for i, x, y in np.loadtxt('data/%s/cellmodule.txt' % version):
            self._cellMap[i] = (x, y)
        self._layerZs = np.loadtxt('data/%s/layer.txt' % version)
        print("Loaded detector info")

    def _parse_cid(self, cid):  # Retooled for v12
        # For id details, see (?):  DetDescr/src/EcalID.cxx
        # Flatten arrays to 1D numpy arrays so zip, map will work
        # NOTE:  Now input arrays are 1D numpy arrays!  Much simpler.
        cell   = (cid >> 0)  & 0xFFF #(awkward.to_numpy(awkward.flatten(cid)) >> 0)  & 0xFFF
        module = (cid >> 12) & 0x1F  #(awkward.to_numpy(awkward.flatten(cid)) >> 12) & 0x1F
        layer  = (cid >> 17) & 0x3F  #(awkward.to_numpy(awkward.flatten(cid)) >> 17) & 0x3F
        
        mcid = 10 * cell + module
        x, y = zip(*map(self._cellMap.__getitem__, mcid))
        z = list(map(self._layerZs.__getitem__, layer))

        return (x, y, z), layer




class _SimpleCustomBatch:

    def __init__(self, data, min_nodes=None):
        pts, fts, labels = list(zip(*data))
        self.coordinates = torch.tensor(pts)
        self.features = torch.tensor(fts)
        self.label = torch.tensor(labels)

    def pin_memory(self):
        self.coordinates = self.coordinates.pin_memory()
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch):
    return _SimpleCustomBatch(batch)
