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

# Note:  I suggest downloading+importing the psutil module if you need to monitor RAM/GPU usage.

# Create ThreadPoolExecutor to accelerate loading with uproot
# Note:  Number of threads has not been optimized for lazy-loading; it may be worth experimenting w/ lower values.
#executor = concurrent.futures.ThreadPoolExecutor(12)

torch.set_default_dtype(torch.float32)

# Should match value in the preselection.  Determines size of ParticleNet position arrays.
MAX_NUM_ECAL_HITS = 50 #60  #110  # NOW REDUCED!

# NEW: Radius of containment data
# Note:  Should still be valid for 2e ParticleNet unless the shower shape has changed
radius_beam_68 = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]

#from 1e
#radius_recoil_68_p_0_500_theta_0_10 = [4.045666158618167, 4.086393662224346, 4.359141107602775, 4.666549994726691, 5.8569181911416015, 6.559716356124256, 8.686967529043072, 10.063482736354674, 13.053528344041274, 14.883496407943747, 18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406, 28.329169392203216, 29.468032123356345, 34.03271241527079, 35.03747443690781, 38.50748727211848, 39.41576583301171, 42.63622296033334, 45.41123601592071, 48.618139095742876, 48.11801717451056, 53.220539860213655, 58.87753380915155, 66.31550881539764, 72.94685877928593, 85.95506228335348, 89.20607201266672, 93.34370253818409, 96.59471226749734, 100.7323427930147, 103.98335252232795]

#radius_recoil_68_p_500_1500_theta_0_10 = [4.081926458777424, 4.099431732299409, 4.262428482867968, 4.362017581473145, 4.831341579961153, 4.998346041276382, 6.2633736512415705, 6.588371889265881, 8.359969947444522, 9.015085558044309, 11.262722588206483, 12.250305471269183, 15.00547660437276, 16.187264014640103, 19.573764900578503, 20.68072032434797, 24.13797140783321, 25.62942209291236, 29.027596514735617, 30.215039667389316, 33.929540248019585, 36.12911729771914, 39.184563500620946, 42.02062468386282, 46.972125628650204, 47.78214816041894, 55.88428562462974, 59.15520134927332, 63.31816666637158, 66.58908239101515, 70.75204770811342, 74.022963432757, 78.18592874985525, 81.45684447449884]

#radius_recoil_68_theta_10_20 = [4.0251896715647115, 4.071661598616328, 4.357690094817289, 4.760224640141712, 6.002480766325418, 6.667318981016246, 8.652513285172342, 9.72379373302137, 12.479492693251478, 14.058548828317289, 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302, 31.768946746958296, 35.79247330197688, 37.27810357669942, 41.657281051476545, 42.628141392692626, 47.94208483539388, 49.9289473559796, 54.604030254423975, 53.958762417361655, 53.03339560920388, 57.026277390001425, 62.10810455035879, 66.10098633115634, 71.1828134915137, 75.17569527231124, 80.25752243266861, 84.25040421346615, 89.33223137382352, 93.32511315462106]

#radius_recoil_68_theta_20_end = [4.0754238481177705, 4.193693485630508, 5.14209420056253, 6.114996249971468, 7.7376807326481645, 8.551663213602291, 11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604, 25.04088272634407, 28.853696411302344, 34.72538105333071, 40.21218694947545, 46.07344239520299, 50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047, 74.82378572939959, 89.4528387422834, 93.18228303096758, 92.51751129204555, 98.80228884380018, 111.17537347472128, 120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185, 177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696]

#radius_68 = [radius_beam_68,radius_recoil_68_p_0_500_theta_0_10, radius_recoil_68_p_500_1500_theta_0_10,radius_recoil_68_theta_10_20,radius_recoil_68_theta_20_end]

#NEW: RoC bins for v14 1e
radius68_thetalt10 = [  10.12233413, 9.921772, 11.38255086, 11.67991867, 13.14337347, 
                                13.17120624, 16.80994665, 17.83787244, 22.44684374, 23.74239886, 
                                28.60564083, 30.27889678, 34.86404888, 36.39009394, 41.29309474, 
                                43.34682279, 48.55982854, 50.80565589, 55.29496257, 57.92737879, 
                                60.64828824, 65.51760517, 68.26709803, 76.32877518, 84.61219467, 
                                103.3649691, 111.1692293, 119.2928089, 127.7357081, 136.4979268, 
                                145.579465, 154.9803228, 164.7005, 174.7399968 ]

radius68_theta10to15 = [ 10.82307758, 11.17850518, 16.2185281, 18.62488713, 22.63408229, 
                                24.71769042, 30.11217538, 32.69939046, 37.99753196, 40.81619543, 
                                45.89054775, 49.03066318, 54.00440948, 59.31733555, 63.40789682, 
                                64.77580021, 73.00113678, 73.25561396, 78.8914776, 86.73962133, 
                                97.05926327, 96.6932739, 111.6226151, 106.5960265, 109.477541, 
                                144.2545942, 153.7581461, 163.5921179, 173.7565094, 184.2513208, 
                                195.076552, 206.2322029, 217.7182737, 229.5347642 ]

radius68_theta15to20 = [ 12.79450901, 13.02698578, 21.27450933, 25.66008312, 31.78592103, 
                                35.99689874, 44.37101115, 48.82709363, 55.05972458, 59.68948687, 
                                65.39866214, 70.59280337, 76.06007787, 82.22695257, 87.50371819, 
                                90.60099831, 96.34848268, 101.4928478, 106.7157092, 105.0540604, 
                                110.0653355, 148.3428736, 133.1449443, 146.997265, 173.3954389, 
                                185.1307166, 196.1408667, 207.4772729, 219.1399351, 231.1288534, 
                                243.4440277, 256.085458, 269.0531444, 282.3470868 ]

radius68_theta20to30 = [ 14.16989595, 15.4488322, 28.31044668, 37.54285657, 48.57288885, 
                                57.04243339, 68.99836079, 75.33388728, 85.00572867, 91.52574074, 
                                102.5044698, 106.5315986, 116.2341378, 127.1121442, 133.8866375, 
                                144.5121759, 162.1726963, 160.2986579, 171.386638, 182.5653112, 
                                205.5853241, 196.3113071, 200.5907513, 228.7275694, 234.0298491, 
                                253.7990618, 263.6872702, 273.5754785, 283.4636869, 293.3518953, 
                                303.2401036, 313.128312, 323.0165203, 332.9047287 ]

radius68_theta30to60 = [ 22.50983127, 26.44537503, 58.24642887, 90.59076279, 130.0592014, 
                                157.4611392, 184.2187293, 202.6994588, 225.3488816, 243.3454167, 
                                269.2456428, 280.6119298, 303.8591523, 322.0522722, 335.1780181, 
                                350.3398234, 353.7763544, 373.9942362, 382.9453608, 401.9703438, 
                                441.6281859, 432.5241826, 455.2878243, 492.2888656, 502.6653722, 
                                519.9101788, 539.1604349, 558.410691, 577.6609471, 596.9112032, 
                                616.1614593, 635.4117154, 654.6619715, 673.9122276 ]

radius_68 = [radius_beam_68, radius68_thetalt10, radius68_theta10to15, radius68_theta15to20, radius68_theta20to30, radius68_theta30to60]



def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array()
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)



class ECalHitsDataset(Dataset):

    def __init__(self, siglist, bkglist, load_range=(0, 1), obs_branches=[], coord_ref=None, detector_version='v14', nRegions=1, regSizes=None):
        super(ECalHitsDataset, self).__init__()
        print("Initializing EcalHitsDataset")
        # load cell map (for calculating xyz+layer from hit IDs)
        self._load_cellMap(version=detector_version)
        self.detector_version = detector_version
        # Specify the two branches necessary for all ParticleNet features:  (xyz+layer+energy)
        self._id_branch = 'id_rec_'  # load from 'EcalRecHits_v12.id_'
        self._pos_branch = '{}pos_rec_'
        self._energy_branch = 'energy_rec_'  # load from 'EcalRecHits_v12.energy_'
        if detector_version == 'v12':
            self._branches = [self._id_branch, self._energy_branch]
        else:
            self._branches = [self._id_branch] + [self._pos_branch.format(v) for v in ['x', 'y', 'z']]

        self.obs_branches = obs_branches
        # NOTE:  Need to explicitly keep track of and save all obs_dict data!  Fortunately, order doesn't matter.
        self.obs_dict = {br:[] for br in self.obs_branches}
        # Also need to keep track of events that have been loaded into obs_dict, to ensure no duplicates
        # Just store a list of the event numbers
        self.loaded_events = []

        self.coord_ref = coord_ref
        assert(detector_version != 'v9')  # v9 compatibility would be nontrivial to add, and is probably unnecessary

        self.nRegions = nRegions

        # General approach to init():
        # - Input events have all been preselected.
        # - All event data is stored in a "simple" root tree with no sub-branches
        # - Need to create a mapping:  event number -> returns sig/bkg, root file, and evt number within that file
        #    - [[mass, filename, i_file], ...]
        #    - Get element i of the list whenever PN requests an event, and find the event location based on that info

        self.event_list = []  # == [[mass, filename, i_file], ...]
        self.extra_labels = []  # mass in MeV if sig, 0 if bkg; converted to np array below
        print("Filling event_list")
        # fill event_list to make event access easy
        filelist = {}
        for label, fname in bkglist.items():
            filelist[label] = fname
        for label, fname in siglist.items():
            filelist[label] = fname
        print("Using filelist=", filelist)

        for extra_label in filelist:  # For each mass:
            filepath, max_events = filelist[extra_label]
            if max_events == -1:
                max_events = 1e8  # Unrealistically large so it never constrains the results
            num_loaded_events = 0  # Number of events so far for this mass
            #print("   Filling for m={}".format(extra_label))
            for fp in glob.glob(filepath):
                # For each file, check the number of events, then add to event_list accordingly
                if num_loaded_events == max_events:  break
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
                 #print("      {} events in file, {} total for current mass".format(f_event, num_loaded_events))
            print("   Loaded m={}:  using {} events".format(extra_label, num_loaded_events))

        self.extra_labels = np.array(self.extra_labels)
        self.label = np.array([1 if l > 0 else 0 for l in self.extra_labels])  # 1 if sig, 0 if bkg

        if regSizes:  assert(nRegions == len(regSizes))
        self.regSizes = regSizes

        print("Initialization finished.")



    @property
    def num_features(self):
        # Hard-coded; not worried about generalizing atm
        return 5

    def __len__(self):
        return len(self.event_list)


    def __getitem__(self, i):
        # On-demand, read event file_index from filename and process it
        # By assumption, events have already been preselected!
        # returns:  label (sig/bkg), coords (xyz), features (xyzLE)

        # Get info on event location from event_list:
        label, filename, file_index = self.event_list[i]

        self.obs_data = {k:[] for k in self.obs_branches}

        tfile = r.TFile(filename)  # NOTE:  This is the bottleneck!  (Slow, limits performance)
        # ...could theoretically speed up by maintaining a list of all TFiles in use
        self.ttree = tfile.Get('skimmed_events')
        # Prepare to load data from event [file_index]:
        self.ttree.GetEntry(file_index)
        # load_sp_data():  Need to get info from TargetScoringPlanes to compute projected electron/photon
        # trajectories -> decide what region each event goes into
        self._load_sp_data()
        # read_event():  Fills var_dict and obs_dict
        # obs_dict contains obs_branches info loaded from train.py; saved for plotting
        # var_dict contains feature info necessary for PN:  x, y, z, layer, log(E); multi-dimensional
        # if other regions included
        var_data, obs_data = self._read_event()
        # o_d data must be saved for plotting/etc.  Ensure data from that event hasn't been recorded first:
        if not i in self.loaded_events:
            for branch in self.obs_branches:
                self.obs_dict[branch].append(obs_data[branch])
            self.loaded_events.append(i)

        # create features and coordinates:
        # NOTE:  Always 3-dimensional!  [[a, b...]] for 1-region PN
        coordinates = np.stack((var_data['x_'], var_data['y_'], var_data['z_']), axis=1)
        features    = np.stack((var_data['x_'], var_data['y_'], var_data['z_'],
                                var_data['layer_id_'], var_data['log_energy_']), axis=1)
        return coordinates, features, label


    # _load_sp_data():  calculate the projected/predicted electron/photon trajectories from SPHits data
    def _load_sp_data(self):
        pdgID_leaf = self.ttree.GetLeaf('pdgID_')
        z_leaf     = self.ttree.GetLeaf('z_')
        pz_leaf    = self.ttree.GetLeaf('pz_')
        pdgID_ = [int(pdgID_leaf.GetValue(i)) for i in range(pdgID_leaf.GetLen())]
        z_     = [z_leaf.GetValue(i)          for i in range(z_leaf.GetLen())    ]
        pz_    = [pz_leaf.GetValue(i)         for i in range(pz_leaf.GetLen())   ]
        pdgID_leaf_tsp = self.ttree.GetLeaf('pdgID_tsp_')
        z_leaf_tsp     = self.ttree.GetLeaf('z_tsp_')
        pz_leaf_tsp    = self.ttree.GetLeaf('pz_tsp_')
        #print(pdgID_leaf, pdgID_leaf_tsp)
        #print(pdgID_leaf_tsp.GetLen())
        pdgID_tsp_ = [int(pdgID_leaf_tsp.GetValue(i)) for i in range(pdgID_leaf_tsp.GetLen())]
        z_tsp_     = [z_leaf_tsp.GetValue(i)          for i in range(z_leaf_tsp.GetLen())    ]
        pz_tsp_    = [pz_leaf_tsp.GetValue(i)         for i in range(pz_leaf_tsp.GetLen())   ]
        #el_ = 0  # SP index of recoil electron
        #pmax = 0  # Max pz
        #max_index = 0

        # TODO WIP WIP:  Must revise this!
        #assert(False)

        # first, find the recoil electron at the target (for computing photon momentum):
        r_tsp = 0
        pmax_tsp = 0
        for j in range(pdgID_leaf_tsp.GetLen()):
            if pdgID_tsp_[j] == 11 and z_tsp_[j] > 0.176 and z_tsp_[j] < 0.178 and pz_tsp_[j] > pmax_tsp:
                r_tsp = j
                pmax_tsp = pz_tsp_[j]

        # Find the recoil electron at the ecal SP:
        r_ecal = 0
        pmax_ecal = 0
        for j in range(pdgID_leaf.GetLen()):
            if pdgID_[j] == 11 and z_[j] > 239 and z_[j] < 240 and pz_[j] > pmax_ecal:
                pmax_ecal = pz_[j]
                r_ecal = j
        has_e = pmax_ecal != 0  # Check whether event has a SP electron
        
        etraj_ecal = {v+'_':  self.ttree.GetLeaf(v+'_').GetValue(r_ecal)     for v in ['x', 'y', 'z', 'px', 'py', 'pz']}
        etraj_tsp  = {v+'_':  self.ttree.GetLeaf(v+'_tsp_').GetValue(r_tsp) for v in ['x', 'y', 'z', 'px', 'py', 'pz']}
        #etraj_x_sp  = self.ttree.GetLeaf('x_').GetValue(el_)
        #etraj_y_sp  = self.ttree.GetLeaf('y_').GetValue(el_)
        #etraj_z_sp  = self.ttree.GetLeaf('z_').GetValue(el_)
        #etraj_px_sp = self.ttree.GetLeaf('px_').GetValue(el_)
        #etraj_py_sp = self.ttree.GetLeaf('py_').GetValue(el_)
        #etraj_pz_sp = self.ttree.GetLeaf('pz_').GetValue(el_)

        #self.etraj_sp = np.array((etraj_x_sp, etraj_y_sp, etraj_z_sp))

        # Create vectors holding the electron/photon momenta so the trajectory projections can be found later
        # Set xtraj_p_norm relative to z=1 to make projecting easier:
        E_beam = 4000.0  # In MeV
        target_dist = 240.5 # distance from ecal to target, mm
        
        # Compute and store trajectories:
        if has_e:
            self.etraj_sp = np.array((etraj_ecal['x_'], etraj_ecal['y_'], etraj_ecal['z_']))
            self.enorm_sp = np.array((etraj_ecal['px_']/etraj_ecal['pz_'], etraj_ecal['py_']/etraj_ecal['pz_'], 1.0))
            self.pnorm_sp = np.array((-etraj_tsp['px_']/(E_beam - etraj_tsp['pz_']), 
                                      -etraj_tsp['py_']/(E_beam - etraj_tsp['pz_']),
                                      1.0))
            self.ptraj_sp = np.array((etraj_tsp['x_'] + target_dist*self.pnorm_sp[0],
                                      etraj_tsp['y_'] + target_dist*self.pnorm_sp[1],
                                      etraj_tsp['z_'] + target_dist))

        else:
            self.etraj_sp = np.array((-999,-999,-999))
            self.enorm_sp = np.array((-999,-999,-999))
            self.pnorm_sp = np.array((-999,-999,-999))
            self.ptraj_sp = np.array((-999,-999,-999))



    def _read_event(self):
        # Read data from event and fill var_dict and obs_dict:
        # obs_dict contains obs_branches info (branches specified in train.py); saved for plotting
        # var_dict contains info necessary for PN:  x, y, z, layer, log(E); more if other regions included
        if self.detector_version == 'v12':
            eid_leaf    = self.ttree.GetLeaf(self._id_branch)
            energy_leaf = self.ttree.GetLeaf(self._energy_branch)
            eid    = np.array([eid_leaf.GetValue(i)    for i in range(eid_leaf.GetLen())   ], dtype='int')  #table[self._id_branch]
            energy = np.array([energy_leaf.GetValue(i) for i in range(energy_leaf.GetLen())], dtype='float32')  #table[self._energy_branch]
            #print("TEMP: energy len 1=", len(energy))
            pos = (energy > 0)
            eid = eid[pos]  # Gets rid of all (AND ONLY) hits with 0 energy
            energy = energy[pos]

            (x, y, z), layer_id = self._parse_cid(eid)  # layer_id > 0, so can use layer_id-1 to index e/ptraj_ref
        else:
            xyz_leaves = [self.ttree.GetLeaf(self._pos_branch.format(v)) for v in ['x', 'y', 'z']]
            (x, y, z) = [np.array([lf.GetValue(i) for i in range(lf.GetLen())], dtype='float32') for lf in xyz_leaves]
            energy_leaf = self.ttree.GetLeaf(self._energy_branch)
            energy = np.array([energy_leaf.GetValue(i) for i in range(energy_leaf.GetLen())], dtype='float32')
            layer_id = self._getlayer(z)


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

        regionIndices = [0, 0, 0]  # Indices of last hit added to feature arrays


        for j in range(len(layer_id)):  #eid_leaf.GetLen()):  # For every hit...
            # Calculate xy coord of point on projected trajectory in same layer
            delta_z = z[j] - self.etraj_sp[2]
            if self.etraj_sp[2] != -999:  # If fiducial
                etraj_point = (self.etraj_sp[0] + self.enorm_sp[0]*delta_z, self.etraj_sp[1] + self.enorm_sp[1]*delta_z)
                ptraj_point = (self.ptraj_sp[0] + self.pnorm_sp[0]*delta_z, self.ptraj_sp[1] + self.pnorm_sp[1]*delta_z)
                # Additionally, calculate recoil angle (angle of pnorm_sp):
                recoilangle = self.enorm_sp[2] / np.sqrt(self.enorm_sp[0]**2 + self.enorm_sp[1]**2 + self.enorm_sp[2]**2)
                recoil_p = np.sqrt(self.enorm_sp[0]**2 + self.enorm_sp[1]**2 + self.enorm_sp[2]**2) 
            else:
                etraj_point = self.etraj_sp  # (-999, -999, -999)
                ptraj_point = self.ptraj_sp

                recoilangle = -999
                recoil_p    = -999
            ir = -1
            #if recoilangle==-1 or recoil_p==-1:  ir = 1  # Not used for now
            # Select the class of containment radii based on trajectory angle/energy:
            if self.detector_version=='v14':
                if recoilangle < 10:
                    ir = 1
                elif 10 <= recoilangle < 15:
                    ir = 2
                elif 15 <= recoilangle < 20:
                    ir = 3
                elif 20 <= recoilangle < 30:
                    ir = 4
                else:
                    ir = 5
            else: # else version v12 or v13
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
                    (etraj_point[1] - y[j])**2) < 2.0 * radius_68[ir][layer_id[j]]
            insidePhotonRadius   = np.sqrt((ptraj_point[0] - x[j])**2 + \
                    (ptraj_point[1] - y[j])**2) < 2.0 * radius_68[ir][layer_id[j]]
            # NEW:  If an SP electron hit is missing, place all hits in the event into the "other" region
            # 3-region:
            if self.enorm_sp[2] == -999:
                insideElectronRadius = False
                insidePhotonRadius   = True
            
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
                    x_[r][j] = x[j] - etraj_point[0]  # Store relative to xy distance from trajectory
                    y_[r][j] = y[j] - etraj_point[1]
                    z_[r][j] = z[j]  # - self._layerZs[0]  # Used to be defined relative to the ecal face; changed to absolute bc of Huilin's old results
                    layer_id_[r][j] = layer_id[j]
                    if energy[j] > 0:
                        log_energy_[r][j] = np.log(energy[j]) 
                    elif energy[j] == 0:
                        log_energy_[r][j] = -2
                    else: # else E<0
                        log_energy_[r][j] = -1   # Note:  E<0 is very uncommon, so -1 is okay to round to.

        # Create and fill var_dict w/ feature information:
        var_dict = {'x_':x_, 'y_':y_, 'z_':z_,
                    'layer_id_':layer_id_,
                    'log_energy_':log_energy_,
                   }

        # Lastly, create and fill obs_dict w/ branches specified in train.py:
        o_dict = {}
        for branch in self.obs_branches:
            o_leaf = self.ttree.GetLeaf(branch)
            o_arr = np.array([o_leaf.GetValue(i) for i in range(o_leaf.GetLen())], dtype='float32')
            o_dict[branch] = o_arr

        return var_dict, o_dict



        
    # NOTE/WARNING:  After use, obs_dict will consist of np arrays, not lists, and cannot be appended to.
    # Shouldn't be an issue--should never need to call get_obs_data() before alll events have been added.
    # Could just return a new array instead, ofc, but unnecessary+eats up a lot of ram.
    def get_obs_data(self):
        for branch in self.obs_branches:
            if self.obs_dict[branch] is list:
                self.obs_dict[branch] = np.concatenate(self.obs_dict[branch])
        return self.obs_dict


    def _load_cellMap(self, version='v14'):
        self._cellMap = {}  # cellMap used for v12 only
        if version=='v12':
            for i, x, y in np.loadtxt('data/%s/cellmodule.txt' % version):
                self._cellMap[i] = (x, y)
            self._layerZs = np.loadtxt('data/%s/layer.txt' % version)
        if version=='v13' or version=='v14':
            zd = np.loadtxt('data/%s/layer.txt' % version)
            self._layerZs = {round(zd[i]):i for i in range(len(zd))}
        print("Loaded geometry info")

    def _getlayer(self, zarr):
        # Pass in multidim array of z positions, return array of layer numbers
        #print(self._layerZs)
        #print(zarr[:10])
        def roundreturn(val):
            return self._layerZs[round(val)]
        return list(map(roundreturn, zarr))  #self._layerZs.__getitem__, zarr))


    def _parse_cid(self, cid):  # Retooled for v12
        # Translate hit IDs into xyz+layer data
        # For id details, see (?):  ldmx-sw/DetDescr/src/EcalID.cxx
        
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
