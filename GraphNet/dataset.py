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
executor = concurrent.futures.ThreadPoolExecutor(12)

MAX_NUM_ECAL_HITS = 40 #50 NOTE:  MODIFIED
# NEW:  LayerZ data (may be outdated)

layerZs = [223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,
        266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,
        322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,
        375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,
        448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125]

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


def _pad(a, pad_value=0):
    return a.pad(MAX_NUM_ECAL_HITS, clip=True).fillna(0).regular()


class ECalHitsDataset(Dataset):

    def __init__(self, siglist, bkglist, load_range=(0, 1), apply_preselection=True, ignore_evt_limits=False, obs_branches=[], coord_ref=None, detector_version='v9'):
        super(ECalHitsDataset, self).__init__()
        # first load cell map
        self._load_cellMap(version=detector_version)
        self._id_branch = 'EcalRecHits_sim.id_'
        self._energy_branch = 'EcalRecHits_sim.energy_'
        if detector_version == 'v9':
            self._id_branch = 'ecalDigis_recon.id_'
            self._energy_branch = 'ecalDigis_recon.energy_'
        self._branches = [self._id_branch, self._energy_branch]

        self.extra_labels = []
        self.presel_eff = {}
        self.var_data = {}
        self.obs_data = {k:[] for k in obs_branches}

        print('Using coord_ref=%s' % coord_ref)
        def _load_coord_ref(t, table):
            # Find recoil electron (approx)
            el = (t['EcalScoringPlaneHits_sim.pdgID_'].array() == 11) * \
                 (t['EcalScoringPlaneHits_sim.layerID_'].array() == 1) * \
                 (t['EcalScoringPlaneHits_sim.pz_'].array() > 0)
            # Note:  pad() below ensures that only one SP electron is used if there's multiple (I believe)

            etraj_x_sp = t['EcalScoringPlaneHits_sim.x_'].array()[el].pad(1, clip=True).fillna(0).flatten()  #Arr of floats.  [0][0] fails.
            etraj_y_sp = t['EcalScoringPlaneHits_sim.y_'].array()[el].pad(1, clip=True).fillna(0).flatten()
            etraj_z_sp = t['EcalScoringPlaneHits_sim.z_'].array()[el].pad(1, clip=True).fillna(0).flatten()
            etraj_px_sp = t['EcalScoringPlaneHits_sim.px_'].array()[el].pad(1, clip=True).fillna(0).flatten()
            etraj_py_sp = t['EcalScoringPlaneHits_sim.py_'].array()[el].pad(1, clip=True).fillna(0).flatten()
            etraj_pz_sp = t['EcalScoringPlaneHits_sim.pz_'].array()[el].pad(1, clip=True).fillna(0).flatten()

            # Create vectors holding the electron/photon momenta so the trajectory projections can be found later
            # Set xtraj_p_norm relative to z=1 to make projecting easier:
            E_beam = 4000.0  # In GeV
            etraj_p_norm = []
            for i in range(len(etraj_pz_sp)):
                if etraj_pz_sp[i] != 0:
                    etraj_p_norm.append((etraj_px_sp[i]/etraj_pz_sp[i], etraj_py_sp[i]/etraj_pz_sp[i], 1.0))
                else:
                    etraj_p_norm.append((0,0,0))

            ptraj_p_norm = []
            for i in range(len(etraj_pz_sp)):
                if etraj_pz_sp[i] != 0:
                    ptraj_p_norm.append((-etraj_px_sp[i]/(E_beam - etraj_pz_sp[i]), -etraj_py_sp[i]/(E_beam - etraj_pz_sp[i]), 1.0))
                else:
                    ptraj_p_norm.append((0,0,0))



            # Calc z relative to ecal face

            etraj_ref = np.zeros((len(etraj_p_norm), 2, 3))  # Note the 2:  Only storing start and pvec_norm
            ptraj_ref = np.zeros((len(etraj_p_norm), 2, 3))
            # Format is [event#] x [start of traj/p_norm] x [etraj_xyz]
            for i in range(len(etraj_p_norm)):
                etraj_ref[i][0][0] = etraj_x_sp[i]
                etraj_ref[i][0][1] = etraj_y_sp[i]
                etraj_ref[i][0][2] = etraj_z_sp[i]
                etraj_ref[i][1][0] = etraj_p_norm[i][0]
                etraj_ref[i][1][1] = etraj_p_norm[i][1]
                etraj_ref[i][1][2] = etraj_p_norm[i][2]
                ptraj_ref[i][0][0] = etraj_x_sp[i]
                ptraj_ref[i][0][1] = etraj_y_sp[i]
                ptraj_ref[i][0][2] = etraj_z_sp[i]
                ptraj_ref[i][1][0] = ptraj_p_norm[i][0]
                ptraj_ref[i][1][1] = ptraj_p_norm[i][1]
                ptraj_ref[i][1][2] = ptraj_p_norm[i][2]

            table['etraj_ref'] = etraj_ref
            table['ptraj_ref'] = ptraj_ref
            #print("Finished loading coord ref")


        def _load_recoil_pt(t, table):  # NOTE:  Is this ever used?
            if len(obs_branches):
                el = (t['TargetScoringPlaneHits_sim.pdgID_'].array() == 11) * \
                     (t['TargetScoringPlaneHits_sim.layerID_'].array() == 2) * \
                     (t['TargetScoringPlaneHits_sim.pz_'].array() > 0)
                table['TargetSPRecoilE_pt'] = np.sqrt(t['TargetScoringPlaneHits_sim.px_'].array()[el] ** 2 + t['TargetScoringPlaneHits_sim.py_'].array()[el] ** 2).pad(1, clip=True).fillna(-999).flatten()


        def _read_file(table):
            # load data from one file
            start, stop = [int(x * len(table[self._branches[0]])) for x in load_range]
            for k in table:
                table[k] = table[k][start:stop]
            n_inclusive = len(table[self._branches[0]])  # before preselection

            if apply_preselection:
                pos_pass_presel = (table[self._energy_branch] > 0).sum() < MAX_NUM_ECAL_HITS
                for k in table:
                    table[k] = table[k][pos_pass_presel]
            n_selected = len(table[self._branches[0]])  # after preselection

            for k in table:
                if isinstance(table[k], awkward.array.objects.ObjectArray):
                    table[k] = awkward.JaggedArray.fromiter(table[k]).flatten()

            eid = table[self._id_branch]
            energy = table[self._energy_branch]
            pos = (energy > 0)
            eid = eid[pos]
            energy = energy[pos]
            
            (x, y, z), layer_id = self._parse_cid(eid)  # layer_id > 0, so can use layer_id-1 to index e/ptraj_ref

            # Now, work with table['etraj_ref'] and table['ptraj_ref'].
            # Create lists:  x/y/z_e, p
            # For each event, look through all hits.
            # - Determine whether hit falls inside either the e or p RoCs
            # - If so, fill corresp xyzlayer, energy, eid lists...
            x_e =           np.zeros((len(x), MAX_NUM_ECAL_HITS))  # In theory, can lower size of 2nd dimension...
            y_e =           np.zeros((len(x), MAX_NUM_ECAL_HITS))
            z_e =           np.zeros((len(x), MAX_NUM_ECAL_HITS))
            eid_e =         np.zeros((len(x), MAX_NUM_ECAL_HITS))
            log_energy_e =  np.zeros((len(x), MAX_NUM_ECAL_HITS))
            layer_id_e =    np.zeros((len(x), MAX_NUM_ECAL_HITS))
            x_p =           np.zeros((len(x), MAX_NUM_ECAL_HITS))
            y_p =           np.zeros((len(x), MAX_NUM_ECAL_HITS))
            z_p =           np.zeros((len(x), MAX_NUM_ECAL_HITS))
            eid_p =         np.zeros((len(x), MAX_NUM_ECAL_HITS))
            log_energy_p =  np.zeros((len(x), MAX_NUM_ECAL_HITS))
            layer_id_p =    np.zeros((len(x), MAX_NUM_ECAL_HITS))

            for i in range(len(x)):  # For every event...
                etraj_sp = table['etraj_ref'][i][0]  # e- location at scoring plane (approximate)
                enorm_sp = table['etraj_ref'][i][1]  # normalized (dz=1) momentum = direction of trajectory
                ptraj_sp = table['ptraj_ref'][i][0]
                pnorm_sp = table['ptraj_ref'][i][1]
                for j in range(len(x[i])):  #range(MAX_NUM_ECAL_HITS):  # For every hit...
                    layer_index = int(layer_id[i][j])
                    # Calculate xy for projected trajectory in same layer
                    delta_z = layerZs[layer_index] - etraj_sp[0]
                    etraj_point = (etraj_sp[0] + enorm_sp[0]*delta_z, etraj_sp[1] + enorm_sp[0]*delta_z)
                    ptraj_point = (ptraj_sp[0] + pnorm_sp[0]*delta_z, ptraj_sp[1] + pnorm_sp[0]*delta_z)
                    # Additionally, calculate recoil angle (angle of pnorm_sp):
                    recoilangle = enorm_sp[2] / np.sqrt(enorm_sp[0]**2 + enorm_sp[1]**2 + enorm_sp[2]**2)
                    recoil_p = np.sqrt(enorm_sp[0]**2 + enorm_sp[1]**2 + enorm_sp[2]**2)
                    ir = -1
                    #if recoilangle==-1 or recoil_p==-1:  ir = 1  # Not used for now
                    if recoilangle<10 and recoil_p<500:
                        ir = 1
                    elif recoilangle<10 and recoil_p >= 500:
                        ir = 2
                    elif recoilangle<=20:
                        ir = 3
                    else:
                        ir = 4
                    if np.sqrt((etraj_point[0] - x[i][j])**2 + (etraj_point[1] - y[i][j])**2) < 2.0 * radius_68[ir][layer_index]:  #If inside e- RoC:
                        x_e[i][j] = x[i][j] - etraj_point[0]  # Store coordinates relative to the xy distance from the trajectory
                        y_e[i][j] = y[i][j] - etraj_point[1]
                        z_e[i][j] = z[i][j] - layerZs[0]  # Defined relative to the ecal face
                        eid_e[i][j] = eid[i][j]
                        log_energy_e[i][j] = np.log(energy[i][j]) if energy[i][j] > 0 else 0
                        layer_id_e[i][j] = layer_id[i][j]
                        #ehits += 1

                    if np.sqrt((ptraj_point[0] - x[i][j])**2 + (ptraj_point[1] - y[i][j])**2) < 2.0 * radius_68[ir][layer_index]:  #If inside photon RoC:
                        x_p[i][j] = x[i][j] - ptraj_point[0]  # Store coordinates relative to the xy distance from the trajectory
                        y_p[i][j] = y[i][j] - ptraj_point[1]
                        z_p[i][j] = z[i][j] - layerZs[0]  # Defined relative to the ecal face
                        eid_p[i][j] = eid[i][j]
                        log_energy_p[i][j] = np.log(energy[i][j]) if energy[i][j] > 0 else 0
                        layer_id_p[i][j] = layer_id[i][j]

            var_dict = {'id_e':eid_e, 'log_energy_e':log_energy_e,
                        'x_e':x_e, 'y_e':y_e, 'z_e':z_e, 'layer_id_e':layer_id_e,
                        'etraj_ref':np.array(table['etraj_ref']),
                        'id_p':eid_p, 'log_energy_p':log_energy_p,
                        'x_p':x_e, 'y_p':y_p, 'z_p':z_p, 'layer_id_p':layer_id_p,
                        'ptraj_ref':np.array(table['ptraj_ref'])}

            obs_dict = {k: table[k] for k in obs_branches}

            return (n_inclusive, n_selected), var_dict, obs_dict

        def _load_dataset(filelist, name):
            # load data from all files in the siglist or bkglist
            n_sum = 0
            for extra_label in filelist:
                filepath, max_event = filelist[extra_label]
                if len(glob.glob(filepath)) == 0:
                    print('No matches for filepath %s: %s, skipping...' % (extra_label, filepath))
                    return
                if ignore_evt_limits:
                    max_event = -1
                n_total_inclusive = 0
                n_total_selected = 0
                var_dict = {}
                obs_dict = {k:[] for k in obs_branches}
                # NEW:  Dictionary storing particle data for e/p trajectory
                # Want position, momentum of e- hit; calc photon info from it
                spHit_dict = {}
                print('Start loading dataset %s (%s)' % (filepath, name))

                with tqdm.tqdm(glob.glob(filepath)) as tq:
                    for fp in tq:
                        t = uproot.open(fp)['LDMX_Events']
                        if len(t.keys()) == 0:
#                             print('... ignoring empty file %s' % fp)
                            continue
                        load_branches = [k for k in self._branches + obs_branches if '.' in k and k[-1] == '_']
                        table = t.arrays(load_branches, namedecode='utf-8', executor=executor)
                        _load_coord_ref(t, table)
                        _load_recoil_pt(t, table)
                        (n_inc, n_sel), v_d, o_d = _read_file(table)
                        n_total_inclusive += n_inc
                        n_total_selected += n_sel
                        for k in v_d:
                            if k in var_dict:
                                var_dict[k].append(v_d[k])
                            else:
                                var_dict[k] = [v_d[k]]
                        for k in obs_dict:
                            obs_dict[k].append(o_d[k])
                        if max_event > 0 and n_total_selected >= max_event:
                            break

                # calc preselection eff before dropping events more than `max_event`
                self.presel_eff[extra_label] = float(n_total_selected) / n_total_inclusive
                # now we concat the arrays and remove the extra events if needed
                n_total_loaded = None
                upper = None
                if max_event > 0 and max_event < n_total_selected:
                    upper = max_event - n_total_selected
                for k in var_dict:
                    var_dict[k] = _concat(var_dict[k])[:upper]
                    if n_total_loaded is None:
                        n_total_loaded = len(var_dict[k])
                    else:
                        assert(n_total_loaded == len(var_dict[k]))
                for k in obs_dict:
                    obs_dict[k] = _concat(obs_dict[k])[:upper]
                    assert(n_total_loaded == len(obs_dict[k]))
                print('Total %d events, selected %d events, finally loaded %d events.' % (n_total_inclusive, n_total_selected, n_total_loaded))

                self.extra_labels.append(extra_label * np.ones(n_total_loaded, dtype='int32'))
                for k in var_dict:
                    if k in self.var_data:
                        self.var_data[k].append(var_dict[k])
                    else:
                        self.var_data[k] = [var_dict[k]]
                for k in obs_branches:
                    self.obs_data[k].append(obs_dict[k])
                n_sum += n_total_loaded
            return n_sum

        nsig = _load_dataset(siglist, 'sig')
        nbkg = _load_dataset(bkglist, 'bkg')
        # label for training
        self.label = np.zeros(nsig + nbkg, dtype='float32')
        self.label[:nsig] = 1

        self.extra_labels = np.concatenate(self.extra_labels)
        for k in self.var_data:
            self.var_data[k] = _concat(self.var_data[k])
        for k in obs_branches:
            self.obs_data[k] = _concat(self.obs_data[k])

        # training features
        # Redone for new e/p features
        # NOTE:  Used to be _pad(a), but the padding is already done implicitly.
        xyz_e = [a for a in (self.var_data['x_e'], self.var_data['y_e'], self.var_data['z_e'])]
        layer_id_e = self.var_data['layer_id_e']
        log_e_e = self.var_data['log_energy_e']
        xyz_p = [a for a in (self.var_data['x_p'], self.var_data['y_p'], self.var_data['z_p'])]
        layer_id_p = self.var_data['layer_id_p']
        log_e_p = self.var_data['log_energy_p']
        # Merge into a singe features array:
        e_coords = np.stack(xyz_e, axis=1)
        p_coords = np.stack(xyz_p, axis=1)
        self.coordinates = np.stack([e_coords, p_coords], axis=1).astype('float32')
        e_features = np.stack(xyz_e + [layer_id_e, log_e_e], axis=1)
        p_features = np.stack(xyz_p + [layer_id_p, log_e_p], axis=1)
        self.features = np.stack([e_features, p_features], axis=1).astype('float32')

        assert(len(self.coordinates) == len(self.label))
        assert(len(self.features) == len(self.label))

    def _load_cellMap(self, version='v9'):
        self._cellMap = {}
        for i, x, y in np.loadtxt('data/%s/cellmodule.txt' % version):
            self._cellMap[i] = (x, y)
        self._layerZs = np.loadtxt('data/%s/layer.txt' % version)

    def _parse_cid(self, cid):
        cell = (cid.content & 0xFFFF8000) >> 15
        module = (cid.content & 0x7000) >> 12
        layer = (cid.content & 0xFF0) >> 4
        mcid = 10 * cell + module
        x, y = zip(*map(self._cellMap.__getitem__, mcid))
        z = list(map(self._layerZs.__getitem__, layer))
        x = cid.copy(content=np.array(x, dtype='float32'))
        y = cid.copy(content=np.array(y, dtype='float32'))
        z = cid.copy(content=np.array(z, dtype='float32'))
        layer_id = cid.copy(content=np.array(layer, dtype='float32'))
        return (x, y, z), layer_id

    @property
    def num_features(self):
        #return self.features.shape[1]
        return self.features.shape[2]  # Modified

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):  # NOTE:  This now returns e/p data.  May need modification.
        pts = self.coordinates[i]
        fts = self.features[i]
        y = self.label[i]
        return pts, fts, y


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
