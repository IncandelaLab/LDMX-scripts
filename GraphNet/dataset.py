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

MAX_NUM_ECAL_HITS = 50

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
            if coord_ref is None or (coord_ref == 'none' or coord_ref == 'ecal_centroid'):
                table['x_ref'] = np.zeros(t.numentries, dtype='float32')
                table['y_ref'] = np.zeros(t.numentries, dtype='float32')
            elif coord_ref == 'ecal_sp':
                el = (t['EcalScoringPlaneHits_sim.pdgID_'].array() == 11) * \
                     (t['EcalScoringPlaneHits_sim.layerID_'].array() == 1) * \
                     (t['EcalScoringPlaneHits_sim.pz_'].array() > 0)
                table['x_ref'] = t['EcalScoringPlaneHits_sim.x_'].array()[el].pad(1, clip=True).fillna(0).flatten()
                table['y_ref'] = t['EcalScoringPlaneHits_sim.y_'].array()[el].pad(1, clip=True).fillna(0).flatten()
            elif coord_ref == 'target_sp':
                el = (t['TargetScoringPlaneHits_sim.pdgID_'].array() == 11) * \
                     (t['TargetScoringPlaneHits_sim.layerID_'].array() == 2) * \
                     (t['TargetScoringPlaneHits_sim.pz_'].array() > 0)
                table['x_ref'] = t['TargetScoringPlaneHits_sim.x_'].array()[el].pad(1, clip=True).fillna(0).flatten()
                table['y_ref'] = t['TargetScoringPlaneHits_sim.y_'].array()[el].pad(1, clip=True).fillna(0).flatten()
            else:
                raise RuntimeError('Invalid coord_ref: %s' % coord_ref)

        def _load_recoil_pt(t, table):
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

            (x, y, z), layer_id = self._parse_cid(eid)
            if coord_ref == 'ecal_centroid':
                e_sum = np.maximum(energy.sum(), 1e-6)
                table['x_ref'] = (x * energy).sum() / e_sum
                table['y_ref'] = (y * energy).sum() / e_sum
            x = x - table['x_ref']
            y = y - table['y_ref']

            var_dict = {self._id_branch:eid, self._energy_branch:energy,
                        'x':x, 'y':y, 'z':z, 'layer_id':layer_id,
                        'x_ref':table['x_ref'], 'y_ref':table['y_ref']}
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
        xyz = [_pad(a) for a in (self.var_data['x'], self.var_data['y'], self.var_data['z'])]
        layer_id = _pad(self.var_data['layer_id'])
        log_e = _pad(np.log(self.var_data[self._energy_branch]))
        self.coordinates = np.stack(xyz, axis=1).astype('float32')
        self.features = np.stack(xyz + [layer_id, log_e], axis=1).astype('float32')

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
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
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
