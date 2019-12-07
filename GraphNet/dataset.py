from __future__ import print_function

import dgl
import networkx as nx
import numpy as np
from dgl.transform import remove_self_loop
from dgl_utils import knn_graph
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import uproot
from uproot_methods import TLorentzVectorArray, TLorentzVector
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(12)


class DGLGraphDatasetECALHits(Dataset):

    def __init__(self, filepath_bkg, filepath_sig, fraction=(0, 1)):
        super(DGLGraphDatasetECALHits, self).__init__()
        # first load cell map
        self._load_cellMap()

        self.data = []
        branches = [
            'ecalDigis_recon.id_',
#             'ecalDigis_recon.amplitude_',
            'ecalDigis_recon.energy_',
#             'ecalDigis_recon.time_',
#             'ecalDigis_recon.isNoise_'
            ]

        def _load_data(data):
            start, stop = [int(x * len(data['ecalDigis_recon.id_'])) for x in fraction]
            for cid, e in zip(data['ecalDigis_recon.id_'][start:stop], data['ecalDigis_recon.energy_'][start:stop]):
                # iterate over each event
                g = dgl.DGLGraph()
                if len(cid) == 0:
                    g.add_nodes(1, {'coordinates':torch.zeros((1, 3), dtype=torch.float32), 'features':torch.zeros((1, 4), dtype=torch.float32)})
                else:
                    coords = self._parse_cellId(cid)
                    e = np.asfarray(e, dtype='float32').reshape((-1, 1))
                    features = np.concatenate([coords, e], axis=1)
                    g.add_nodes(len(coords), {'coordinates':torch.tensor(coords, dtype=torch.float32), 'features':torch.tensor(features, dtype=torch.float32)})
                self.data.append(g)

        print('Start loading dataset %s (bkg)' % filepath_bkg)
        for data in uproot.iterate(filepath_bkg, 'LDMX_Events', branches, namedecode='utf-8', executor=executor):
            # iterate over the chunk
            _load_data(data)
        nbkg = len(self.data)

        print('Start loading dataset %s (sig)' % filepath_sig)
        for data in uproot.iterate(filepath_sig, 'LDMX_Events', branches, namedecode='utf-8', executor=executor):
            # iterate over the chunk
            _load_data(data)
        ntotal = len(self.data)

        self.label = torch.zeros(ntotal, dtype=torch.float32)
        self.label[nbkg:] = 1

    def _load_cellMap(self, filename = 'cellmodule.txt'):
        self._cellMap = {}
        for i, x, y in np.loadtxt(filename):
            self._cellMap[i] = (x, y)
        self._layerZs = [223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,
            266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,
            322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,
            375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,
            448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125]

    def _parse_cellId(self, cid):
        cell = (cid & 0xFFFF8000) >> 15
        module = (cid & 0x7000) >> 12
        layer = (cid & 0xFF0) >> 4
        mcid = 10 * cell + module
        x, y = zip(*map(self._cellMap.__getitem__, mcid))
        z = list(map(self._layerZs.__getitem__, layer))
        return np.asfarray(np.stack([x, y, z], axis=1), dtype='float32')

    @property
    def num_features(self):
        return self.data[0].ndata['features'].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        return x, y


def pad_array(a, min_len=20, pad_value=0):
    if a.shape[0] < min_len:
        return F.pad(a, (0, 0, 0, min_len - a.shape[0]), mode='constant', value=pad_value)
    else:
        return a


class _SimpleCustomBatch:

    def __init__(self, data, k, min_nodes=20):
        transposed_data = list(zip(*data))
        graphs = []
        features = []
        for g in transposed_data[0]:
            nng = remove_self_loop(knn_graph(g.ndata['coordinates'], min(g.number_of_nodes(), k + 1)))
            if nng.number_of_nodes() < min_nodes:
                nng.add_nodes(min_nodes - nng.number_of_nodes())
            graphs.append(nng)
            fts = pad_array(g.ndata['features'], min_nodes, 0)
            features.append(fts)
            assert(nng.number_of_nodes() == fts.shape[0])
        self.batch_graph = dgl.batch(graphs)
        self.features = torch.cat(features, 0)
        self.label = torch.tensor(transposed_data[1])

    def pin_memory(self):
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch, k):
    return _SimpleCustomBatch(batch, k)
