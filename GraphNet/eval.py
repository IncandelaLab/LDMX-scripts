from __future__ import print_function

import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))

import numpy as np
import torch
from torch.utils.data import DataLoader

import tqdm
import glob
import os
import datetime
import argparse

from ParticleNet import ParticleNet
from dataset import ECalHitsDataset
from dataset import collate_wrapper as collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--test-sig', type=str, default='')
parser.add_argument('--test-bkg', type=str, default='')
parser.add_argument('--coord-ref', type=str, default='ecal_sp', choices=['none', 'ecal_sp', 'target_sp', 'ecal_centroid'])
parser.add_argument('--save-extra', action='store_true', default=False)
parser.add_argument('--network', type=str, default='particle-net-lite', choices=['particle-net', 'particle-net-lite', 'particle-net-k5', 'particle-net-k7'])
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--test-output-path', type=str, default='')
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

obs_branches = []

if args.save_extra:
    obs_branches = [
        'ecalDigis_recon.id_',
        'ecalDigis_recon.energy_',

        'EcalVetoGabriel_recon.nReadoutHits_',
        'EcalVetoGabriel_recon.deepestLayerHit_',
        'EcalVetoGabriel_recon.summedDet_',
        'EcalVetoGabriel_recon.summedTightIso_',
        'EcalVetoGabriel_recon.maxCellDep_',
        'EcalVetoGabriel_recon.showerRMS_',
        'EcalVetoGabriel_recon.xStd_',
        'EcalVetoGabriel_recon.yStd_',
        'EcalVetoGabriel_recon.avgLayerHit_',
        'EcalVetoGabriel_recon.stdLayerHit_',
        'EcalVetoGabriel_recon.ecalBackEnergy_',
        'EcalVetoGabriel_recon.electronContainmentEnergy_',
        'EcalVetoGabriel_recon.photonContainmentEnergy_',
        'EcalVetoGabriel_recon.outsideContainmentEnergy_',
        'EcalVetoGabriel_recon.outsideContainmentNHits_',
        'EcalVetoGabriel_recon.outsideContainmentXStd_',
        'EcalVetoGabriel_recon.outsideContainmentYStd_',
        'EcalVetoGabriel_recon.discValue_',
        'EcalVetoGabriel_recon.recoilPx_',
        'EcalVetoGabriel_recon.recoilPy_',
        'EcalVetoGabriel_recon.recoilPz_',
        'EcalVetoGabriel_recon.recoilX_',
        'EcalVetoGabriel_recon.recoilY_',
        'EcalVetoGabriel_recon.ecalLayerEdepReadout_',

        'TargetSPRecoilE_pt',
        ]

# model parameter
if args.network == 'particle-net':
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
elif args.network == 'particle-net-lite':
    conv_params = [
        (7, (32, 32, 32)),
        (7, (64, 64, 64))
        ]
    fc_params = [(128, 0.1)]
elif args.network == 'particle-net-k5':
    conv_params = [
        (5, (64, 64, 64)),
        (5, (128, 128, 128)),
        (5, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
elif args.network == 'particle-net-k7':
    conv_params = [
        (7, (64, 64, 64)),
        (7, (128, 128, 128)),
        (7, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]

print('conv_params: %s' % conv_params)
print('fc_params: %s' % fc_params)

# device
dev = torch.device(args.device)

# load data
input_dims = 5

# model
model = ParticleNet(input_dims=input_dims, num_classes=2,
                    conv_params=conv_params,
                    fc_params=fc_params,
                    use_fusion=True)
model = model.to(dev)


def evaluate(model, test_loader, dev, return_scores=False):
    model.eval()

    total_correct = 0
    count = 0
    scores = []

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch in tq:
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                logits = model(batch.coordinates.to(dev), batch.features.to(dev))
                _, preds = logits.max(1)

                if return_scores:
                    scores.append(torch.softmax(logits, dim=1).cpu().detach().numpy())

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix({
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

    if return_scores:
        return np.concatenate(scores)
    else:
        return total_correct / count


# load saved model
model_path = args.load_model_path
if not model_path.endswith('.pt'):
    model_path += '_state.pt'
print('Loading model %s for eval' % model_path)
model.load_state_dict(torch.load(model_path))

# evaluate model on test dataset
path = args.test_output_path
if not os.path.exists(path):
    os.makedirs(path)


def run_one_file(filepath, extra_label=0):
    pred_file = os.path.join(path, os.path.basename(filepath).replace('.root', '.awkd'))
    if os.path.exists(pred_file):
        print('skip %s' % filepath)
        return

    siglist = {}
    bkglist = {}
    if extra_label == 0:
        bkglist = {0:(filepath, -1)}
    else:
        siglist = {extra_label:(filepath, -1)}

    test_frac = (0, 1) if args.test_sig or args.test_bkg else (0, 0.2)
    test_data = ECalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=test_frac, ignore_evt_limits=True, obs_branches=obs_branches, coord_ref=args.coord_ref)
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)

    test_preds = evaluate(model, test_loader, dev, return_scores=True)
    test_labels = test_data.label
#     test_extra_labels = test_data.extra_labels

    import awkward
    out_data = test_data.obs_data
#     out_data['ParticleNet_extra_label'] = test_extra_labels
    out_data['ParticleNet_disc'] = test_preds[:, 1]
    awkward.save(pred_file, out_data, mode='w')
    print('Written pred to %s' % pred_file)


info_dict = {'model_name':args.network,
             'model_params': {'conv_params':conv_params, 'fc_params':fc_params},
             'date': str(datetime.date.today()),
             'model_path': args.load_model_path,
             'siglist': args.test_sig,
             'bkglist': args.test_bkg,
             }

info_file = os.path.join(path, 'eval_INFO.txt')
with open(info_file, 'w') as f:
    for k in info_dict:
        f.write('%s: %s\n' % (k, info_dict[k]))

bkg_filelist = sorted(glob.glob(args.test_bkg))
for idx, f in enumerate(bkg_filelist):
    print('%d/%d' % (idx, len(bkg_filelist)))
    run_one_file(f)

for f in sorted(glob.glob(args.test_sig)):
    run_one_file(f)
