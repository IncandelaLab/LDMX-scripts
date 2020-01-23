from __future__ import print_function

import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm
from functools import partial
import os
import datetime
import argparse

from ParticleNet import ParticleNet
import dataset
from dataset import DGLGraphDatasetECALHits, collate_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--demo', action='store_true', default=False)
# parser.add_argument('--train-sig', type=str, default='/data/hqu/ldmx/mc/v9/signal_hs/*/*.root')
# parser.add_argument('--train-bkg', type=str, default='/data/hqu/ldmx/mc/v9/4pt0_gev_e_ecal_pn_bdt_training/*.root')
# parser.add_argument('--test-sig', type=str, default='')
# parser.add_argument('--test-bkg', type=str, default='')
# parser.add_argument('--data-format', type=str, default='particle', choices=['particle', 'lund'])
# parser.add_argument('--lund-dim', type=int, default=0)
parser.add_argument('--predict', action='store_true', default=False)
parser.add_argument('--network', type=str, default='particle-net-lite', choices=['particle-net', 'particle-net-lite', 'particle-net-k5', 'particle-net-k7'])
parser.add_argument('--focal-loss-gamma', type=float, default=0)
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--test-output-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=30)
parser.add_argument('--start-lr', type=float, default=0.001)
parser.add_argument('--lr-steps', type=str, default='10,20')
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

if args.demo:
    dataset.bkglist = {
        # label: (filepath, num_events_for_training)
        0: ('/data/hqu/ldmx/mc/v9/4gev_1e_ecal_pn_02_1.48e13_gab/4gev_1e_ecal_pn_v9_1.8e8eot_20190507_00388e25_tskim_recon.root', -1)
        }

    dataset.siglist = {
        # label: (filepath, num_events_for_training)
        1: ('/data/hqu/ldmx/mc/v9/signal_hs/*mA.0.001*.root', 1000),
        10: ('/data/hqu/ldmx/mc/v9/signal_hs/*mA.0.01*.root', 1000),
        100: ('/data/hqu/ldmx/mc/v9/signal_hs/*mA.0.1*.root', 1000),
        1000: ('/data/hqu/ldmx/mc/v9/signal_hs/*mA.1.0*.root', 1000)
        }

# training/testing mode
if args.predict:
    assert(args.load_model_path)
    training_mode = False
else:
    training_mode = True

# data format
DGLGraphDataset = DGLGraphDatasetECALHits

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

# device
dev = torch.device(args.device)

# load data
collate_fn = partial(collate_wrapper, k=conv_params[0][0])
if training_mode:
    train_data = DGLGraphDataset(fraction=(0.2, 1))
    val_data = DGLGraphDataset(fraction=(0, 0.2))
    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                              collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)
    print('Train: %d events, Val: %d events' % (len(train_data), len(val_data)))
    print('Using val sample for testing!')
    test_data = val_data
    test_loader = val_loader
else:
    test_data = DGLGraphDataset(fraction=(0, 0.2), ignore_evt_limits=True)
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)

input_dims = test_data.num_features

# model
model = ParticleNet(input_dims=input_dims, num_classes=2,
                    conv_params=conv_params,
                    fc_params=fc_params)
model = model.to(dev)


def train(model, opt, scheduler, train_loader, dev):
    model.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for batch in tq:
            label = batch.label
            num_examples = label.shape[0]
            label = label.to(dev).squeeze().long()
            opt.zero_grad()
            logits = model(batch.batch_graph, batch.features.to(dev))
            loss = loss_func(logits, label)
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            num_batches += 1
            count += num_examples
            loss = loss.item()
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    scheduler.step()


def evaluate(model, test_loader, dev, return_scores=False):
    model.eval()

    total_correct = 0
    count = 0
    scores = []

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for batch in tq:
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                logits = model(batch.batch_graph, batch.features.to(dev))
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


if training_mode:
    # loss function
    if args.focal_loss_gamma > 0:
        print('Using focal loss w/ gamma=%s' % args.focal_loss_gamma)
        from focal_loss import FocalLoss
        loss_func = FocalLoss(gamma=args.focal_loss_gamma)
    else:
        loss_func = nn.CrossEntropyLoss()

    # optimizer
    opt = optim.Adam(model.parameters(), lr=args.start_lr)

    # learning rate
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)

    # training loop
    best_valid_acc = 0
    for epoch in range(args.num_epochs):
        train(model, opt, scheduler, train_loader, dev)

        print('Epoch #%d Validating' % epoch)
        valid_acc = evaluate(model, val_loader, dev)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if args.save_model_path:
                dirname = os.path.dirname(args.save_model_path)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                torch.save(model.state_dict(), args.save_model_path + '_state.pt')
                torch.save(model, args.save_model_path + '_full.pt')
        torch.save(model.state_dict(), args.save_model_path + '_state_epoch-%d_acc-%.4f.pt' % (epoch, valid_acc))
        print('Current validation acc: %.5f (best: %.5f)' % (valid_acc, best_valid_acc))

# load saved model
model_path = args.load_model_path if args.predict else args.save_model_path
if not model_path.endswith('.pt'):
    model_path += '_state.pt'
print('Loading model %s for eval' % model_path)
model.load_state_dict(torch.load(model_path))

# evaluate model on test dataset
path, name = os.path.split(args.test_output_path)
if path and not os.path.exists(path):
    os.makedirs(path)

test_preds = evaluate(model, test_loader, dev, return_scores=True)
test_labels = test_data.label.cpu().detach().numpy()
test_extra_labels = test_data.extra_labels

info_dict = {'model_name':args.network,
             'model_params': {'conv_params':conv_params, 'fc_params':fc_params},
             'date': str(datetime.date.today()),
             'model_path': args.load_model_path,
             'siglist': dataset.siglist,
             'bkglist': dataset.bkglist,
             'presel_eff': test_data.presel_eff,
             }
if training_mode:
    info_dict.update({'model_path': args.save_model_path})

from plot_utils import plotROC, get_signal_effs
for k in dataset.siglist:
    mass = '%d MeV' % k
    fpr, tpr, auc, acc = plotROC(test_preds, test_labels, sample_weight=np.logical_or(test_extra_labels == 0, test_extra_labels == k),
                                 sig_eff=test_data.presel_eff[k], bkg_eff=test_data.presel_eff[0],
                                 output=os.path.splitext(args.test_output_path)[0] + 'ROC_%s.pdf' % mass, label=mass, xlim=[1e-6, .01], ylim=[0, 1], logx=True)
    info_dict[mass] = {'auc-presel': auc,
                       'acc-presel': acc,
                       'effs': get_signal_effs(fpr, tpr)
                       }

print(' === Summary ===')
for k in info_dict:
    print('%s: %s' % (k, info_dict[k]))

info_file = os.path.splitext(args.test_output_path)[0] + '_INFO.txt'
with open(info_file, 'w') as f:
    for k in info_dict:
        f.write('%s: %s\n' % (k, info_dict[k]))

import pickle
pred_file = os.path.splitext(args.test_output_path)[0] + '_pred.pickle'
with open(pred_file, 'wb') as f:
    pickle.dump({'prediction':test_preds, 'label':test_extra_labels}, f)
