from __future__ import print_function

import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))

import pickle
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
from dataset import DGLGraphDatasetECALHits, collate_wrapper

from analysis_tools import ROC_from_model, save_ROC, inv_bkg_at_threshold, accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--demo', action='store_true', default=False)
parser.add_argument('--train-sig', type=str, default='/data/hqu/ldmx/mc/v9/signal_bdt_training/*/*.root')
parser.add_argument('--train-bkg', type=str, default='/data/hqu/ldmx/mc/v9/4pt0_gev_e_ecal_pn_bdt_training/*.root')
parser.add_argument('--test-sig', type=str, default='')
parser.add_argument('--test-bkg', type=str, default='')
# parser.add_argument('--data-format', type=str, default='particle', choices=['particle', 'lund'])
# parser.add_argument('--lund-dim', type=int, default=0)
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
    args.train_sig = '/data/hqu/ldmx/mc/v9/4pt0_gev_e_ecal_pn_bdt_training/4pt0_gev_1e_ecal_pn_v5_20190507_1eaf2f76_tskim_recon.root'
    args.train_bkg = '/data/hqu/ldmx/mc/v9/signal_bdt_training/1000mev/LDMX_W_UndecayedAP.4.0GeV.W.pMax.2.0.mA.1.0.2004_unweighted_events_tskim_recon.root'

# training/testing mode
if args.train_bkg and args.train_sig:
    training_mode = True
else:
    assert(args.load_model_path)
    training_mode = False

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
    train_data = DGLGraphDataset(args.train_bkg, args.train_sig, fraction=(0.2, 1))
    val_data = DGLGraphDataset(args.train_bkg, args.train_sig, fraction=(0, 0.2))
    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                              collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
    print('Train: %d events, Val: %d events' % (len(train_data), len(val_data)))

if args.test_sig and args.test_bkg:
    test_data = DGLGraphDataset(args.test_bkg, args.test_sig, fraction=(0, 0.2))
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)
else:
    print('Using val sample for testing!')
    test_data = val_data
    test_loader = val_loader
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
        print('Current validation acc: %.5f (best: %.5f)' % (valid_acc, best_valid_acc))

else:
    # load saved model
    model_path = args.load_model_path
    if not model_path.endswith('.pt'):
        model_path += '_state.pt'
    model.load_state_dict(torch.load(model_path))

# evaluate model on test dataset
path, name = os.path.split(args.test_output_path)
if path and not os.path.exists(path):
    os.makedirs(path)

test_preds = evaluate(model, test_loader, dev, return_scores=True)
test_labels = test_data.label.cpu().detach().numpy()

info_dict = {'model_name':args.network,
             'model_params': {'conv_params':conv_params, 'fc_params':fc_params},
             'date': str(datetime.date.today()),
             'model_path': args.load_model_path,
             'test_sig': args.test_sig,
             'test_bkg': args.test_bkg,
             'label': test_labels,
             'prediction':test_preds}
if training_mode:
    info_dict.update({'train_sig': args.train_sig,
                      'train_bkg': args.train_bkg,
                      'model_path': args.save_model_path})

eff_s, eff_b, auc = ROC_from_model(None, test_preds, test_labels)
info_dict['accuracy'] = accuracy(test_preds, test_labels)
info_dict['auc'] = auc
info_dict['inv_bkg_at_sig_50'] = inv_bkg_at_threshold(eff_s, eff_b, 0.5)
info_dict['inv_bkg_at_sig_30'] = inv_bkg_at_threshold(eff_s, eff_b, 0.3)

print(' === Summary ===')
for k in info_dict:
    print('%s: %s' % (k, info_dict[k]))

info_file = os.path.splitext(args.test_output_path)[0] + '_INFO.txt'
with open(info_file, 'w') as f:
    for k in info_dict:
        f.write('%s: %s\n' % (k, info_dict[k]))

info_file = os.path.splitext(args.test_output_path)[0] + '_pred.pickle'
with open(info_file, 'wb') as f:
    pickle.dump({'prediction':test_preds, 'label':test_labels}, f)

save_ROC(None, test_preds, test_labels, name=name, path=path, show=False, description=str(args))
