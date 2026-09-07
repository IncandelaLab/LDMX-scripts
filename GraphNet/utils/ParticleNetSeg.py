"""Per-hit variant of ParticleNetX for splitting overlapping ECal showers,
with no tracker and no projected trajectory.

ParticleNetX pools over hits and SplitNetX concatenates one vector per region,
giving one number per event. This keeps the EdgeConv backbone, drops the
pooling, and runs over all ECal hits at once. No region split: the showers
overlap across region boundaries and SplitNetX's sub-nets cannot exchange
information.

Without a trajectory the task splits into two problems of very different
difficulty, and they should be trained and reported separately:

  1. Separation. Partition the hits into two showers. Well posed from ECal
     hits alone, but only up to a swap: nothing in the hits says which cluster
     is which. Train it with symmetric_soft_bce, which takes the better of the
     two labellings per event. The head output then means "cluster A vs B",
     not "electron vs photon".

  2. Identification. Decide which cluster is the electron. Weak. The only real
     handle is depth: the electron ionises from the front face while the photon
     must convert first, ~9/7 X0 in. Total cluster energy helps if the beam
     energy is known. Use cluster_summary and a cut or small MLP, and quote its
     accuracy on its own.

Mixing the two into one number hides which half is failing, and the separation
half is the one that can actually work.

Target is the electron energy fraction per hit, soft in [0, 1]. A cell can
collect energy from both showers, so a hard label is not well defined in the
overlap core. Truth is still available at training time; it is only the
trajectory as an input that is gone.
"""

from __future__ import print_function

import torch
import torch.nn as nn

from utils.ParticleNetX import EdgeConvBlock, Mish

# ParticleNetX sets float64 globally because the veto's scores saturate near
# 1 - 1e-6. A fraction in [0, 1] does not need it, and float64 roughly halves
# GPU throughput. Drop this line if sharing a process with the veto model.
torch.set_default_dtype(torch.float32)


class ParticleNetSeg(nn.Module):
    """EdgeConv backbone with a 1x1-conv per-hit head.

    forward(points (E,3,P), features (E,F,P), mask (E,1,P)) -> logits (E,out,P).
    Pass mask explicitly; the fallback infers it from all-zero features, which
    drops any real hit whose features happen to sum to zero.

    Defaults are one block deeper and wider than the veto config, which
    node-level tasks generally need. use_global_context appends a masked global
    average to every hit, which matters more here than with a trajectory: it is
    how a hit learns whether the event is separable at all.

    To try max aggregation (the original DGCNN/ParticleNet choice; the LDMX
    EdgeConvBlock uses mean), copy EdgeConvBlock and change
    `torch.mean(x, axis=-1)` to `torch.amax(x, dim=-1)`. Worth testing here:
    with no trajectory the model leans entirely on local shower structure.
    """

    def __init__(self,
                 input_dims,
                 conv_params=((16, (64, 64, 64)),
                              (16, (128, 128, 128)),
                              (16, (128, 128, 128))),
                 head_channels=(128, 64),
                 dropout=0.1,
                 out_dims=1,
                 use_fusion=True,
                 use_global_context=True,
                 **kwargs):
        super(ParticleNetSeg, self).__init__(**kwargs)

        conv_params = list(conv_params)
        self.use_fusion = use_fusion
        self.use_global_context = use_global_context

        self.bn_fts = nn.BatchNorm1d(input_dims)

        self.edge_convs = nn.ModuleList()
        for index, (num_neighbours, channels) in enumerate(conv_params):
            in_feat = input_dims if index == 0 else conv_params[index - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=num_neighbours,
                                                 in_feat=in_feat,
                                                 out_feats=list(channels)))

        if use_fusion:
            fusion_in = sum(channels[-1] for _, channels in conv_params)
            fusion_out = int(min(max((fusion_in // 128) * 128, 128), 1024))
            self.fusion_block = nn.Sequential(
                nn.Conv1d(fusion_in, fusion_out, kernel_size=1),
                nn.BatchNorm1d(fusion_out), Mish())
            node_channels = fusion_out
        else:
            node_channels = conv_params[-1][1][-1]

        # doubled because the global context is concatenated onto each hit
        head_in = node_channels * 2 if use_global_context else node_channels
        head_layers = []
        for out_channels in head_channels:
            head_layers += [nn.Conv1d(head_in, out_channels, kernel_size=1,
                                      bias=False),
                            nn.BatchNorm1d(out_channels), Mish(),
                            nn.Dropout(dropout)]
            head_in = out_channels
        head_layers.append(nn.Conv1d(head_in, out_dims, kernel_size=1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, points, features, mask=None):
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)
        mask = mask.to(features.dtype)

        # Push padding far away so the kNN skips it. Inherited caveat: if an
        # event has fewer real hits than k+1, real hits still pick up padding.
        coord_shift = (mask == 0).to(features.dtype) * 9999.0

        features = self.bn_fts(features) * mask  # zero the padding after BN

        block_outputs = []
        for index, conv in enumerate(self.edge_convs):
            # block 0 uses real space; later blocks rebuild the kNN in
            # learned feature space, which is what groups hits by shower
            coords = (points if index == 0 else features) + coord_shift
            features = conv(coords, features) * mask
            block_outputs.append(features)

        if self.use_fusion:
            features = self.fusion_block(torch.cat(block_outputs, dim=1)) * mask

        if self.use_global_context:
            counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            # masked mean over real hits, handed back to every hit
            context = (features * mask).sum(dim=-1, keepdim=True) / counts
            features = torch.cat(
                [features, context.expand(-1, -1, features.size(-1))], dim=1)

        return self.head(features) * mask


def _per_event_bce(logits, target, weights, eps=1e-6):
    per_hit = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    return (per_hit * weights).sum(dim=-1) / (weights.sum(dim=-1) + eps)


def _flatten_inputs(logits, mask, hit_energy):
    logits = logits.squeeze(1)
    if mask.dim() == 3:
        mask = mask.squeeze(1)
    mask = mask.to(logits.dtype)
    weights = mask if hit_energy is None else mask * hit_energy.to(logits.dtype)
    return logits, mask, weights


def symmetric_soft_bce(logits, target_frac, mask, hit_energy=None,
                       return_flip=False):
    """Permutation-invariant loss: the better of f and 1-f, per event.

    Use this when there is no trajectory. It supervises the split without
    demanding that the network also decide which cluster is the electron, so
    the swap ambiguity does not propagate into the gradients as noise.

    Energy weighting makes the loss track how much energy is correctly split
    rather than the raw hit count, which the halo dominates.
    """
    logits, mask, weights = _flatten_inputs(logits, mask, hit_energy)
    target = target_frac.to(logits.dtype)

    direct = _per_event_bce(logits, target, weights)
    flipped = _per_event_bce(logits, 1.0 - target, weights)  # same split, swapped
    loss = torch.minimum(direct, flipped).mean()  # per event, take the better one
    return (loss, flipped < direct) if return_flip else loss


def masked_soft_bce(logits, target_frac, mask, hit_energy=None):
    """Non-symmetric version: asks the network to separate AND identify.

    Only meaningful if you want the model to learn e/gamma identity from shower
    depth on its own. Compare it against symmetric_soft_bce; if it is much
    worse, the identification is what is failing, not the separation.
    """
    logits, mask, weights = _flatten_inputs(logits, mask, hit_energy)
    return _per_event_bce(logits, target_frac.to(logits.dtype), weights).mean()


@torch.no_grad()
def purity_completeness(logits, target_frac, hit_energy, mask, threshold=0.5,
                        symmetric=True, eps=1e-6):
    """Per-event energy purity and completeness of the recovered electron.

    With symmetric=True the better of the two cluster-to-truth assignments is
    taken, matching symmetric_soft_bce, so this measures separation quality
    only. Bin it in photon energy and in the truth opening angle: that is the
    plot showing where the split turns on.

    Returns two (E,) tensors.
    """
    probability = torch.sigmoid(logits.squeeze(1))
    if mask.dim() == 3:
        mask = mask.squeeze(1)
    mask = mask.to(probability.dtype)

    energy = hit_energy.to(probability.dtype) * mask
    electron_energy = energy * target_frac.to(probability.dtype)
    total_electron = electron_energy.sum(dim=-1)

    def score(assigned):
        captured = (assigned * electron_energy).sum(dim=-1)
        return (captured / ((assigned * energy).sum(dim=-1) + eps),
                captured / (total_electron + eps))

    cluster_a = (probability > threshold).to(probability.dtype) * mask
    purity, completeness = score(cluster_a)
    if symmetric:
        purity_b, completeness_b = score((1.0 - cluster_a) * mask)
        take_b = completeness_b > completeness  # match the loss's choice of swap
        purity = torch.where(take_b, purity_b, purity)
        completeness = torch.where(take_b, completeness_b, completeness)
    return purity, completeness


@torch.no_grad()
def cluster_summary(logits, layer, hit_energy, mask, threshold=0.5,
                    start_frac=0.02, eps=1e-6):
    """Per-cluster quantities for the identification step.

    Returns (E, 2, 3): for cluster A then cluster B, the total energy, the
    energy-weighted mean layer, and the shower-start layer, defined as the
    first layer where the cluster's cumulative energy exceeds start_frac of its
    total.

    The shower-start layer is the discriminant: the electron ionises from the
    front face, the photon has to convert first. It is a weak handle, so
    measure its separation power before building anything on top of it.
    """
    probability = torch.sigmoid(logits.squeeze(1))
    if mask.dim() == 3:
        mask = mask.squeeze(1)
    mask = mask.to(probability.dtype)
    energy = hit_energy.to(probability.dtype) * mask
    layer = layer.to(probability.dtype)

    cluster_a = (probability > threshold).to(probability.dtype) * mask
    summaries = []
    for assigned in [cluster_a, (1.0 - cluster_a) * mask]:
        cluster_energy = assigned * energy
        total = cluster_energy.sum(dim=-1)
        mean_layer = (cluster_energy * layer).sum(dim=-1) / (total + eps)

        # sort by depth, with hits outside this cluster pushed to the back
        order = torch.argsort(layer + (1.0 - assigned) * 1e6, dim=-1)
        energy_sorted = torch.gather(cluster_energy, -1, order)
        layer_sorted = torch.gather(layer, -1, order)
        cumulative = torch.cumsum(energy_sorted, dim=-1) / (total[:, None] + eps)
        first_index = torch.argmax(  # argmax of a bool gives the first True
            (cumulative >= start_frac).to(probability.dtype), dim=-1,
            keepdim=True)
        start_layer = torch.gather(layer_sorted, -1, first_index).squeeze(-1)

        summaries.append(torch.stack([total, mean_layer, start_layer], dim=-1))
    return torch.stack(summaries, dim=1)


if __name__ == '__main__':
    torch.manual_seed(0)
    num_events, max_hits, num_features, num_real = 8, 300, 10, 220

    mask = torch.zeros(num_events, 1, max_hits)
    mask[:, :, :num_real] = 1.0
    points = torch.randn(num_events, 3, max_hits) * mask
    features = torch.randn(num_events, num_features, max_hits) * mask
    hit_energy = torch.rand(num_events, max_hits) * mask.squeeze(1)
    target_frac = torch.rand(num_events, max_hits) * mask.squeeze(1)
    layer = torch.randint(0, 34, (num_events, max_hits)).float() * mask.squeeze(1)

    model = ParticleNetSeg(input_dims=num_features)
    logits = model(points, features, mask)
    print('logits:', tuple(logits.shape))

    sym_loss, flipped = symmetric_soft_bce(logits, target_frac, mask, hit_energy,
                                           return_flip=True)
    asym_loss = masked_soft_bce(logits, target_frac, mask, hit_energy)
    print('symmetric loss :', round(sym_loss.detach().item(), 4),
          '| flipped:', flipped.tolist())
    print('asymmetric loss:', round(asym_loss.detach().item(), 4))
    print('symmetric <= asymmetric:',
          sym_loss.detach().item() <= asym_loss.detach().item() + 1e-9)

    sym_loss.backward()
    print('grad flows:', sum(parameter.grad.abs().sum().item()
                             for parameter in model.parameters()
                             if parameter.grad is not None) > 0)

    purity, completeness = purity_completeness(logits, target_frac, hit_energy,
                                               mask)
    print('purity/completeness:', round(purity.mean().item(), 3),
          '/', round(completeness.mean().item(), 3))
    stats = cluster_summary(logits, layer, hit_energy, mask)
    print('cluster summary:', tuple(stats.shape))
    print('padding clean:',
          logits[:, :, num_real:].detach().abs().max().item() == 0.0)
