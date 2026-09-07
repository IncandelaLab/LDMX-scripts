"""Build per-hit inputs and soft labels for ECal e-/gamma segmentation,
using ECal information only: no tracker, no projected trajectory.
"""

from __future__ import print_function

import numpy as np
import awkward as ak

# DetDescr/include/DetDescr/EcalID.h
LAYER_MASK, LAYER_SHIFT = 0x3F, 17
MODULE_MASK, MODULE_SHIFT = 0x1F, 12
CELL_MASK, CELL_SHIFT = 0xFFF, 0

NUM_ECAL_LAYERS = 34


def decode_ecal_id(ids):
    """Packed EcalID -> (layer, module, cell). numpy or awkward."""
    return ((ids >> LAYER_SHIFT) & LAYER_MASK,
            (ids >> MODULE_SHIFT) & MODULE_MASK,
            (ids >> CELL_SHIFT) & CELL_MASK)


def electron_fraction_per_simhit(edep_contribs, incident_id_contribs,
                                 recoil_track_id, eps=1e-12):
    """Soft label per sim hit. Inputs doubly jagged (event, hit, contrib)."""
    from_electron = incident_id_contribs == recoil_track_id
    # electron edep over total edep, per cell -> soft label in [0, 1]
    return (ak.sum(edep_contribs * from_electron, axis=-1)
            / (ak.sum(edep_contribs, axis=-1) + eps))


def _match_key(event_index, ids):
    # pack (event, cell) into one int64 so matching is a single sorted lookup
    return event_index.astype(np.int64) * (1 << 32) + ids.astype(np.int64)


def match_rec_to_sim(rec_ids, sim_ids, sim_values):
    """Look up a per-sim-hit value for each rec hit, matching on EcalID.

    Returns flat (values, matched) aligned with the flattened rec hits.
    matched is False for noise hits and threshold artefacts.
    """
    rec_counts = np.asarray(ak.num(rec_ids, axis=1))
    sim_counts = np.asarray(ak.num(sim_ids, axis=1))
    rec_events = np.repeat(np.arange(len(rec_counts)), rec_counts)
    sim_events = np.repeat(np.arange(len(sim_counts)), sim_counts)

    rec_keys = _match_key(rec_events, np.asarray(ak.flatten(rec_ids)))
    sim_keys = _match_key(sim_events, np.asarray(ak.flatten(sim_ids)))
    values = np.asarray(ak.flatten(sim_values), dtype=np.float64)

    order = np.argsort(sim_keys, kind='stable')
    sim_keys, values = sim_keys[order], values[order]

    position = np.clip(np.searchsorted(sim_keys, rec_keys), 0,
                       max(len(sim_keys) - 1, 0))
    # searchsorted returns an insertion point, so confirm it is a real hit
    matched = (len(sim_keys) > 0) & (sim_keys[position] == rec_keys)
    return np.where(matched, values[position], 0.0), matched


def energy_centroid(xpos, ypos, energy, mask, layer=None, max_layer=None,
                    eps=1e-6):
    """Energy-weighted (x, y) centroid per event, as (E, 1) arrays.

    Restrict to the front layers with max_layer for a proxy of the entry point.
    """
    weights = energy * mask
    if max_layer is not None:
        weights = weights * (layer <= max_layer)  # front layers only
    total = weights.sum(axis=1, keepdims=True) + eps
    return ((weights * xpos).sum(axis=1, keepdims=True) / total,
            (weights * ypos).sum(axis=1, keepdims=True) / total)


def layer_energy_totals(energy, layer, mask):
    """Total energy in each hit's own layer, broadcast back per hit."""
    totals = np.zeros_like(energy)
    for layer_index in range(NUM_ECAL_LAYERS + 1):
        selected = (layer == layer_index) * mask
        # layer sum written back onto every hit in that layer
        totals += selected * (energy * selected).sum(axis=1, keepdims=True)
    return totals


def build_features(xpos, ypos, zpos, energy, layer, mask, front_layers=5,
                   scale_xy=50.0, eps=1e-6):
    """(E, F, P) trajectory-free feature tensor.

    Positions enter relative to the two centroids rather than in absolute ECal
    coordinates, so the network is not asked to relearn translation invariance
    from a finite sample. The last feature, energy relative to the hit's own
    layer total, gives the core/halo distinction without the network having to
    infer the longitudinal profile itself.
    """
    centroid_x, centroid_y = energy_centroid(xpos, ypos, energy, mask)
    front_x, front_y = energy_centroid(xpos, ypos, energy, mask, layer,
                                       front_layers)

    delta_x = (xpos - centroid_x) / scale_xy
    delta_y = (ypos - centroid_y) / scale_xy
    front_delta_x = (xpos - front_x) / scale_xy
    front_delta_y = (ypos - front_y) / scale_xy

    return np.stack([
        np.log(np.clip(energy, eps, None)),
        layer.astype(np.float64) / float(NUM_ECAL_LAYERS),
        zpos / 1000.0,
        delta_x, delta_y, np.hypot(delta_x, delta_y),
        front_delta_x, front_delta_y, np.hypot(front_delta_x, front_delta_y),
        energy / (layer_energy_totals(energy, layer, mask) + eps),
    ], axis=1)


def pad_event_arrays(flat, counts, max_hits, dtype=np.float64):
    """Flat per-hit array -> dense (E, max_hits).

    Truncation follows rec-hit order. Prefer setting max_hits above the bulk of
    the distribution: unlike the veto, this needs the whole shower.
    """
    padded = np.zeros((len(counts), max_hits), dtype=dtype)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    for event in range(len(counts)):
        num_kept = min(int(counts[event]), max_hits)
        padded[event, :num_kept] = flat[offsets[event]:offsets[event] + num_kept]
    return padded


def assemble(rec, sim, recoil_track_id, max_hits=800,
             drop_noise_from_loss=True, front_layers=5):
    """Everything the network and loss need, as dense numpy arrays.

    rec: dict of ak arrays with id_, energy_, xpos_, ypos_, zpos_, is_noise_
    sim: dict of ak arrays with id_, edep_contribs_, incident_id_contribs_

    Returns points (E,3,P), features (E,F,P), target (E,P), energy (E,P),
    layer (E,P), hit_mask (E,1,P), loss_mask (E,1,P).
    """
    counts = np.asarray(ak.num(rec['id_'], axis=1))
    counts_kept = np.minimum(counts, max_hits)

    sim_fraction = electron_fraction_per_simhit(
        sim['edep_contribs_'], sim['incident_id_contribs_'], recoil_track_id)
    fraction_flat, matched_flat = match_rec_to_sim(rec['id_'], sim['id_'],
                                                   sim_fraction)

    layer_flat, _, _ = decode_ecal_id(np.asarray(ak.flatten(rec['id_'])))

    def dense(branch):
        return pad_event_arrays(np.asarray(ak.flatten(branch)), counts, max_hits)

    xpos, ypos, zpos = dense(rec['xpos_']), dense(rec['ypos_']), dense(rec['zpos_'])
    energy, is_noise = dense(rec['energy_']), dense(rec['is_noise_'])
    layer = pad_event_arrays(layer_flat, counts, max_hits)
    target = pad_event_arrays(fraction_flat, counts, max_hits)
    matched = pad_event_arrays(matched_flat.astype(np.float64), counts, max_hits)

    hit_mask = (np.arange(max_hits)[None, :]
                < counts_kept[:, None]).astype(np.float64)

    # network sees every real hit; the loss only sees hits with a sim match
    loss_mask = hit_mask * matched
    if drop_noise_from_loss:
        loss_mask = loss_mask * (1.0 - is_noise)

    features = build_features(xpos, ypos, zpos, energy, layer, hit_mask,
                              front_layers)

    return {
        # z compressed so the kNN is not dominated by layer spacing
        'points': np.stack([xpos, ypos, zpos / 10.0], axis=1) * hit_mask[:, None, :],
        'features': features * hit_mask[:, None, :],
        'target': target * hit_mask,
        'energy': energy * hit_mask,
        'layer': layer * hit_mask,
        'hit_mask': hit_mask[:, None, :],
        'loss_mask': loss_mask[:, None, :],
    }


# recoil_track_id is usually 1 in single-electron samples, but confirm it on
# your files via SimParticles. Events where it is not are the pathological ones
# and deserve a look rather than a silent mislabel.


def _make_synthetic_events(num_events, recoil_id, photon_id, seed=0):
    """Synthetic rec/sim collections with the pathologies that matter:
    sim hits killed by thresholds, pure-noise rec hits, multi-contribution
    cells shared between the two incident particles.
    """
    rng = np.random.default_rng(seed)
    rec, sim = {key: [] for key in
                ['id_', 'xpos_', 'ypos_', 'zpos_', 'energy_', 'is_noise_']}, \
               {key: [] for key in
                ['id_', 'edep_contribs_', 'incident_id_contribs_']}

    for _ in range(num_events):
        num_cells = int(rng.integers(20, 40))
        cell_ids = np.unique(
            ((rng.integers(0, NUM_ECAL_LAYERS, num_cells) << LAYER_SHIFT)
             | (1 << MODULE_SHIFT)
             | rng.integers(0, 400, num_cells)).astype(np.int64))

        edeps, incidents = [], []
        for _ in range(len(cell_ids)):
            num_contribs = int(rng.integers(1, 4))
            edeps.append(list(rng.uniform(0.1, 5.0, num_contribs)))
            incidents.append(list(rng.choice([recoil_id, photon_id],
                                             num_contribs)))
        sim['id_'].append(list(cell_ids))
        sim['edep_contribs_'].append(edeps)
        sim['incident_id_contribs_'].append(incidents)

        survives = rng.random(len(cell_ids)) < 0.85  # threshold losses
        num_survived = int(survives.sum())
        rec_ids = list(cell_ids[survives]) + \
            list(np.unique(rng.integers(1 << 20, 1 << 21, 2)))
        num_rec = len(rec_ids)
        rec['id_'].append(rec_ids)
        rec['xpos_'].append(list(rng.normal(0, 30, num_rec)))
        rec['ypos_'].append(list(rng.normal(0, 30, num_rec)))
        rec['zpos_'].append(list(rng.uniform(240, 700, num_rec)))
        rec['energy_'].append(list(rng.uniform(0.05, 8.0, num_rec)))
        rec['is_noise_'].append([False] * num_survived
                                + [True] * (num_rec - num_survived))

    return ({key: ak.Array(value) for key, value in rec.items()},
            {key: ak.Array(value) for key, value in sim.items()})


if __name__ == '__main__':
    import torch
    from utils.ParticleNetSeg import (ParticleNetSeg, symmetric_soft_bce,
                                      purity_completeness, cluster_summary)

    num_events, max_hits, recoil_id, photon_id = 6, 64, 1, 7
    rec, sim = _make_synthetic_events(num_events, recoil_id, photon_id)
    batch = assemble(rec, sim, np.full(num_events, recoil_id), max_hits=max_hits)

    for name, array in batch.items():
        print('{:<10} {}'.format(name, tuple(array.shape)))

    target = batch['target']
    hit_mask = batch['hit_mask'][:, 0]
    loss_mask = batch['loss_mask'][:, 0]
    print('target range      : [{:.3f}, {:.3f}]'.format(target.min(), target.max()))
    print('real hits/event   :', hit_mask.sum(1).astype(int).tolist())
    print('labelled/event    :', loss_mask.sum(1).astype(int).tolist())
    print('noise excluded    :', bool((loss_mask <= hit_mask).all())
          and loss_mask.sum() < hit_mask.sum())
    print('finite features   :', bool(np.isfinite(batch['features']).all()))
    print('padding clean     :', float(np.abs(
        batch['features'] * (1 - hit_mask[:, None])).max()) == 0.0)

    event = 0
    reference = {}
    for cell_id, edeps, incidents in zip(np.asarray(sim['id_'][event]),
                                         ak.to_list(sim['edep_contribs_'][event]),
                                         ak.to_list(sim['incident_id_contribs_'][event])):
        electron = sum(edep for edep, incident in zip(edeps, incidents)
                       if incident == recoil_id)
        reference[int(cell_id)] = electron / sum(edeps)
    rec_ids = np.asarray(rec['id_'][event])
    print('label closure     :', all(
        abs(target[event, index] - reference[int(rec_ids[index])]) < 1e-9
        for index in range(len(rec_ids)) if int(rec_ids[index]) in reference))

    def to_tensor(array):
        return torch.tensor(array, dtype=torch.float32)

    model = ParticleNetSeg(input_dims=batch['features'].shape[1])
    logits = model(to_tensor(batch['points']), to_tensor(batch['features']),
                   to_tensor(batch['hit_mask']))
    loss = symmetric_soft_bce(logits, to_tensor(batch['target']),
                              to_tensor(batch['loss_mask']),
                              to_tensor(batch['energy']))
    loss.backward()
    purity, completeness = purity_completeness(
        logits, to_tensor(batch['target']), to_tensor(batch['energy']),
        to_tensor(batch['loss_mask']))
    stats = cluster_summary(logits, to_tensor(batch['layer']),
                            to_tensor(batch['energy']),
                            to_tensor(batch['hit_mask']))
    print('end-to-end loss   : {:.4f}'.format(loss.detach().item()))
    print('purity/complete   : {:.3f} / {:.3f}'.format(
        purity.mean().item(), completeness.mean().item()))
    print('cluster summary   :', tuple(stats.shape))
