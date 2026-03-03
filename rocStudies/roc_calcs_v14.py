# Authors: Oscar Lewis, Jihoon Yoo, Danyi Zhang
# Date: 2025-11-27

import sys, os, math
import ROOT as r
import numpy as np
import awkward as ak
import csv
import bisect
from collections import defaultdict
# PATH is at LDMX install/lib which 
# contains all the shared object libraries for ldmx-sw
r.gSystem.Load('libEcal_Event.so')
r.gSystem.Load('libDetDescr.so')
r.gSystem.Load('libFramework.so')
r.gSystem.Load('libSimCore_Event.so')

ecal_front_z = 240.0
ecal_layerZs = ecal_front_z + np.array([
                      7.582, 16.062, 33.226, 43.206, 60.370, 71.350, 90.014, 101.594, 
                      120.258, 131.838, 150.502, 162.082, 180.746, 192.326, 210.990, 
                      222.570, 241.234, 252.814, 271.478, 283.058, 301.722, 313.302, 
                      331.966, 343.546, 365.710, 380.690, 402.854, 417.834, 439.998,
                      454.978, 477.142, 492.122, 514.286, 529.266                   ])
sp_thickness = 0.001

# Debugging Pause
def pause():
    programPause = input("Press the <ENTER> key to continue...")

def printHitInfo(hit:r.ldmx.EcalHit):
    print("    z={}, pz={}, pid={}, trackid={}".format(hit.getPosition()[2], hit.getMomentum()[2], hit.getPdgID(), hit.getTrackID()))

def debugPrint(string:str, debug:bool=True) -> None:
    if debug: print(string)
    return None

def writeCSV(columns:np.ndarray|list, data:np.ndarray, out_name:str) -> None:
    """
    Takes in a list of column titles and a numpy array of data
    indexed with arr[row][column], and writes a CSV file to `out_name.csv`
    in the working directory.
    """
    with open(f'{out_name}.csv', 'w', newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in data:
            w.writerow(row)
    print(f"Wrote {len(data)} rows to: {out_name}.csv")
    
    return None

def loadEventFiles(file_paths:list[str]) -> tuple[int, int, r.TChain]:
    """Takes a list of file paths containing LDMX_Events tree objects
     and loads a TChain containing all event data."""
    print(f'Loading event files with loadEventFiles')
    tree_name = 'LDMX_Events'
    nfiles = 0
    chain = r.TChain(tree_name)
    for path in file_paths:
        chain.Add(path)
        nfiles += 1
        print(f'    Added file at {path}')
    print(f'    Added {nfiles} files to TChain')
    nentries = chain.GetEntries()
    print(f'    Total entries for analysis: {nentries}')

    return nentries, nfiles, chain

def addBranch(tree, ldmx_class, branch_name) -> r.TBranch:
    """
    Parameters
    ---
    tree : r.TTree or r.TChain
        The tree(s) in which to activate the specified branch.
    ldmx_class : str
        The data type of the branch which will be accessed.
    branch_name : str
        The name of the branch to be activated.
    
    Returns
    ---
    branch
        The activated branch object.
    """
    print(f'Adding branch {branch_name} to tree {tree}')

    if tree == None:
        sys.exit('Set tree')

    if ldmx_class == 'EventHeader': branch = r.ldmx.EventHeader()
    elif ldmx_class == 'EcalVetoResult': branch = r.ldmx.EcalVetoResult()
    elif ldmx_class == 'EcalMipResult': branch = r.ldmx.EcalMipResult()
    elif ldmx_class == 'FiducialFlag':  branch = r.ldmx.FiducialFlag()
    elif ldmx_class == 'HcalVetoResult': branch = r.ldmx.HcalVetoResult()
    elif ldmx_class == 'TriggerResult': branch = r.ldmx.TriggerResult()
    elif ldmx_class == 'SimParticle': branch = r.std.map(int, 'ldmx::'+ldmx_class)()
    else: branch = r.std.vector('ldmx::'+ldmx_class)()

    tree.SetBranchStatus(branch_name + "*", 1)
    tree.SetBranchAddress(branch_name, r.AddressOf(branch))

    print(f'    Set branch address of {branch_name} to activate branch')

    return branch

def findEleEcalSPHit(ecalSPHits):
    print(f'Finding the electron hit in the Ecal scoring plane')
    eSPHit = None
    pmax = 0
    for hit in ecalSPHits:
        if hit.getPosition()[2] > ecal_front_z + sp_thickness + 0.5 or\
                hit.getMomentum()[2] <= 0 or\
                hit.getPdgID() != 11:
            # printHitInfo(hit)
            continue

        recoil_p = np.linalg.norm(hit.getMomentum())
        if recoil_p > pmax:
            eSPHit = hit
            pmax = recoil_p

    if eSPHit is not None:
        print(f'    Found a hit!')
        printHitInfo(eSPHit)
    else:
        print(f'    No hits found, returning None by default')
    
    return eSPHit

# Project poimt to z_final
def getTrajectory(pos, mom, layer_z = ecal_layerZs) -> np.ndarray[list]:
    """
    Takes in an initial position and momentum and
    calculates the final x,y positions of the electron in each
    layer of the Ecal, returning a numpy array containing xy
    ordered pairs in list format of coordinates, indexed by
    Ecal layer.
    """
    print(f'Fetching trajectory projection for initial position {pos} and momentum {mom}')
    x = pos[0] + (mom[0]/mom[2])*(layer_z - pos[2])
    y = pos[1] + (mom[1]/mom[2])*(layer_z - pos[2])
    return np.array([[x_final, y_final] for x_final, y_final in zip(x,y)])
    
def process_event(
            n_event, 
            n_skipped,
            tree, 
            ecal_sp_hits, 
            ecal_rec_hits, 
            ang_bins, 
            roc_computed_values, 
            n_entries_per_layer, 
            total_momentum,
            ele_angle,
            roc_frac, 
            layer_z = ecal_layerZs, 
            debug=True):

    """Note that n_event should be indexed from zero, as usual."""
    print(f'Processing event {n_event+1}')
    tree.GetEntry(n_event)

    # finds electron hit in the Ecal scoring plane
    ele_sp_hit = findEleEcalSPHit(ecal_sp_hits)

    # skip event if no hit found
    if ele_sp_hit is None: 
        debugPrint(f'   No electron hit found in the Ecal SP; skipping event', debug)
        n_skipped['ele_sp_hit_empty'] += 1
        return (roc_computed_values, n_entries_per_layer, n_skipped, None, None, total_momentum, ele_angle)
    
    # get kinematic variables
    recoil_p = ele_sp_hit.getMomentum()
    recoil_pos = ele_sp_hit.getPosition()
    recoil_traj = getTrajectory(recoil_pos, recoil_p) # index is layer, elements are (x,y) tuples

    # calculate binning variables
    recoil_p_tot = np.linalg.norm(recoil_p)
    recoil_ang = np.arccos(recoil_p[2]/recoil_p_tot) * (180/np.pi)
    debugPrint(f'   Event has {recoil_p_tot = }, and {recoil_ang = }', debug)

    # append binning variables to outputs
    total_momentum.append(recoil_p_tot)
    ele_angle.append(recoil_ang)

    # sort the event into a particular angle bin
    bin_index = bisect.bisect_right(ang_bins, recoil_ang) - 1 # indexed from 0, depending on angle
    if bin_index >= ( len(ang_bins) - 1 ): 
        debugPrint(f'   Recoil electron angle outside of bin range with {recoil_ang = :.3f}', debug)
        n_skipped['overflow_bin'] += 1
        return (roc_computed_values, n_entries_per_layer, n_skipped, None, None, total_momentum, ele_angle) 
    debugPrint(f'   Event has {bin_index = }', debug)

    # iterate through all hits in the event,
    # sorting them by layer into tuples of energy
    # deposition indexed by the in-layer distance 
    # from the electron trajectory
    dist_energy = defaultdict(list)
    energy_total = np.zeros(34)
    for hit in ecal_rec_hits:
        edep = hit.getEnergy()
        hit_z = round(hit.getZPos(), 3) # rounds to three decimal places
        hit_xy = [ hit.getXPos(), hit.getYPos() ]
        # debugPrint(f'    Hit has {hit_z = }', debug)
        layer_index = np.argmin(np.abs(layer_z - hit_z))
        debugPrint(f'    Hit has {layer_index = } for layer_z = {layer_z[layer_index]} = hit_z = {hit_z}', debug)

        energy_total[layer_index] += edep
        dist_ele_traj = np.linalg.norm(hit_xy - recoil_traj[layer_index]) # distance from hit to electron
        dist_energy[layer_index].append( (dist_ele_traj, edep) )
        debugPrint(f'    Hit appended to dist_energy at {layer_index = }; {dist_ele_traj = }, {edep = }')
    
    # for each layer, add up the hit energies until
    # the roc_frac * energy_total is surpassed, at which
    # point return that distance
    roc_values = np.zeros(34)
    for layer_index, dist_edep in dist_energy.items():
        if not dist_edep: continue

        # sort by distance
        dist_edep_sorted = sorted(dist_edep, key=lambda x:x[0]) 
        # debugPrint(f'    {dist_edep_sorted = }', debug)

        # calculate radius of containment for this event and this layer
        current_energy = 0
        k = 0
        debugPrint(f'    energy_total[{layer_index}] = {energy_total[layer_index]}', debug)
        
        assert np.isclose(sum([e for d,e in dist_edep_sorted]), energy_total[layer_index], rtol=1e-6), f"Layer {layer_index}: energy_total mismatch!"
        
        while current_energy < roc_frac * energy_total[layer_index]:
            current_energy += dist_edep_sorted[k][1]
            debugPrint(f'    dist_ele_traj = {dist_edep_sorted[k][0]}, edep = {dist_edep_sorted[k][1]}, ethresh = {roc_frac * energy_total[layer_index]}', debug)
            k += 1
        
        roc_values[layer_index] = dist_edep_sorted[k-1][0]
        debugPrint(f'    roc_values[{layer_index}] = {dist_edep_sorted[k-1][0]}', debug)
        n_entries_per_layer[bin_index][layer_index] += 1 # upstream event count for determining the RoC in this layer
    
    if np.all(roc_values == 0.0):
        debugPrint(f'    Distance and energy pair is missing in all hits; skipping event!', debug)
        n_skipped['dist_edep_missing'] += 1
        return (roc_computed_values, n_entries_per_layer, n_skipped, None, None, total_momentum, ele_angle) 

    for layer_i in range(len(roc_computed_values[bin_index])):
        if roc_values[layer_i] == 0.0: continue
        roc_computed_values[bin_index][layer_i].append( roc_values[layer_i] )
        debugPrint(f'    Appended RoC value {roc_values[layer_i]} to list of computed values!', debug)

    # debugPrint(f'    New roc_computed_values is:\n{roc_computed_values}', debug)

    return roc_computed_values, n_entries_per_layer, n_skipped, roc_values, bin_index, total_momentum, ele_angle

def main(debug=True):
    # load ROOT files
    file_paths = ['1e-signal10mev-8gev-100k_ecalSkim.root', '1e-signal1mev-8gev-100k_ecalSkim.root', '1e-signal100mev-8gev-100k_ecalSkim.root', '1e-signal1000mev-8gev-100k_ecalSkim.root']
    n_entries, _, tree = loadEventFiles(file_paths)

    # load relevant branches
    pass_name = 'signal'
    tree.SetBranchStatus("*", 0)
    ecal_sp_hits = addBranch(tree, 'SimTrackerHit', f'EcalScoringPlaneHits_{pass_name}')
    ecal_rec_hits = addBranch(tree, 'EcalHit', f'EcalRecHits_{pass_name}')

    # initialize arrays for calculation and binning
    ang_bins = [0, 10, 15, 25, 30, 40, 50, 60, 70] # angle bins in degrees
    num_ang_bins = len(ang_bins) - 1
    num_ecal_layers = 34
    roc_computed_values = [ [ [] for _ in range(num_ecal_layers)] for _ in range(num_ang_bins) ]
    n_entries_per_layer = np.array([np.zeros(num_ecal_layers)] * num_ang_bins)
    total_momentum = []
    ele_angle = []
    n_skipped = defaultdict(int)

    # specify fraction of energy to calculate the
    # radii of containment for
    roc_frac = 0.95

    # iterate through all events in the TChain and 
    # calculate radii of containment 
    for n_event in range(n_entries):
        # if n_event > 1000:
        #     break
        roc_computed_values, n_entries_per_layer, n_skipped, roc_values, bin_index, total_momentum, ele_angle = process_event(
            n_event,
            n_skipped,
            tree,
            ecal_sp_hits,
            ecal_rec_hits,
            ang_bins,
            roc_computed_values,
            n_entries_per_layer,
            total_momentum,
            ele_angle,
            roc_frac,
            debug=debug,
        )
        if roc_values is None: 
            debugPrint(f'    Event skipped!', debug)
        else:
            debugPrint(f'    Calculated RoCs for event {n_event+1} in bin {bin_index} are:\n{roc_values}', debug)
            debugPrint(f'    New n_entries_per_layer is:\n{n_entries_per_layer}', debug)

    # save binning variables to npy outputs
    np.save('total_momentum.npy', np.array(total_momentum))
    np.save('ele_angle.npy', np.array(ele_angle))
    # np.save('roc_computed_values.npy', np.array(roc_computed_values))
    np.save('n_entries_per_layer.npy', np.array(n_entries_per_layer))

    # create output CSV columns containing the bin edges for 
    # recoil_ang and recoil_p_tot binning
    bins_prepend = np.array([
        [float(ang_bins[i]), float(ang_bins[i+1])] for i in range(num_ang_bins)
    ])
    mom_prepend = np.array(
        [[0.0, '']] + [['', ''] for i in range(num_ang_bins-1)]
    )

    roc_computed_values = ak.Array(roc_computed_values)

    # average all the RoC values computed event-wise
    avg_roc_computed_values = np.array([
        [ak.mean(layer_hits) for layer_hits in ang_bin] 
        for ang_bin in roc_computed_values
    ])

    std_roc_computed_values = np.array([
        [ak.std(layer_hits) for layer_hits in ang_bin] 
        for ang_bin in roc_computed_values
    ])
    stderr_roc_computed_values = std_roc_computed_values / np.sqrt(n_entries_per_layer)

    # put together the output values
    # and write them to CSV
    output_values = np.concatenate((bins_prepend, mom_prepend, avg_roc_computed_values), axis=1)
    writeCSV(
        columns = ['theta_min', 'theta_max', 'p_min', 'p_max'] + list(range(1, num_ecal_layers + 1)),
        data = output_values,
        out_name = f'RoC_v14_8gev_{roc_frac}',
    )
    std_values = np.concatenate((bins_prepend, mom_prepend, std_roc_computed_values), axis=1)
    writeCSV(
        columns = ['theta_min', 'theta_max', 'p_min', 'p_max'] + list(range(1, num_ecal_layers + 1)),
        data = std_values,
        out_name = f'RoC_v14_8gev_{roc_frac}_std',
    )
    stderr_values = np.concatenate((bins_prepend, mom_prepend, stderr_roc_computed_values), axis=1)
    writeCSV(
        columns = ['theta_min', 'theta_max', 'p_min', 'p_max'] + list(range(1, num_ecal_layers + 1)),
        data = stderr_values,
        out_name = f'RoC_v14_8gev_{roc_frac}_stderr',
    )
    print(f'RoC values written to RoC_v14_8gev_{roc_frac}.csv...')
    # print(f'In total, {n_entries - np.sum(n_skipped.values())} events were used in the calculation, with {np.sum(n_skipped.values())} events skipped over for various errors')
    print(f'The profile of skipped events are these:\n{n_skipped}')

if __name__ == "__main__":
    main(debug=True)