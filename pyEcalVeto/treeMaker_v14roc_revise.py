import os
import math
import ROOT as r
import numpy as np
from mods import ROOTmanager as manager
from mods import physTools, mipTracking
cellMap = np.loadtxt('mods/cellmodule.txt')
r.gSystem.Load('libFramework.so')

# TreeModel to build here
branches_info = {
        # Base variables
        'nReadoutHits':              {'rtype': int,   'default': 0 },
        'summedDet':                 {'rtype': float, 'default': 0.},
        'summedTightIso':            {'rtype': float, 'default': 0.},
        'maxCellDep':                {'rtype': float, 'default': 0.},
        'showerRMS':                 {'rtype': float, 'default': 0.},
        'xStd':                      {'rtype': float, 'default': 0.},
        'yStd':                      {'rtype': float, 'default': 0.},
        'avgLayerHit':               {'rtype': float, 'default': 0.},
        'stdLayerHit':               {'rtype': float, 'default': 0.},
        'deepestLayerHit':           {'rtype': int,   'default': 0 },
        'ecalBackEnergy':            {'rtype': float, 'default': 0.},
        # MIP tracking variables
        'straight4':                 {'rtype': int,   'default': 0 },
        'firstNearPhLayer':          {'rtype': int,   'default': 33},
        'nNearPhHits':               {'rtype': int,   'default': 0 },
        'fullElectronTerritoryHits': {'rtype': int,   'default': 0 },
        'fullPhotonTerritoryHits':   {'rtype': int,   'default': 0 },
        'fullTerritoryRatio':        {'rtype': float, 'default': 1.},
        'electronTerritoryHits':     {'rtype': int,   'default': 0 },
        'photonTerritoryHits':       {'rtype': int,   'default': 0 },
        'TerritoryRatio':            {'rtype': float, 'default': 1.},
        'epSep':                     {'rtype': float, 'default': 0.},
        'epDot':                     {'rtype': float, 'default': 0.},
        # Quantities needed for BDT analysis
        'isAtTSP':                   {'rtype': int,   'default': 0 },
        'isAtESP':                   {'rtype': int,   'default': 0 },
        'recoilPT':                  {'rtype': float, 'default': 0.},
        'recoilP':                   {'rtype': float, 'default': 0.},
        'electronAngle':             {'rtype': float, 'default': 0.},
        'energy1'               :     {'rtype': float, 'default': 0.},
        'distance1'             :     {'rtype': float, 'default': 0.},
        }

# RoC variables in 8 binnings, 34 layers
nbinning = 8
for i in range(1, nbinning +1):
    for j in range(1, len(physTools.ecal_layerZs) + 1):
        branches_info['roc68_binning{}_layer{}'.format(i, j)] = {'rtype': float, 'default': 0.}


def floriaSorter(arr,E):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                E[j], E[j+1] = E[j+1], E[j]
        
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return arr,E
    return arr,E        

def roc_binning(recoilPMag, recoilTheta):
    # RoC binning
    b = 0
    
    if recoilTheta < 10 and 500 <= recoilPMag < 750:
        b = 1
    elif recoilTheta < 10 and 750 <= recoilPMag < 1000:
        b = 2
    elif recoilTheta < 10 and 1000 <= recoilPMag < 1500:
        b = 3
    elif recoilTheta < 10 and recoilPMag < 500:
        b = 4
    elif 10 <= recoilTheta < 15 and recoilPMag < 1500:
        b = 5
    elif 15 <= recoilTheta < 20 and recoilPMag < 1500:
        b = 6
    elif 20 <= recoilTheta < 30 and recoilPMag < 1500:
        b = 7
    elif 30 <= recoilTheta < 60 and recoilPMag < 1500:
        b = 8
    
    return b

def printHitInfo(hit):
    print("z={}, pz={}, pid={}, trackid={}".format(hit.getPosition()[2], hit.getMomentum()[2], hit.getPdgID(), hit.getTrackID()))

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    batch_mode = pdict['batch']
    separate = pdict['separate']
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    startEvent = pdict['startEvent']
    maxEvents = pdict['maxEvents']
    # Should maybe put in parsing eventually and make event_process *arg

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels,inlist):
        procs.append( manager.TreeProcess(event_process, group,
                                          ID=gl, batch=batch_mode, pfreq=100) )

    # Process jobs
    for proc in procs:

        # Move into appropriate scratch dir
        os.chdir(proc.tmp_dir)

        # Branches needed
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_signal')
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_signal')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_signal')
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_signal')
        proc.simParticles = proc.addBranch('SimParticle','SimParticles_signal')

        # Tree/Files(s) to make
        print('\nRunning %s'%(proc.ID))

        proc.separate = separate

        proc.tfMakers = {'unsorted': None}
        if proc.separate:
            proc.tfMakers = {
                'egin': None,
                'ein': None,
                'gin': None,
                'none': None
                }

        for tfMaker in proc.tfMakers:
            proc.tfMakers[tfMaker] = manager.TreeMaker(group_labels[procs.index(proc)]+\
                                        '_{}.root'.format(tfMaker),\
                                        "EcalVeto",\
                                        branches_info,\
                                        outlist[procs.index(proc)]
                                        )

        # Gets executed at the end of run()
        proc.extrafs = [ proc.tfMakers[tfMaker].wq for tfMaker in proc.tfMakers ]

        # RUN
        proc.run(strEvent=startEvent, maxEvents=maxEvents)

    # Remove scratch directory if there is one
    if not batch_mode:     # Don't want to break other batch jobs when one finishes
        manager.rmScratch()

    print('\nDone!\n')



# Process an event
def event_process(self):


    # Initialize BDT input variables w/ defaults
    feats = next(iter(self.tfMakers.values())).resetFeats()

    # Assign pre-computed variables 
    # variables/branches already calculated in the tree
    feats['nReadoutHits']       = self.ecalVeto.getNReadoutHits()
    feats['summedDet']          = self.ecalVeto.getSummedDet()
    feats['summedTightIso']     = self.ecalVeto.getSummedTightIso()
    feats['maxCellDep']         = self.ecalVeto.getMaxCellDep()
    feats['showerRMS']          = self.ecalVeto.getShowerRMS()
    feats['xStd']               = self.ecalVeto.getXStd()
    feats['yStd']               = self.ecalVeto.getYStd()
    feats['avgLayerHit']        = self.ecalVeto.getAvgLayerHit()
    feats['stdLayerHit']        = self.ecalVeto.getStdLayerHit()
    feats['deepestLayerHit']    = self.ecalVeto.getDeepestLayerHit() 
    feats['ecalBackEnergy']     = self.ecalVeto.getEcalBackEnergy()
    
    ###################################
    # Determine event type
    ###################################

    # Get e position and momentum from EcalSP
    
    # test
    # for hit in self.ecalSPHits:
    #     printHitInfo(hit)
    
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    # test
    # print("e_ecalHit: ")
    # if e_ecalHit != None:
    #     printHitInfo(e_ecalHit)
    
    if e_ecalHit != None:
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()

    # Get electron and photon trajectories
    e_traj = g_traj = None

    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)
    #print(e_traj)
    
    # print("e_ecalPos = ",e_ecalPos)
    # print("e_ecalP = ",e_ecalP)

    # Fiducial categories (filtered into different output trees)
    if self.separate:
        e_fid = g_fid = False

        if e_traj != None:
            for cell in cellMap:
                if physTools.dist( cell[1:], e_traj[0] ) <= physTools.cell_radius:
                    e_fid = True
                    break

        if g_traj != None:
            for cell in cellMap:
                if physTools.dist( cell[1:], g_traj[0] ) <= physTools.cell_radius:
                    g_fid = True
                    break
    
    # test
    # print("e_traj = ", e_traj)
    # print("g_traj = ", g_traj)
    
    ###################################
    # Compute extra RoC variables
    ###################################

    # Recoil electron momentum magnitude and angle with z-axis                   # !!!!!!!!!!!!!!!!
    recoilPMag  = physTools.mag(  e_ecalP )                 if e_ecalHit != None else -1.0
    recoilTheta = physTools.angle(e_ecalP, units='degrees') if recoilPMag > 0    else -1.0
    
    # print("recoil electron p = {}, theta = {}".format(recoilPMag, recoilTheta))
    

    feats['recoilP'] = recoilPMag
    feats['electronAngle'] = recoilTheta
    
    # Find the RoC binning of this event
    binning = roc_binning(recoilPMag, recoilTheta)
    # print("binning = ", binning)
    
    for b in range(nbinning):
        
        if binning == b + 1:     # binning = 1, 2, 3, 4; b = 0, 1, 2, 3
        
            # Initiate distance-energy tuples
            dist_energy = {}
            for l in range(34):
                dist_energy[l] = []

            E_tot = np.zeros(34)
            
            if e_traj != None:
                
                # Major ECal loop
                for hit in self.ecalRecHits:
                    
                    if hit.getEnergy() > 0:
                        
                        layer = physTools.ecal_layer(hit)
                        xy_pair = ( hit.getXPos(), hit.getYPos() )
                        
                        for l in range(34):
                            if layer == l:   # layer = 1, 2, ..., 34; l = 0, 1, ..., 33
                                E_tot[l] += hit.getEnergy()
                                xy_e_traj = ( e_traj[l][0], e_traj[l][1] )
                                dist_energy[l].append( (physTools.dist(xy_pair, xy_e_traj), hit.getEnergy()) )    # (distance, energy) tuple
                                
            
            # Calculate RoC
            roc_frac = 0.68    # 68% energy containment
            rocComputed = -10 * np.ones(34)

            for l in range(34):
                # print("layer", l)
                print(dist_energy[l])
                
                if dist_energy[l]:
                    # Sort the distance-energy tuples according to distance
                    dist_energy_sorted = sorted(dist_energy[l], key=lambda x:x[0])
                    # print(dist_energy_sorted)
                    currentE = 0
                    k = 0
                    while currentE < roc_frac * E_tot[l]:
                        currentE += dist_energy_sorted[k][1]
                        k += 1
                    
                    rocComputed[l] = dist_energy_sorted[k-1][0]
                    
                    # Save the RoC variables
                    feats['roc68_binning{}_layer{}'.format(binning, l+1)] = rocComputed[l]

       
    # Fill the tree (according to fiducial category) with values for this event
    if not self.separate:
        self.tfMakers['unsorted'].fillEvent(feats)
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)
# print(layer)
if __name__ == "__main__":
    main()