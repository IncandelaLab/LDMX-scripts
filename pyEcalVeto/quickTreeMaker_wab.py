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
        'xmean':                     {'rtype': float, 'default': 0.},
        'weightedxmean':             {'rtype': float, 'default': 0.},
        'xdeviation':                {'rtype': float, 'default': 0.},
        'weightedxdeviation':        {'rtype': float, 'default': 0.},
        'electronPx':                     {'rtype': float, 'default': 0.},
        'electronPy':                     {'rtype': float, 'default': 0.},
        'electronPz':                     {'rtype': float, 'default': 0.},
        'nFirstLayerHit':                     {'rtype': float, 'default': 0.},
        }
 

#think about points exactly in between

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
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_v12')
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_v12')
        #proc.hcalRecHits = proc.addBranch( 'HcalHit', 'HcalRecHits_v13' )
        #proc.hcalVeto = proc.addBranch( 'HcalVetoResult', 'HcalVeto_v13' )

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
    #if not batch_mode:     # Don't want to break other batch jobs when one finishes
    #    manager.rmScratch()

    print('\nDone!\n')
def weightedSum(PEreadout,distance):
    return PEreadout/distance

def section(hit):
    SECTION_MASK = 0x7 # space for up to 7 sections                                 
    SECTION_SHIFT = 18 
    # HcalSection BACK = 0, TOP = 1, BOTTOM = 2, RIGHT = 3, LEFT = 4
    return (hit.getID() >> SECTION_SHIFT) & SECTION_MASK


# Process an event
def event_process(self):

    # Initialize BDT input variables w/ defaults
    feats = next(iter(self.tfMakers.values())).resetFeats()

    # Assign pre-computed variables
    
    ###################################
    # Determine event type
    ###################################
    feats[ 'nReadoutHits' ] = self.ecalVeto.getNReadoutHits()
    feats[ 'summedDet' ] = self.ecalVeto.getSummedDet()
    feats[ 'summedTightIso' ] = self.ecalVeto.getSummedTightIso()
    feats[ 'maxCellDep' ] = self.ecalVeto.getMaxCellDep()
    feats[ 'showerRMS' ] = self.ecalVeto.getShowerRMS()
    feats[ 'xStd' ] = self.ecalVeto.getXStd()
    feats[ 'yStd' ] = self.ecalVeto.getYStd()
    feats[ 'avgLayerHit' ] = self.ecalVeto.getAvgLayerHit()
    feats[ 'stdLayerHit' ] = self.ecalVeto.getStdLayerHit()
    feats[ 'deepestLayerHit' ] = self.ecalVeto.getDeepestLayerHit()
    feats[ 'ecalBackEnergy' ] = self.ecalVeto.getEcalBackEnergy()
    # Get e position and momentum from EcalSP
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()
    e_traj = g_traj = None
    e_trajHcal = g_trajHcal = None
    # Photon Info from targetSP
    te = True
    e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
    if e_targetHit != None:
        g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
    else:  # Should about never happen -> division by 0 in g_traj
        print('no e at targ!')
        te = False
        g_targPos = g_targP = np.zeros(3)

    # Get electron and photon trajectories
    
    
    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)

    if e_targetHit != None:
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    # Fiducial categories (filtered into different output trees)
    #if self.separate:
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


    ###################################
    # Compute extra BDT input variables
    ###################################

    # Find epSep and epDot, and prepare electron and photon trajectory vectors
    if e_traj != None and g_traj != None:

        # Create arrays marking start and end of each trajectory
        e_traj_ends = [np.array([e_traj[0][0], e_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([e_traj[-1][0], e_traj[-1][1], physTools.ecal_layerZs[-1] ])]
        g_traj_ends = [np.array([g_traj[0][0], g_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([g_traj[-1][0], g_traj[-1][1], physTools.ecal_layerZs[-1] ])]

        e_norm  = physTools.unit( e_traj_ends[1] - e_traj_ends[0] )
        g_norm  = physTools.unit( g_traj_ends[1] - g_traj_ends[0] )

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_ends   = [np.array([999 ,999 ,0   ]), np.array([999 ,999 ,999 ]) ]
        g_traj_ends   = [np.array([1000,1000,0   ]), np.array([1000,1000,1000]) ]


    # Territory setup (consider missing case)
    gToe    = physTools.unit( e_traj_ends[0] - g_traj_ends[0] )
    origin  = g_traj_ends[0] + 0.5*8.7*gToe

    # Recoil electron momentum magnitude and angle with z-axis
    recoilPMag  = physTools.mag(  e_ecalP )                 if e_ecalHit != None else -1.0
    recoilTheta = physTools.angle(e_ecalP, units='degrees') if recoilPMag > 0    else -1.0

    # Set electron RoC binnings
    e_radii = physTools.radius68_thetalt10_plt500
    if recoilTheta < 10 and recoilPMag >= 500: e_radii = physTools.radius68_thetalt10_pgt500
    elif recoilTheta >= 10 and recoilTheta < 20: e_radii = physTools.radius68_theta10to20
    elif recoilTheta >= 20: e_radii = physTools.radius68_thetagt20

    # Always use default binning for photon RoC
    g_radii = physTools.radius68_thetalt10_plt500

    # Big data
    trackingHitList = []
    ecaleWeightedE = 0
    ecalgWeightedE = 0    
    # Major ECal loop
    xmean = 0
    layer1hits = 0
    xmeanweighted = 0
    totalE = 0 
    totaloffset = 0
    totaloffsetweighted = 0
    for hit in self.ecalRecHits:
        hitE = hit.getEnergy()
        if hitE > 0:
            totalE = totalE + hitE
            layer = physTools.ecal_layer(hit)
            if layer == 0:
                layer1hits = layer1hits+1
            xy_pair = ( hit.getXPos(), hit.getYPos() )
            xmean = xmean+ xy_pair[0]
            xmeanweighted = xmean+ xy_pair[0]*hitE
            if not e_traj ==None:
                offset = xy_pair[0] - e_traj[layer][0]
                totaloffset = totaloffset+offset
                totaloffsetweighted = totaloffsetweighted+offset*hitE 

            # Distance to electron trajectory
            if e_traj != None:
                xy_e_traj = ( e_traj[layer][0], e_traj[layer][1] )
                distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
            else: distance_e_traj = -1.0

            # Distance to photon trajectory
            if g_traj != None:
                xy_g_traj = ( g_traj[layer][0], g_traj[layer][1] )
                distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
            else: distance_g_traj = -1.0
            ecaleWeightedE =ecaleWeightedE + weightedSum(hit.getEnergy(),distance_e_traj)
            ecalgWeightedE =ecalgWeightedE + weightedSum(hit.getEnergy(),distance_e_traj)

            # Decide which longitudinal segment the hit is in and add to sums
                    # Decide which containment region the hit is in and add to sums

            # Build MIP tracking hit list; (outside electron region or electron missing)
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                trackingHitList.append(hit)
    if not totalE ==0:
        xmeanweighted = xmeanweighted/totalE
        totaloffsetweighted =totaloffsetweighted/totalE 
    feats['xmean'] = xmean
    feats['weightedxmean'] =  xmeanweighted
    feats['xdeviation']= totaloffset
    feats['weightedxdeviation']= totaloffsetweighted
    if not e_traj == None:
        feats['electronPx'] = e_ecalP[0]
        feats['electronPy'] = e_ecalP[1]
        feats['electronPz'] = e_ecalP[2]
    feats['nFirstLayerHit'] = layer1hits


    #ldmx python3 quickTreeMaker.py -i $PWD/triggerSkimmedSamples/signal/4gev_v12_trigger_map_1.0_1_ldmx-det-v12_run1_seeds_2_3_1.0.root -g sig1 --out tester0507 -m 10000 




    
    
    # Fill the tree (according to fiducial category) with values for this event
    if (not self.separate) and te:
        self.tfMakers['unsorted'].fillEvent(feats)

if __name__ == "__main__":
    main()
