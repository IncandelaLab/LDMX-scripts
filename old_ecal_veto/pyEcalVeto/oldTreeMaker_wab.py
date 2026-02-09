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
        'epSep':                    {'rtype': float, 'default': 0.},
        'epDot':                    {'rtype': float, 'default': 0.}
        }

for i in range(1, physTools.nSegments + 1):

    # Longitudinal segment variables
    branches_info['energy_s{}'.format(i)]          = {'rtype': float, 'default': 0.}
    branches_info['nHits_s{}'.format(i)]           = {'rtype': int,   'default': 0 }
    branches_info['xMean_s{}'.format(i)]           = {'rtype': float, 'default': 0.}
    branches_info['yMean_s{}'.format(i)]           = {'rtype': float, 'default': 0.}
    branches_info['layerMean_s{}'.format(i)]       = {'rtype': float, 'default': 0.}
    branches_info['xStd_s{}'.format(i)]            = {'rtype': float, 'default': 0.}
    branches_info['yStd_s{}'.format(i)]            = {'rtype': float, 'default': 0.}
    branches_info['layerStd_s{}'.format(i)]        = {'rtype': float, 'default': 0.}

    for j in range(1, physTools.nRegions + 1):

        # Electron RoC variables
        branches_info['eContEnergy_x{}_s{}'.format(j,i)]    = {'rtype': float, 'default': 0.}
        branches_info['eContNHits_x{}_s{}'.format(j,i)]     = {'rtype': int,   'default': 0 }
        branches_info['eContXMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['eContYMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['eContLayerMean_x{}_s{}'.format(j,i)] = {'rtype': float, 'default': 0.}
        branches_info['eContXStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['eContYStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['eContLayerStd_x{}_s{}'.format(j,i)]  = {'rtype': float, 'default': 0.}

        # Photon RoC variables
        branches_info['gContEnergy_x{}_s{}'.format(j,i)]    = {'rtype': float, 'default': 0.}
        branches_info['gContNHits_x{}_s{}'.format(j,i)]     = {'rtype': int,   'default': 0 }
        branches_info['gContXMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['gContYMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['gContLayerMean_x{}_s{}'.format(j,i)] = {'rtype': float, 'default': 0.}
        branches_info['gContXStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['gContYStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['gContLayerStd_x{}_s{}'.format(j,i)]  = {'rtype': float, 'default': 0.}

        # Outside RoC variables
        branches_info['oContEnergy_x{}_s{}'.format(j,i)]    = {'rtype': float, 'default': 0.}
        branches_info['oContNHits_x{}_s{}'.format(j,i)]     = {'rtype': int,   'default': 0 }
        branches_info['oContXMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['oContYMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['oContLayerMean_x{}_s{}'.format(j,i)] = {'rtype': float, 'default': 0.}
        branches_info['oContXStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['oContYStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['oContLayerStd_x{}_s{}'.format(j,i)]  = {'rtype': float, 'default': 0.}

branches_info['ecalFid']= {     # 0 for nonfiducial with no photon, 1 for nonfiducial with photon, 2 for fiducial without photon, 3 for having everything
    'rtype': int,
    'default': 0.
    }
branches_info['numberNoiseHit']= {
    'rtype': int,
    'default': 0.
    }
branches_info['energyNoiseHit']= {
    'rtype': float,
    'default': 0.
    }
branches_info['eTheta']= {
    'rtype': float,
    'default': 0.
    }
branches_info['gTheta']= {
    'rtype': float,
    'default': 0.
    }
branches_info['egTheta']= {
    'rtype': float,
    'default': 0.
    }
branches_info['targetElectron']= {
    'rtype': int,
    'default':0.5
    }
branches_info['avglayerNoiseHit']= {
    'rtype': float,
    'default': 0.
    }
branches_info['ePz']= {
    'rtype': float,
    'default':0.5
    }
branches_info['eP']= {
    'rtype': float,
    'default':0.5
    }
branches_info['hcalScore']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['epAng']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['ECloseToPhoton']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['ECloseToElectron']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['nHitsCloseToPhoton']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['nHitsCloseToElectron']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['PECloseToPhoton']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['PECloseToElectron']= {
    'rtype': float,
    'default':-0.5
    }
branches_info['n5PECloseToPhoton']= { #number of acceptable hits (greater than 5PE) that is closer to photon trajectory
    'rtype': float,
    'default':-0.5
    }
branches_info['n5PECloseToElectron']= { #number of acceptable hits (greater than 5PE) that is closer to electron trajectory
    'rtype': float,
    'default':-0.5
    }
branches_info['acceptedPECloseToPhoton']= { #number of acceptable PE (from events with greater than 5PE) that is closer to photon trajectory
    'rtype': float,
    'default':-0.5
    }
branches_info['acceptedPECloseToElectron']= { #number of acceptable PE (from events with greater than 5PE) that is closer to photon trajectory
    'rtype': float,
    'default':-0.5
    }




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
        proc.hcalVeto = proc.addBranch( 'HcalVetoResult', 'HcalVeto_v12')
        proc.hcalRecHits = proc.addBranch( 'HcalHit', 'HcalRecHits_v12' )

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


# Process an event
def event_process(self):

    # Initialize BDT input variables w/ defaults
    feats = next(iter(self.tfMakers.values())).resetFeats()
    
    # Assign pre-computed variables
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
    feats['hcalScore'] = self.hcalVeto.passesVeto()
    ###################################
    # Determine event type
    ###################################

    # Get e position and momentum from EcalSP
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()

    # Photon Info from targetSP
    e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
    if e_targetHit != None:
        g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
        eP = e_targetHit.getMomentum()
        feats['ePz'] = eP[2]
        feats['eP'] = (eP[1]**2 + eP[2]**2+eP[0]**2)**0.5
        # Extra analysis info
        feats[ 'eTheta' ] = physTools.angle(
            eP, units='degrees'
            )
        temp= physTools.angle( g_targP, units='degrees' )
        feats[ 'gTheta' ] = temp
        feats[ 'egTheta' ] = physTools.angle(
        g_targP, units='degrees', vec2=e_targetHit.getMomentum())
        feats['targetElectron'] = 1
    else:  # Should about never happen -> division by 0 in g_traj
        print('no e at targ!')
        g_targPos = g_targP = np.zeros(3)
        feats[ 'eTheta' ] = -1
        feats[ 'gTheta' ] = -1
        feats[ 'egTheta' ] = -1
        feats['ePz'] = -1
        feats['eP'] = -1
        feats['targetElectron'] = 0

    # Get electron and photon trajectories
    e_traj = g_traj = None

    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)
        

    if e_targetHit != None:
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    # Fiducial categories (filtered into different output trees)
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
        g_traj_endsOld = [np.array([g_trajOld[0][0], g_trajOld[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([g_trajOld[-1][0], g_trajOld[-1][1], physTools.ecal_layerZs[-1] ])]

        # Unused epDot and epSep
        e_norm  = physTools.unit( e_traj_ends[1] - e_traj_ends[0] )
        g_norm  = physTools.unit( g_traj_ends[1] - g_traj_ends[0] )
        g_normOld  = physTools.unit( g_traj_endsOld[1] - g_traj_endsOld[0] )
        feats['epSep'] = physTools.dist( e_traj_ends[0], g_traj_ends[0] )
        feats['epDot'] = physTools.dot(e_norm,g_norm)
        feats['epAng'] = math.acos(physTools.dot(e_norm,g_norm)) * 180.0 / math.pi



    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_ends   = [np.array([999 ,999 ,0   ]), np.array([999 ,999 ,999 ]) ]
        g_traj_ends   = [np.array([1000,1000,0   ]), np.array([1000,1000,1000]) ]

        feats['epSep'] = 10.0 + 1.0 # Don't cut on these in this case
        feats['epDot'] = 3.0 + 1.0
        feats['epAng'] = 3.0 + 1.0
        
        

    # Territory setup (consider missing case)
    gToe    = physTools.unit( e_traj_ends[0] - g_traj_ends[0] )
    origin  = g_traj_ends[0] + 0.5*8.7*gToe

    # Recoil electron momentum magnitude and angle with z-axis
    recoilPMag  = physTools.mag(  e_ecalP )                 if e_ecalHit != None else -1.0
    recoilTheta = physTools.angle(e_ecalP, units='radians') if recoilPMag > 0    else -1.0

    # Set electron RoC binnings
    e_radii = physTools.radius68_thetalt10_plt500
    if recoilTheta < 10 and recoilPMag >= 500: e_radii = physTools.radius68_thetalt10_pgt500
    elif recoilTheta >= 10 and recoilTheta < 20: e_radii = physTools.radius68_theta10to20
    elif recoilTheta >= 20: e_radii = physTools.radius68_thetagt20

    # Always use default binning for photon RoC
    g_radii = physTools.radius68_thetalt10_plt500

    # Big data
    trackingHitList = []

    # Major ECal loop
    for hit in self.ecalRecHits:
        
        if hit.getEnergy() > 0:

            layer = physTools.ecal_layer(hit)
            xy_pair = ( hit.getXPos(), hit.getYPos() )

            # Territory selections
            hitPrime = physTools.pos(hit) - origin
            if np.dot(hitPrime, gToe) > 0: feats['fullElectronTerritoryHits'] += 1
            else: feats['fullPhotonTerritoryHits'] += 1

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

            # Decide which longitudinal segment the hit is in and add to sums
            for i in range(1, physTools.nSegments + 1):

                if (physTools.segLayers[i - 1] <= layer)\
                  and (layer <= physTools.segLayers[i] - 1):
                    feats['energy_s{}'.format(i)] += hit.getEnergy()
                    feats['nHits_s{}'.format(i)] += 1
                    feats['xMean_s{}'.format(i)] += xy_pair[0]*hit.getEnergy()
                    feats['yMean_s{}'.format(i)] += xy_pair[1]*hit.getEnergy()
                    feats['layerMean_s{}'.format(i)] += layer*hit.getEnergy()

                    # Decide which containment region the hit is in and add to sums
                    for j in range(1, physTools.nRegions + 1):

                        if ((j - 1)*e_radii[layer] <= distance_e_traj)\
                          and (distance_e_traj < j*e_radii[layer]):
                            feats['eContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['eContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['eContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['eContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['eContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        if ((j - 1)*g_radii[layer] <= distance_g_traj)\
                          and (distance_g_traj < j*g_radii[layer]):
                            feats['gContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['gContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['gContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['gContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['gContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        if (distance_e_traj > j*e_radii[layer])\
                          and (distance_g_traj > j*g_radii[layer]):
                            feats['oContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['oContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['oContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['oContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['oContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

            # Build MIP tracking hit list; (outside electron region or electron missing)
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                trackingHitList.append(hit) 

    # If possible, quotient out the total energy from the means
    for i in range(1, physTools.nSegments + 1):

        if feats['energy_s{}'.format(i)] > 0:
            feats['xMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]
            feats['yMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]
            feats['layerMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]

        for j in range(1, physTools.nRegions + 1):

            if feats['eContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['eContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]
                feats['eContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]
                feats['eContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]

            if feats['gContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['gContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]
                feats['gContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]
                feats['gContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]

            if feats['oContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['oContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]
                feats['oContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]
                feats['oContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]

    # Loop over hits again to calculate the standard deviations
    noiseE = 0
    noiseHit = 0
    avgnh = 0
    for hit in self.ecalRecHits:
        layer = physTools.ecal_layer(hit)
        if hit.isNoise():
            noiseE = noiseE+hit.getEnergy()
            noiseHit = noiseHit + 1
            avgnh =avgnh + layer
        
        xy_pair = (hit.getXPos(), hit.getYPos())

        # Distance to electron trajectory
        if e_traj != None:
            xy_e_traj = (e_traj[layer][0], e_traj[layer][1])
            distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
        else:
            distance_e_traj = -1.0

        # Distance to photon trajectory
        if g_traj != None:
            xy_g_traj = (g_traj[layer][0], g_traj[layer][1])
            distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
        else:
            distance_g_traj = -1.0

        # Decide which longitudinal segment the hit is in and add to sums
        for i in range(1, physTools.nSegments + 1):

            if (physTools.segLayers[i - 1] <= layer) and\
                    (layer <= physTools.segLayers[i] - 1):
                feats['xStd_s{}'.format(i)] += ((xy_pair[0] -\
                        feats['xMean_s{}'.format(i)])**2)*hit.getEnergy()
                feats['yStd_s{}'.format(i)] += ((xy_pair[1] -\
                        feats['yMean_s{}'.format(i)])**2)*hit.getEnergy()
                feats['layerStd_s{}'.format(i)] += ((layer -\
                        feats['layerMean_s{}'.format(i)])**2)*hit.getEnergy()

                # Decide which containment region the hit is in and add to sums
                for j in range(1, physTools.nRegions + 1):

                    if ((j - 1)*e_radii[layer] <= distance_e_traj)\
                      and (distance_e_traj < j*e_radii[layer]):
                        feats['eContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['eContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['eContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['eContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['eContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['eContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

                    if ((j - 1)*g_radii[layer] <= distance_g_traj)\
                      and (distance_g_traj < j*g_radii[layer]):
                        feats['gContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['gContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['gContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['gContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['gContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['gContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

                    if (distance_e_traj > j*e_radii[layer])\
                      and (distance_g_traj > j*g_radii[layer]):
                        feats['oContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['oContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['oContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['oContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['oContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['oContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
    feats['numberNoiseHit'] = noiseE
    feats['energyNoiseHit'] = noiseHit
    if noiseHit>0:
        avgnh =avgnh/noiseHit
    else:
        if avgnh>0:
            print('error!')
    feats['avglayerNoiseHit'] =avgnh
    # Quotient out the total energies from the standard deviations if possible and take root
    for i in range(1, physTools.nSegments + 1):

        if feats['energy_s{}'.format(i)] > 0:
            feats['xStd_s{}'.format(i)] = math.sqrt(feats['xStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])
            feats['yStd_s{}'.format(i)] = math.sqrt(feats['yStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])
            feats['layerStd_s{}'.format(i)] = math.sqrt(feats['layerStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])

        for j in range(1, physTools.nRegions + 1):

            if feats['eContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['eContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContXStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])
                feats['eContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContYStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])
                feats['eContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])

            if feats['gContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['gContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContXStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])
                feats['gContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContYStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])
                feats['gContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])

            if feats['oContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['oContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContXStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])
                feats['oContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContYStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])
                feats['oContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])


    # Find the first layer of the ECal where a hit near the projected photon trajectory
    # AND the total number of hits around the photon trajectory
    if g_traj != None: # If no photon trajectory, leave this at the default

        # First currently unusued; pending further study; performance drop from  v9 and v12
        #print(trackingHitList, g_traj)
        feats['firstNearPhLayer'], feats['nNearPhHits'] = mipTracking.nearPhotonInfo(
                                                            trackingHitList, g_traj )
    else: feats['nNearPhHits'] = feats['nReadoutHits']


    # Territories limited to trackingHitList
    if e_traj != None:
        for hit in trackingHitList:
            hitPrime = physTools.pos(hit) - origin
            if np.dot(hitPrime, gToe) > 0: feats['electronTerritoryHits'] += 1
            else: feats['photonTerritoryHits'] += 1
    else:
        feats['photonTerritoryHits'] = feats['nReadoutHits']
        feats['TerritoryRatio'] = 10
        feats['fullTerritoryRatio'] = 10
    if feats['electronTerritoryHits'] != 0:
        feats['TerritoryRatio'] = feats['photonTerritoryHits']/feats['electronTerritoryHits']
    if feats['fullElectronTerritoryHits'] != 0:
        feats['fullTerritoryRatio'] = feats['fullPhotonTerritoryHits']/\
                                            feats['fullElectronTerritoryHits']


    # Find MIP tracks
    feats['straight4'], trackingHitList = mipTracking.findStraightTracks(
                                trackingHitList, e_traj_ends, g_traj_ends,
                                mst = 4, returnHitList = True)
    for hit in self.hcalRecHits:
        #print(str(hit.getLayer())+" , "+str(hit.getStrip()))
        x = hit.getXPos()
        y = hit.getXPos()

    # Fill the tree (according to fiducial category) with values for this event
    if not self.separate:
        if e_fid and g_fid: feats['ecalFid'] = 3
        elif e_fid and not g_fid: feats['ecalFid'] = 2
        elif not e_fid and g_fid: feats['ecalFid'] = 1
        else: feats['ecalFid'] = 0
        self.tfMakers['unsorted'].fillEvent(feats)
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)

if __name__ == "__main__":
    main()
