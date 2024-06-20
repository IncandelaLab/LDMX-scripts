import os
import math
import ROOT as r
import numpy as np
from mods import ROOTmanager as manager
from mods import physTools_wab, mipTracking
cellMap = np.loadtxt( 'mods/cellmodule.txt' )
#np.loadtxt('/nfs/slac/g/ldmx/users/aechavez/ldmx-sw-v3.0.0/LDMX-scripts/pyEcalVeto/mods/cellmodule.txt')
r.gSystem.Load(
    '/home/billy/ultimateLDMX/ldmx-sw/install/lib/libFramework.so'
    )
#r.gSystem.Load('/nfs/slac/g/ldmx/users/aechavez/ldmx-sw-v3.0.0/ldmx-sw/install/lib/libFramework.so')

# Tree model for ECal information only
branches_info = {
        # Event number
        'eventNumber'              : {'rtype': int,   'default': 0 },
        # Electron kinematics
        'electronESPXPos'          : {'rtype': float, 'default': 0.},
        'electronESPYPos'          : {'rtype': float, 'default': 0.},
        'electronESPZPos'          : {'rtype': float, 'default': 0.},
        'electronESPXMom'          : {'rtype': float, 'default': 0.},
        'electronESPYMom'          : {'rtype': float, 'default': 0.},
        'electronESPZMom'          : {'rtype': float, 'default': 0.},
        'electronESPMagMom'        : {'rtype': float, 'default': 0.},
        'electronESPThetaMom'      : {'rtype': float, 'default': 0.},
        'electronTSPXPos'          : {'rtype': float, 'default': 0.},
        'electronTSPYPos'          : {'rtype': float, 'default': 0.},
        'electronTSPZPos'          : {'rtype': float, 'default': 0.},
        'electronTSPXMom'          : {'rtype': float, 'default': 0.},
        'electronTSPYMom'          : {'rtype': float, 'default': 0.},
        'electronTSPZMom'          : {'rtype': float, 'default': 0.},
        'electronTSPMagMom'        : {'rtype': float, 'default': 0.},
        'electronTSPThetaMom'      : {'rtype': float, 'default': 0.},
        'isAtESP'                  : {'rtype': int,   'default': 0 },
        'isAtTSP'                  : {'rtype': int,   'default': 0 },
        'isFiducial'               : {'rtype': int,   'default': 0 },
        # Inferred photon kinematics
        'photonESPXPos'            : {'rtype': float, 'default': 0.},
        'photonESPYPos'            : {'rtype': float, 'default': 0.},
        'photonESPZPos'            : {'rtype': float, 'default': 0.},
        'photonTSPXPos'            : {'rtype': float, 'default': 0.},
        'photonTSPYPos'            : {'rtype': float, 'default': 0.},
        'photonTSPZPos'            : {'rtype': float, 'default': 0.},
        'photonXMom'               : {'rtype': float, 'default': 0.},
        'photonYMom'               : {'rtype': float, 'default': 0.},
        'photonZMom'               : {'rtype': float, 'default': 0.},
        'photonMagMom'             : {'rtype': float, 'default': 0.},
        'photonThetaMom'           : {'rtype': float, 'default': 0.},
        # Rec hit information
        'totalRecAmplitude'        : {'rtype': float, 'default': 0.},
        'totalRecEnergy'           : {'rtype': float, 'default': 0.},
        'nRecHits'                 : {'rtype': int,   'default': 0 },
        # Sim hit information
        'totalSimEDep'             : {'rtype': float, 'default': 0.},
        'nSimHits'                 : {'rtype': int,   'default': 0 },
        # Noise information
        'totalNoiseEnergy'         : {'rtype': float, 'default': 0.},
        'nNoiseHits'               : {'rtype': int,   'default': 0 },
        # Base Fernand variables
        'nReadoutHits'             : {'rtype': int,   'default': 0 },
        'summedDet'                : {'rtype': float, 'default': 0.},
        'summedTightIso'           : {'rtype': float, 'default': 0.},
        'maxCellDep'               : {'rtype': float, 'default': 0.},
        'showerRMS'                : {'rtype': float, 'default': 0.},
        'xStd'                     : {'rtype': float, 'default': 0.},
        'yStd'                     : {'rtype': float, 'default': 0.},
        'avgLayerHit'              : {'rtype': float, 'default': 0.},
        'stdLayerHit'              : {'rtype': float, 'default': 0.},
        'deepestLayerHit'          : {'rtype': int,   'default': 0 },
        'ecalBackEnergy'           : {'rtype': float, 'default': 0.},
        # MIP tracking variables
        'straight4'                : {'rtype': int,   'default': 0 },
        'firstNearPhLayer'         : {'rtype': int,   'default': 33},
        'nNearPhHits'              : {'rtype': int,   'default': 0 },
        'fullElectronTerritoryHits': {'rtype': int,   'default': 0 },
        'fullPhotonTerritoryHits'  : {'rtype': int,   'default': 0 },
        'fullTerritoryRatio'       : {'rtype': float, 'default': 1.},
        'electronTerritoryHits'    : {'rtype': int,   'default': 0 },
        'photonTerritoryHits'      : {'rtype': int,   'default': 0 },
        'TerritoryRatio'           : {'rtype': float, 'default': 1.},
        'epSep'                    : {'rtype': float, 'default': 0.},
        'epDot'                    : {'rtype': float, 'default': 0.}
}

for i in range(1, physTools.nRegions + 1):

    # Electron RoC variables
    branches_info['electronContainmentEnergy_x{}'.format(i)] = {'rtype': float, 'default': 0.}

    # Photon RoC variables
    branches_info['photonContainmentEnergy_x{}'.format(i)]   = {'rtype': float, 'default': 0.}

    # Outside RoC variables
    branches_info['outsideContainmentEnergy_x{}'.format(i)]  = {'rtype': float, 'default': 0.}
    branches_info['outsideContainmentNHits_x{}'.format(i)]   = {'rtype': int,   'default': 0 }
    branches_info['outsideContainmentXStd_x{}'.format(i)]    = {'rtype': float, 'default': 0.}
    branches_info['outsideContainmentYStd_x{}'.format(i)]    = {'rtype': float, 'default': 0.}

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

# Tree models for ECal and hit-by-hit/particle-by-particle information
recHitBranches = {
    'amplitude'      : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'energy'         : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'xPos'           : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'yPos'           : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'zPos'           : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'matchingSimEDep': {'address': np.zeros(1, dtype = float), 'rtype': float}
}
for varName in branches_info:
    recHitBranches[varName] = {'address': np.zeros(1, dtype = branches_info[varName]['rtype']),
                               'rtype': branches_info[varName]['rtype']}

simHitBranches = {
    'eDep': {'address': np.zeros(1, dtype = float), 'rtype': float},
    'xPos': {'address': np.zeros(1, dtype = float), 'rtype': float},
    'yPos': {'address': np.zeros(1, dtype = float), 'rtype': float},
    'zPos': {'address': np.zeros(1, dtype = float), 'rtype': float}
}
for varName in branches_info:
    simHitBranches[varName] = {'address': np.zeros(1, dtype = branches_info[varName]['rtype']),
                               'rtype': branches_info[varName]['rtype']}

simParticleBranches = {
    'energy'    : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'trackID'   : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'pdgID'     : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'vertXPos'  : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'vertYPos'  : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'vertZPos'  : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'endXPos'   : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'endYPos'   : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'endZPos'   : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'xMom'      : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'yMom'      : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'zMom'      : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'mass'      : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'charge'    : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'nDaughters': {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'nParents'  : {'address': np.zeros(1, dtype = int  ), 'rtype': int  }
}
# HcalVeto
branches_info['maxPE']= {
    'rtype': int,
    'default': 0.5
    }
branches_info['topmaxPE']= {
    'rtype': int,
    'default': 0.5
    }
branches_info['bottommaxPE']= {
    'rtype': int,
    'default': 0.5
    }
branches_info['rightmaxPE']= {
    'rtype': int,
    'default': 0.5
    }
branches_info['leftmaxPE']= {
    'rtype': int,
    'default': 0.5
    }
branches_info['backmaxPE']= {
    'rtype': int,
    'default': 0.5
    }
branches_info['sidemaxPE']= {
    'rtype': int,
    'default':0.5
    }
# Extra Momentum and Thetale Info for analysis only (not trained on)
branches_info['recoilPT']= {
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
branches_info['ecalSPHits']= {
    'rtype': int,
    'default':0.5
    }
branches_info['ecalSPEnergy']= {
    'rtype': int,
    'default':0.5
    }
branches_info['targetSPHits']= {
    'rtype': float,
    'default':0.5
    }
branches_info['targetSPEnergy']= { #Total energy on the target scoring plane
    'rtype': float,
    'default':0.5
    }
branches_info['targetElectron']= { #If the there really is an electron on the most downstream target scoring plane 
    'rtype': int,
    'default':0.5
    }
branches_info['gPx']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gPy']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gPz']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gx']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gy']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gz']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gPMagnitude']= {
    'rtype': float,
    'default':0.5
    }
branches_info['gThetaNew']= {
    'rtype': float,
    'default':0.5
    }
branches_info['egThetaNew']= {
    'rtype': float,
    'default': 0.
    }
branches_info['ePz']= {     #The ZComponent of the recoil electron Momentum
    'rtype': float,
    'default': 0.
    }
branches_info['eP']= {   #The magnitude of the recoil electron momentum
    'rtype': float,
    'default': 0.
    }
branches_info['ex']= {   #The XComponent of the recoil electron hit on the target SP
    'rtype': float,
    'default': 0.
    }
branches_info['ey']= {   #The YComponent of the recoil electron hit on the target SP
    'rtype': float,
    'default': 0.
    }
branches_info['ez']= {   #The ZComponent of the recoil electron hit on the target SP
    'rtype': float,
    'default': 0.
    }                            
branches_info['downSPReached']= {  #0 corresponding to hard brem on track2                     This branch is purely purposed to check if the event makes sense or not.
    'rtype': int,                  #-1 corresponding to hard brem not reaching most downstream target scoring plane 
    'default': 0.                  #-2 corresponding to no brem at most downstream target scoring plane at all
    }                              #>0 corresponding to track2 particle is not a photon
branches_info['hcalScore']= {
    'rtype': float,
    'default': 0.
    }     
branches_info['ecalScore']= {
    'rtype': float,
    'default': 0.
    }       
branches_info['ecalFid']= {     # 0 for nonfiducial with no photon, 1 for nonfiducial with photon, 2 for fiducial without photon, 3 for having everything
    'rtype': int,
    'default': 0.
    }
#These recoil branches checks how deep the electron penetrates to
branches_info['recoilLayer1Ne']= {  #recoil first layer number of electron 
    'rtype': int,
    'default': 0.
    }
branches_info['recoilLayer1NeAdjusted']= {  #recoil first layer number of electron, but adjusted so that we do not consider a backward going electron
    'rtype': int,
    'default': 0.
    }
branches_info['recoilTotalNe']= {  #recoil first layer number of electron 
    'rtype': int,
    'default': 0.
    }
branches_info['recoilLayer1E']= {  #recoil first layer number of electron 
    'rtype': float,
    'default': 0.
    }
branches_info['recoilTotalE']= {  #recoil first layer number of electron 
    'rtype': float,
    'default': 0.
    }
branches_info['recoilTheta2L']= {  #recoil first layer number of electron 
    'rtype': float,
    'default': 0.
    }
branches_info['recoilTheta3L']= {  #recoil first layer number of electron 
    'rtype': float,
    'default': 0.
    }
branches_info['recoilDeepestLayerHit']= {  #recoil first layer number of electron 
    'rtype': float,
    'default': 0.
    }  
for varName in branches_info:
    simParticleBranches[varName] = {'address': np.zeros(1, dtype = branches_info[varName]['rtype']),
                                    'rtype': branches_info[varName]['rtype']}

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

        '''# Branches needed
        proc.eventHeader  = proc.addBranch('EventHeader', 'EventHeader')
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_v12')
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_v12')
        proc.ecalSimHits  = proc.addBranch('SimCalorimeterHit', 'EcalSimHits_v12')
        proc.simParticles = proc.addBranch('SimParticle', 'SimParticles_v12')
        '''# Branches needed
        proc.eventHeader  = proc.addBranch('EventHeader', 'EventHeader')
        proc.ecalVeto = proc.addBranch( 'EcalVetoResult', 'EcalVeto_v12' )
        proc.hcalVeto = proc.addBranch( 'HcalVetoResult', 'HcalVeto_v12' )
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalRecHits = proc.addBranch( 'EcalHit', 'EcalRecHits_v12' )
        proc.hcalRecHits = proc.addBranch( 'HcalHit', 'HcalRecHits_v12' )
        proc.simParticles = proc.addBranch('SimParticle', 'SimParticles_v12')
        proc.ecalSimHits  = proc.addBranch('SimCalorimeterHit', 'EcalSimHits_v12')
        proc.RecoilSimHits = proc.addBranch('SimTrackerHit', 'RecoilSimHits_v12')
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
                                        "EcalInfo",\
                                        branches_info,\
                                        outlist[procs.index(proc)]
                                        )

        # Tree for rec hit information
        proc.recHitInfo = r.TTree('RecHitInfo', 'Rec hit information')

        for branch in recHitBranches:

            if str(recHitBranches[branch]['rtype']) == "<class 'float'>"\
              or str(recHitBranches[branch]['rtype']) == "<type 'float'>":
                proc.recHitInfo.Branch(branch, recHitBranches[branch]['address'], branch + '/D')

            elif str(recHitBranches[branch]['rtype']) == "<class 'int'>"\
              or str(recHitBranches[branch]['rtype']) == "<type 'int'>":
                proc.recHitInfo.Branch(branch, recHitBranches[branch]['address'], branch + '/I')

        # Tree for sim hit information
        proc.simHitInfo = r.TTree('SimHitInfo', 'Sim hit information')

        for branch in simHitBranches:

            if str(simHitBranches[branch]['rtype']) == "<class 'float'>"\
              or str(simHitBranches[branch]['rtype']) == "<type 'float'>":
                proc.simHitInfo.Branch(branch, simHitBranches[branch]['address'], branch + '/D')

            elif str(simHitBranches[branch]['rtype']) == "<class 'int'>"\
              or str(simHitBranches[branch]['rtype']) == "<type 'int'>":
                proc.simHitInfo.Branch(branch, simHitBranches[branch]['address'], branch + '/I')

        # Tree for sim particle information
        proc.simParticleInfo = r.TTree('SimParticleInfo', 'Sim particle information')

        for branch in simParticleBranches:

            if str(simParticleBranches[branch]['rtype']) == "<class 'float'>"\
              or str(simParticleBranches[branch]['rtype']) == "<type 'float'>":
                proc.simParticleInfo.Branch(branch, simParticleBranches[branch]['address'], branch + '/D')

            elif str(simParticleBranches[branch]['rtype']) == "<class 'int'>"\
              or str(simParticleBranches[branch]['rtype']) == "<type 'int'>":
                proc.simParticleInfo.Branch(branch, simParticleBranches[branch]['address'], branch + '/I')

        # Gets executed at the end of run()
        proc.extrafs = []
        for tfMaker in proc.tfMakers:
            proc.extrafs.append(proc.recHitInfo.Write)
            proc.extrafs.append(proc.simHitInfo.Write)
            proc.extrafs.append(proc.simParticleInfo.Write)
        for tfMaker in proc.tfMakers:
            proc.extrafs.append(proc.tfMakers[tfMaker].wq)

        # RUN
        proc.run(strEvent=startEvent, maxEvents=maxEvents)

    # Remove scratch directory if there is one
    if not batch_mode:     # Don't want to break other batch jobs when one finishes
        manager.rmScratch()

    print('\nDone!\n')

def section(hit):
    SECTION_MASK = 0x7 # space for up to 7 sections                                 
    SECTION_SHIFT = 18 
    # HcalSection BACK = 0, TOP = 1, BOTTOM = 2, RIGHT = 3, LEFT = 4
    return (hit.getID() >> SECTION_SHIFT) & SECTION_MASK
# Process an event
def event_process(self):

    # Initialize BDT input variables w/ defaults
    feats = next(iter(self.tfMakers.values())).resetFeats()

    feats['eventNumber'] = self.eventHeader.getEventNumber()

    #########################################
    # Assign pre-computed variables
    #########################################

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
    feats['ecalScore'] = self.ecalVeto.passesVeto()
    for i in range(0, physTools.nRegions):
        feats['electronContainmentEnergy_x{}'.format(i + 1)] = self.ecalVeto.getElectronContainmentEnergy()[i]
        feats['photonContainmentEnergy_x{}'.format(i + 1)  ] = self.ecalVeto.getPhotonContainmentEnergy()[i]
        feats['outsideContainmentEnergy_x{}'.format(i + 1) ] = self.ecalVeto.getOutsideContainmentEnergy()[i]
        feats['outsideContainmentNHits_x{}'.format(i + 1)  ] = self.ecalVeto.getOutsideContainmentNHits()[i]
        feats['outsideContainmentXStd_x{}'.format(i + 1)   ] = self.ecalVeto.getOutsideContainmentXStd()[i]
        feats['outsideContainmentYStd_x{}'.format(i + 1)   ] = self.ecalVeto.getOutsideContainmentYStd()[i]
    
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
    else:  # Should about never happen -> division by 0 in g_traj
        g_targPos = g_targP = np.zeros(3)

    # Get electron and photon trajectories
    e_traj = g_traj = None

    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)

    if e_targetHit != None:
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

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

    #############################################
    # Compute extra BDT input variables
    #############################################

    # Find epSep and epDot, and prepare electron and photon trajectory vectors
    if e_traj != None and g_traj != None:

        # Create arrays marking start and end of each trajectory
        e_traj_ends = [np.array([e_traj[0][0], e_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([e_traj[-1][0], e_traj[-1][1], physTools.ecal_layerZs[-1] ])]
        g_traj_ends = [np.array([g_traj[0][0], g_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([g_traj[-1][0], g_traj[-1][1], physTools.ecal_layerZs[-1] ])]

        e_norm  = physTools.unit( e_traj_ends[1] - e_traj_ends[0] )
        g_norm  = physTools.unit( g_traj_ends[1] - g_traj_ends[0] )
        feats['epSep'] = physTools.dist( e_traj_ends[0], g_traj_ends[0] )
        feats['epDot'] = physTools.dot(e_norm,g_norm)

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_ends   = [np.array([999 ,999 ,0   ]), np.array([999 ,999 ,999 ]) ]
        g_traj_ends   = [np.array([1000,1000,0   ]), np.array([1000,1000,1000]) ]

        feats['epSep'] = 10.0 + 1.0 # Don't cut on these in this case
        feats['epDot'] = 3.0 + 1.0

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

        if hit.getEnergy() <= 0:
            continue

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
    for hit in self.ecalRecHits:

        if hit.getEnergy() <= 0:
            continue

        layer = physTools.ecal_layer(hit)
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

    #######################################################
    # Other quantities needed for sample analysis
    #######################################################

    # Kinematic information for electron
    if e_ecalHit != None:
        feats['electronESPXPos'    ] = e_ecalPos[0]
        feats['electronESPYPos'    ] = e_ecalPos[1]
        feats['electronESPZPos'    ] = e_ecalPos[2]
        feats['electronESPXMom'    ] = e_ecalP[0]
        feats['electronESPYMom'    ] = e_ecalP[1]
        feats['electronESPZMom'    ] = e_ecalP[2]
        feats['electronESPMagMom'  ] = physTools.mag(e_ecalP)
        feats['electronESPThetaMom'] = physTools.angle(e_ecalP, units = 'degrees')
        feats['isAtESP'            ] = 1

    if e_targetHit != None:
        e_targPos, e_targP = e_targetHit.getPosition(), e_targetHit.getMomentum()
        feats['electronTSPXPos'    ] = e_targPos[0]
        feats['electronTSPYPos'    ] = e_targPos[1]
        feats['electronTSPZPos'    ] = e_targPos[2]
        feats['electronTSPXMom'    ] = e_targP[0]
        feats['electronTSPYMom'    ] = e_targP[1]
        feats['electronTSPZMom'    ] = e_targP[2]
        feats['electronTSPMagMom'  ] = physTools.mag(e_targP)
        feats['electronTSPThetaMom'] = physTools.angle(e_targP, units = 'degrees')
        feats['isAtTSP'            ] = 1

    fiducial = 0
    if e_ecalHit != None:
        slopeXZ = 99999
        if e_ecalP[0] != 0:
            slopeXZ = e_ecalP[2]/e_ecalP[0]
        fX = (physTools.ecal_layerZs[0] - physTools.ecal_front_z)/slopeXZ + e_ecalPos[0]

        slopeYZ = 99999
        if e_ecalP[1] != 0:
            slopeYZ = e_ecalP[2]/e_ecalP[1]
        fY = (physTools.ecal_layerZs[0] - physTools.ecal_front_z)/slopeYZ + e_ecalPos[1]

        for cell in cellMap:
            xDist = fY - cell[2]
            yDist = fX - cell[1]
            cellDist = physTools.mag([xDist, yDist])
            if cellDist <= physTools.cell_radius:
                fiducial = 1
                break

    feats['isFiducial'] = fiducial

    # Kinematic information for inferred photon
    g_ecalHit = physTools.gammaEcalSPHit(self.ecalSPHits)
    if g_ecalHit != None:
        g_ecalPos = g_ecalHit.getPosition()
        feats['photonESPXPos'] = g_ecalPos[0]
        feats['photonESPYPos'] = g_ecalPos[1]
        feats['photonESPZPos'] = g_ecalPos[2]

    if e_targetHit != None:
        feats['photonTSPXPos'  ] = g_targPos[0]
        feats['photonTSPYPos'  ] = g_targPos[1]
        feats['photonTSPZPos'  ] = g_targPos[2]
        feats['photonXMom'     ] = g_targP[0]
        feats['photonYMom'     ] = g_targP[1]
        feats['photonZMom'     ] = g_targP[2]
        feats['photonMagMom'   ] = physTools.mag(g_targP)
        feats['photonThetaMom' ] = physTools.angle(g_targP, units = 'degrees')

    # ECal rec hit information
    feats['totalRecAmplitude'] = sum([recHit.getAmplitude() for recHit in self.ecalRecHits])
    feats['totalRecEnergy'   ] = sum([recHit.getEnergy() for recHit in self.ecalRecHits])
    feats['nRecHits'         ] = len([recHit for recHit in self.ecalRecHits])

    # ECal sim hit information
#    feats['totalSimEDep'] = sum([simHit.getEdep() for simHit in self.ecalSimHits])
#    feats['nSimHits'    ] = len([simHit for simHit in self.ecalSimHits])

    # Sort the ECal rec hits and sim hits by hit ID
    ecalRecHitsSorted = [hit for hit in self.ecalRecHits]
    ecalRecHitsSorted.sort(key = lambda hit : hit.getID())
    ecalSimHitsSorted = [hit for hit in self.ecalSimHits]
    ecalSimHitsSorted.sort(key = lambda hit : hit.getID())

    # ECal noise information
    for recHit in ecalRecHitsSorted:

        # If the noise flag is set, count the hit as a noise hit
        if recHit.isNoise():
            feats['totalNoiseEnergy'] += recHit.getEnergy()
            feats['nNoiseHits'      ] += 1

        # Otherwise, check for a sim hit whose hitID matches
        nSimHitMatch = 0
        for simHit in ecalSimHitsSorted:

            if simHit.getID() == recHit.getID():
                nSimHitMatch += 1

            elif simHit.getID() > recHit.getID():
                break

        # If no matching sim hit exists, count the hit as a noise hit
        if (not recHit.isNoise()) and (nSimHitMatch == 0):
            feats['totalNoiseEnergy'] += recHit.getEnergy()
            feats['nNoiseHits'      ] += 1

    # Rec hit information
    for recHit in self.ecalRecHits:
        recHitBranches['amplitude']['address'][0] = recHit.getAmplitude()
        recHitBranches['energy'   ]['address'][0] = recHit.getEnergy()
        recHitBranches['xPos'     ]['address'][0] = recHit.getXPos()
        recHitBranches['yPos'     ]['address'][0] = recHit.getYPos()
        recHitBranches['zPos'     ]['address'][0] = recHit.getZPos()
        '''
        matchingSimEDep = 0
        for simHit in self.ecalSimHits:

            if simHit.getID() == recHit.getID():
                matchingSimEDep += simHit.getEdep()

        recHitBranches['matchingSimEDep']['address'][0] = matchingSimEDep

        for varName in branches_info:
            recHitBranches[varName]['address'][0] = feats[varName]
        '''
        self.recHitInfo.Fill()


    # Sim hit information
    for simHit in self.ecalSimHits:
        simHitBranches['eDep']['address'][0] = simHit.getEdep()
        simHitBranches['xPos']['address'][0] = simHit.getPosition()[0]
        simHitBranches['yPos']['address'][0] = simHit.getPosition()[1]
        simHitBranches['zPos']['address'][0] = simHit.getPosition()[2]

        #for varName in branches_info:
            #simHitBranches[varName]['address'][0] = feats[varName]

        self.simHitInfo.Fill()

    # Sim particle information
    for trackID, simParticle in self.simParticles:
        simParticleBranches['energy'    ]['address'][0] = simParticle.getEnergy()
        simParticleBranches['trackID'   ]['address'][0] = trackID
        simParticleBranches['pdgID'     ]['address'][0] = simParticle.getPdgID()
        simParticleBranches['vertXPos'  ]['address'][0] = simParticle.getVertex()[0]
        simParticleBranches['vertYPos'  ]['address'][0] = simParticle.getVertex()[1]
        simParticleBranches['vertZPos'  ]['address'][0] = simParticle.getVertex()[2]
        simParticleBranches['endXPos'   ]['address'][0] = simParticle.getEndPoint()[0]
        simParticleBranches['endYPos'   ]['address'][0] = simParticle.getEndPoint()[1]
        simParticleBranches['endZPos'   ]['address'][0] = simParticle.getEndPoint()[2]
        simParticleBranches['xMom'      ]['address'][0] = simParticle.getMomentum()[0]
        simParticleBranches['yMom'      ]['address'][0] = simParticle.getMomentum()[1]
        simParticleBranches['zMom'      ]['address'][0] = simParticle.getMomentum()[2]
        simParticleBranches['mass'      ]['address'][0] = simParticle.getMass()
        simParticleBranches['charge'    ]['address'][0] = simParticle.getCharge()
        simParticleBranches['nDaughters']['address'][0] = len(simParticle.getDaughters())
        simParticleBranches['nParents'  ]['address'][0] = len(simParticle.getParents())

        for varName in branches_info:
            simParticleBranches[varName]['address'][0] = feats[varName]

        self.simParticleInfo.Fill()
    #Get Scoring Plane and ecal values
    ecalHits = 0
    ecalEnergy = 0
    for hit in self.ecalSPHits:
        ecalHits = ecalHits+1
        ecalEnergy = ecalEnergy+ hit.getEnergy()
    feats['ecalSPHits'] = ecalHits
    feats['ecalSPEnergy'] = ecalEnergy
    targetHits = 0
    targetEnergy = 0



    maxPhotonMomentum = -1
    gP = [0,0,-1] #default, in case of no photon
    gAtDownSP = False # account if there is any photon at the most downstream target scoring plane hit
    feats['downSPReached'] = 0 #default
    hardBremHit = False  #accounts if the hard brem reaches the most downstream plane
    track2isPhoton = True
    for hit in self.targetSPHits:
        c = hit.getTrackID() 
        if abs(hit.getPosition()[2] - physTools.sp_trigger_pad_down_l1_z) <=0.5*physTools.sp_thickness:
            targetHits= targetHits +1
            targetEnergy = targetEnergy+ hit.getEnergy()
        if c == 2:
            if abs(hit.getPdgID()) == 22:
                temp = hit.getMomentum()
                maxPhotonMomentum = (temp[1]**2 + temp[2]**2+temp[0]**2)**0.5
                gP = temp
                pos = hit.getPosition()
                feats['gx'] = pos[0]
                feats['gy'] = pos[1]
                feats['gz'] = pos[2]
                if abs(pos[2] - physTools.sp_trigger_pad_down_l2_z) <=0.5*physTools.sp_thickness:
                    hardBremHit = True
            else:
                track2isPhoton = False
                pdg2id = abs(hit.getPdgID())
        if abs(hit.getPdgID()) == 22 and abs(hit.getPosition()[2] - physTools.sp_trigger_pad_down_l2_z) <=0.5*physTools.sp_thickness:  #determine if there is any photon on the most downstream target scoring plane
            gAtDownSP = True
    if gAtDownSP == False:
        feats['downSPReached'] = -2
    else:
        if not hardBremHit:
            feats['downSPReached'] = -1
    if not track2isPhoton:
        feats['downSPReached'] = pdg2id

    feats['targetSPHits'] = targetHits
    feats['targetSPEnergy'] = targetEnergy
    feats['gPx'] = gP[0]    
    feats['gPy'] = gP[1]
    feats['gPz'] = gP[2]
    feats['gPMagnitude'] = maxPhotonMomentum
    feats['gThetaNew'] = physTools.angle(gP, units='degrees')


    # add a photon scoring plane
    if e_targetHit != None:
        position = e_targetHit.getPosition()
        feats['ex'] = position[0]
        feats['ey'] = position[1]
        feats['ez'] = position[2]
        targetElectron = 1
        g_targPos, g_targP = physTools.gammaTargetInfo( e_targetHit )
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
        feats['egThetaNew'] = physTools.angle(gP, units='degrees', vec2=e_targetHit.getMomentum())
    else: # Should about never happen -> division by 0 in g_traj
        print( 'no e at targ!' )
        g_targPos = g_targP = np.zeros( 3 )
        targetElectron = 0
        feats[ 'eTheta' ] = -1
        feats[ 'gTheta' ] = -1
        feats[ 'egTheta' ] = -1
        feats[ 'egThetaNew' ] = -1
        feats['ePz'] = -1
        feats['eP'] = -1
    feats['targetElectron'] = targetElectron

    # Get electron and photon trajectories AND
    # Fiducial categories (filtered into different output trees)
    #if self.separate:
    e_traj = g_traj = None
    e_fid = g_fid = False

    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts( e_ecalPos, e_ecalP )
        for cell in cellMap:
            if physTools.dist(
                cell[ 1: ], e_traj[ 0 ]
                ) <= physTools.cell_radius:
                e_fid = True
                break

    if e_targetHit != None:
        g_traj = physTools.layerIntercepts( g_targPos, g_targP )
        for cell in cellMap:
            if physTools.dist(
                cell[ 1: ], g_traj[ 0 ]
                ) <= physTools.cell_radius:
                g_fid = True
                break
    



    ###################################
    # Determine hcal_hit section
    ###################################
    self.leftmaxPE = 0
    self.rightmaxPE = 0
    self.topmaxPE = 0
    self.bottommaxPE = 0
    self.backmaxPE = 0
    self.maxPE = 0
    self.sidemaxPE = 0
    for hit in self.hcalRecHits:
        if hit.getPE() > self.maxPE:
            self.maxPE = hit.getPE()
        sec = section(hit)
        currentPE = hit.getPE()
        if sec > 2.5 :
            if sec > 3.5:
                if currentPE > self.leftmaxPE:
                    self.leftmaxPE = currentPE
                    if currentPE > self.sidemaxPE:
                        self.sidemaxPE = currentPE
            else:
                if currentPE > self.rightmaxPE:
                    self.rightmaxPE = currentPE
                    if currentPE > self.sidemaxPE:
                        self.sidemaxPE = currentPE
        else:
            if sec < 0.5:
                if currentPE > self.backmaxPE:
                    self.backmaxPE = currentPE
            else: 
                if sec <1.5:
                    if currentPE > self.topmaxPE:
                        self.topmaxPE = currentPE
                        if currentPE > self.sidemaxPE:
                            self.sidemaxPE = currentPE
                else: 
                    if currentPE > self.bottommaxPE:
                        self.bottommaxPE = currentPE
                        if currentPE > self.sidemaxPE:
                            self.sidemaxPE = currentPE
    feats['rightmaxPE'] = self.rightmaxPE
    feats['leftmaxPE'] = self.leftmaxPE
    feats['bottommaxPE'] = self.bottommaxPE
    feats['topmaxPE'] = self.topmaxPE
    feats['backmaxPE'] = self.backmaxPE
    feats['sidemaxPE'] = self.sidemaxPE
    feats['maxPE'] = self.maxPE
    rl1Ne = 0
    rtotalNe= 0
    rl1E = 0
    rtotalE = 0
    E  = 0
    first2L = True
    first3L = True
    p = [] 
    rl1Nea = 0
    dLayer = -1
    for hit in self.RecoilSimHits:
        if abs(hit.getPdgID()) == 11:
            hitz = hit.getPosition()[2]            
            if dLayer<hitz:
                dLayer = hitz
            if hitz<9.5:
                p = hit.getMomentum()
                E = (p[0]**2+p[1]**2+p[2]**2)**0.5
                rl1E = rl1E +E
                rl1Ne = rl1Ne + 1
                rl1Nea = rl1Nea + 1
            else:
                if first2L and hitz < 16:

                    if p == []:
                        print('backward going electron!')
                        print(hit.getMomentum()[2])
                        rl1Ne = rl1Ne -1
                    else: 
                        feats['recoilTheta2L'] = physTools.angle(p, units='degrees')
                    first2L = False
                else:
                    if (not p == []) and first3L and hitz < 25:
                        feats['recoilTheta3L'] = physTools.angle(p, units='degrees')
                        first3L = False 
                    
            rtotalNe = rtotalNe +1
            rtotalE = rtotalE + E
    feats['recoilTotalNe'] = rtotalNe
    feats['recoilTotalE'] = rtotalE
    feats['recoilLayer1E'] = rl1E
    feats['recoilLayer1Ne'] = rl1Ne
    feats['recoilLayer1NeAdjusted'] = rl1Nea
    feats['recoilDeepestLayerHit'] = dLayer

    # Fill the tree (according to fiducial category) with values for this event
    if not self.separate:
        if e_fid and g_fid: feats['ecalFid'] = 3
        elif e_fid and not g_fid: feats['ecalFid'] = 2
        elif not e_fid and g_fid: feats['ecalFid'] = 1
        else: feats['ecalFid'] = 0
        self.tfMakers[ 'unsorted' ].fillEvent( feats )
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)

if __name__ == "__main__":
    main()
