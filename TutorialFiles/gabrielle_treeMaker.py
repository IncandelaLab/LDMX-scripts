#treeMaker version that gets Gabrielle BDT variables from the unflattened root tree as calculated by EcalVetoProcessor
#simplified such that there is no calculation of variables and this does not separate output trees
#removed unnecessary dependencies

import os
import math
import ROOT as r
import numpy as np
import ROOTmanager as manager
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
        'ecalBackEnergy':            {'rtype': float, 'default': 0.}
}
for i in range(1, 6):
    branches_info['electronContainmentEnergy_x{}'.format(i)]   = {'rtype': float, 'default': 0.}
for i in range(1, 6):
    branches_info['photonContainmentEnergy_x{}'.format(i)]     = {'rtype': float, 'default': 0.}
for i in range(1, 6):
    branches_info['outsideContainmentEnergy_x{}'.format(i)]    = {'rtype': float, 'default': 0.}
for i in range(1, 6):
    branches_info['outsideContainmentNHits_x{}'.format(i)]     = {'rtype': int, 'default': 0.}
for i in range(1, 6):
    branches_info['outsideContainmentXStd_x{}'.format(i)]      = {'rtype': float, 'default': 0.}
for i in range(1, 6):
    branches_info['outsideContainmentYStd_x{}'.format(i)]      = {'rtype': float, 'default': 0.}

    
def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    batch_mode = pdict['batch']
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
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_v3_v13')
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v3_v13')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v3_v13')
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_v3_v13')

        # Tree/Files(s) to make
        print('\nRunning %s'%(proc.ID))

        proc.tfMakers = {'unsorted': None}

        for tfMaker in proc.tfMakers:
            proc.tfMakers[tfMaker] = manager.TreeMaker(group_labels[procs.index(proc)]+".root","EcalVeto",branches_info,outlist[0])

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
    for i in range(1, 6):
        feats['electronContainmentEnergy_x{}'.format(i)]   = self.ecalVeto.getElectronContainmentEnergy()[i-1]
        feats['photonContainmentEnergy_x{}'.format(i)]     = self.ecalVeto.getPhotonContainmentEnergy()[i-1]
        feats['outsideContainmentEnergy_x{}'.format(i)]    = self.ecalVeto.getOutsideContainmentEnergy()[i-1]
        feats['outsideContainmentNHits_x{}'.format(i)]     = self.ecalVeto.getOutsideContainmentNHits()[i-1]
        feats['outsideContainmentXStd_x{}'.format(i)]      = self.ecalVeto.getOutsideContainmentXStd()[i-1]
        feats['outsideContainmentYStd_x{}'.format(i)]      = self.ecalVeto.getOutsideContainmentYStd()[i-1]
    
            
    # Fill the tree with values for this event
    self.tfMakers['unsorted'].fillEvent(feats)


if __name__ == "__main__":
    main()
