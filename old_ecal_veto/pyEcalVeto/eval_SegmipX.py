import os
import sys
import numpy as np
import pickle as pkl
import xgboost as xgb
import mods.ROOTmanager as manager
import math
import ROOT as r
from mods import ROOTmanager as manager
from mods import physTools, mipTracking

pkl_file   = os.getcwd()+'/bdt_test_0/bdt_test_0_weights.pkl'
model = pkl.load(open(pkl_file,'rb'))

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
        'photonTerritoryHits':       {'rtype': int,   'default': 0 },
        'epSep':                     {'rtype': float, 'default': 0.},
        'epDot':                     {'rtype': float, 'default': 0.},
        # Longitudinal segment variables
        'energy_s1':                 {'rtype': float, 'default': 0.},
        'xMean_s1':                {'rtype': float, 'default': 0.},
        'yMean_s1':                  {'rtype': float, 'default': 0.},
        'layerMean_s1':             {'rtype': float, 'default': 0.},
        'energy_s2':                {'rtype': float, 'default': 0.},
        'yMean_s3':                  {'rtype': float, 'default': 0.},
         # Electron RoC variables
        'eContEnergy_x1_s1':        {'rtype': float, 'default': 0.},
        'eContEnergy_x2_s1':         {'rtype': float, 'default': 0.},
        'eContYMean_x1_s1':         {'rtype': float, 'default': 0.},
        'eContEnergy_x1_s2':         {'rtype': float, 'default': 0.},
        'eContEnergy_x2_s2':         {'rtype': float, 'default': 0.},
        'eContYMean_x1_s2':          {'rtype': float, 'default': 0.},
         # Photon RoC variables
        'gContNHits_x1_s1':          {'rtype': int, 'default': 0},
        'gContYMean_x1_s1':         {'rtype': float, 'default': 0.},
        'gContNHits_x1_s2':          {'rtype': int, 'default': 0},
        # Outside RoC variables
        'oContEnergy_x1_s1':         {'rtype': float, 'default': 0.},
        'oContEnergy_x2_s1' :         {'rtype': float, 'default': 0.},
        'oContEnergy_x3_s1':         {'rtype': float, 'default': 0.},
        'oContNHits_x1_s1':           {'rtype': int, 'default': 0},
        'oContXMean_x1_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x1_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x2_s1':          {'rtype': float, 'default': 0.},
        'oContYStd_x1_s1':          {'rtype': float, 'default': 0.},
        'oContEnergy_x1_s2':          {'rtype': float, 'default': 0.},
        'oContEnergy_x2_s2':          {'rtype': float, 'default': 0.},
        'oContEnergy_x3_s2':         {'rtype': float, 'default': 0.},
        'oContLayerMean_x1_s2':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x1_s2':       {'rtype': float, 'default': 0.},
        'oContEnergy_x1_s3':           {'rtype': float, 'default': 0.},
        'oContLayerMean_x1_s3':       {'rtype': float, 'default': 0.}
                                                           






                                
        }

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    maxEvent = pdict['maxEvents']

    branches_info['discValue_EcalVeto'] = {'rtype': float, 'default': 0.5}
    #branches_info['epAng'] = {'rtype': float, 'default':99999}
    branches_info['epAng'] = {'rtype': float, 'default':99999}
    branches_info['HCalVeto_passesVeto'] = {'rtype': int, 'default': 0}
    branches_info['TargetScoringPlaneHits_z'] = {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.])}
    branches_info['TargetScoringPlaneHits_px'] = {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.])}
    branches_info['TargetScoringPlaneHits_py'] = {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.])}
    branches_info['TargetScoringPlaneHits_pz'] = {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.])}
    branches_info['TargetScoringPlaneHits_pdgID'] = {'rtype': 'vector<int>', 'default': r.std.vector('int')([0])}
    # Construct tree processes

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels, inlist):
        procs.append( manager.TreeProcess(event_process, group, ID=gl, tree_name='EcalVeto_flatten',
            pfreq=100) )

    # Process jobs
    for proc in procs:

        print('\nRunning %s'%(proc.ID))
        
        # Move into appropriate scratch dir
        os.chdir(proc.tmp_dir)

        # Make an output file and new tree (copied from input + discValue)
        proc.tfMaker = manager.TreeMaker(group_labels[procs.index(proc)]+'.root',\
                                         "EcalVeto",\
                                         branches_info,\
                                         outlist[procs.index(proc)]
                                         )

        # RUN
        proc.extrafs = [ proc.tfMaker.wq ] # Gets executed at the end of run()
        proc.run(maxEvents=maxEvent)

    # Remove scratch directory if there is one
    manager.rmScratch()

    print('\nDone!\n')


def event_process(self):

    # Feature list from input tree
    # Exp: feats = [ feat_value for feat_value in self.tree~ ]
    # Put all segmentation variables in for now (Take out the ones we won't need once
    # we make sure that all the python bdt stuff works)
    feats = [
            # Base variables
            self.tree.nReadoutHits              ,
            self.tree.summedDet                 ,
            self.tree.summedTightIso            ,
            self.tree.maxCellDep                ,
            self.tree.showerRMS                 ,
            self.tree.xStd                      ,
            self.tree.yStd                      ,
            self.tree.avgLayerHit               ,
            self.tree.stdLayerHit               ,
            self.tree.deepestLayerHit           ,
            self.tree.ecalBackEnergy            ,
            # MIP Tracking variables
            self.tree.straight4                 ,
            self.tree.firstNearPhLayer          ,
            self.tree.nNearPhHits               ,
            self.tree.photonTerritoryHits       ,
            self.tree.epSep                     ,
            self.tree.epDot                     ,
           
            # Longitudinal segment variables
            self.tree.energy_s1                 ,
            self.tree.xMean_s1                  ,
            self.tree.yMean_s1                  ,
            self.tree.layerMean_s1              ,
            self.tree.energy_s2                 ,
            self.tree.yMean_s3                  ,
            # Electron RoC variables
            self.tree.eContEnergy_x1_s1         ,
            self.tree.eContEnergy_x2_s1         ,
            self.tree.eContYMean_x1_s1          ,
            self.tree.eContEnergy_x1_s2         ,
            self.tree.eContEnergy_x2_s2         ,
            self.tree.eContYMean_x1_s2          ,
            # Photon RoC variables
            self.tree.gContNHits_x1_s1          ,
            self.tree.gContYMean_x1_s1          ,
            self.tree.gContNHits_x1_s2          ,
            # Outside RoC variables
            self.tree.oContEnergy_x1_s1         ,
            self.tree.oContEnergy_x2_s1         ,
            self.tree.oContEnergy_x3_s1         ,
            self.tree.oContNHits_x1_s1          ,
            self.tree.oContXMean_x1_s1          ,
            self.tree.oContYMean_x1_s1          ,
            self.tree.oContYMean_x2_s1          ,
            self.tree.oContYStd_x1_s1           ,
            self.tree.oContEnergy_x1_s2         ,
            self.tree.oContEnergy_x2_s2         ,
            self.tree.oContEnergy_x3_s2         ,
            self.tree.oContLayerMean_x1_s2      ,
            self.tree.oContLayerStd_x1_s2       ,
            self.tree.oContEnergy_x1_s3         ,
            self.tree.oContLayerMean_x1_s3      ,
            ]

    # Copy input tree feats to new tree
    for feat_name, feat_value in zip(self.tfMaker.branches_info, feats):
        self.tfMaker.branches[feat_name][0] = feat_value

    # Add prediction to new tree
    evtarray = np.array([feats])
    pred = float(model.predict(xgb.DMatrix(evtarray))[0])
    self.tfMaker.branches['discValue_EcalVeto'][0] = pred
    #self.tfMaker.branches['epAng'][0] = self.tree.epAng
    self.tfMaker.branches['HCalVeto_passesVeto'][0] = self.tree.HCalVeto_passesVeto
    self.tfMaker.branches['TargetScoringPlaneHits_z'].assign(self.tree.TargetScoringPlaneHits_z.begin(), self.tree.TargetScoringPlaneHits_z.end())
    self.tfMaker.branches['TargetScoringPlaneHits_px'].assign(self.tree.TargetScoringPlaneHits_px.begin(), self.tree.TargetScoringPlaneHits_px.end())
    self.tfMaker.branches['TargetScoringPlaneHits_py'].assign(self.tree.TargetScoringPlaneHits_py.begin(), self.tree.TargetScoringPlaneHits_py.end())
    self.tfMaker.branches['TargetScoringPlaneHits_pz'].assign(self.tree.TargetScoringPlaneHits_pz.begin(), self.tree.TargetScoringPlaneHits_pz.end())
    self.tfMaker.branches['TargetScoringPlaneHits_pdgID'].assign(self.tree.TargetScoringPlaneHits_pdgID.begin(), self.tree.TargetScoringPlaneHits_pdgID.end())


    # Fill new tree with current event values
    self.tfMaker.tree.Fill()

if __name__ == "__main__":
    main()
