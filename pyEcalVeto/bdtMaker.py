import os
import sys
import logging
import argparse
import ROOT as r
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib as plt
from array    import array
from optparse import OptionParser


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
plt.use('Agg')

class sampleContainer:
    def __init__(self,filename,maxEvts,trainFrac,isSig):

        print("Initializing Container!")
        self.tree = r.TChain("EcalVeto")
        self.tree.Add(filename)
        self.maxEvts = maxEvts
        self.trainFrac = trainFrac
        self.isSig   = isSig

    def root2PyEvents(self):
        self.events =  []
        for event in self.tree:
            if len(self.events) >= self.maxEvts:
                continue

            evt = [
                    # Base variables
                    event.nReadoutHits              ,
                    event.summedDet                 ,
                    event.summedTightIso            ,
                    event.maxCellDep                ,
                    event.showerRMS                 ,
                    event.xStd                      ,
                    event.yStd                      ,
                    event.avgLayerHit               ,
                    event.stdLayerHit               ,
                    event.deepestLayerHit           ,
                    event.ecalBackEnergy            ,
                    # MIP Tracking variables
                    event.straight4                 ,
                    event.firstNearPhLayer          ,
                    event.nNearPhHits               ,
                    event.fullElectronTerritoryHits ,
                    event.fullPhotonTerritoryHits   ,
                    event.fullTerritoryRatio        ,
                    event.electronTerritoryHits     ,
                    event.photonTerritoryHits       ,
                    event.TerritoryRatio            ,
                    # Longitudinal segment variables
                    event.energy_s1                 ,
                    event.nHits_s1                  ,
                    event.xMean_s1                  ,
                    event.yMean_s1                  ,
                    event.layerMean_s1              ,
                    event.xStd_s1                   ,
                    event.yStd_s1                   ,
                    event.layerStd_s1               ,
                    event.energy_s2                 ,
                    event.nHits_s2                  ,
                    event.xMean_s2                  ,
                    event.yMean_s2                  ,
                    event.layerMean_s2              ,
                    event.xStd_s2                   ,
                    event.yStd_s2                   ,
                    event.layerStd_s2               ,
                    event.energy_s3                 ,
                    event.nHits_s3                  ,
                    event.xMean_s3                  ,
                    event.yMean_s3                  ,
                    event.layerMean_s3              ,
                    event.xStd_s3                   ,
                    event.yStd_s3                   ,
                    event.layerStd_s3               ,
                    # Electron RoC variables
                    event.eContEnergy_x1_s1         ,
                    event.eContEnergy_x2_s1         ,
                    event.eContEnergy_x3_s1         ,
                    event.eContEnergy_x4_s1         ,
                    event.eContEnergy_x5_s1         ,
                    event.eContNHits_x1_s1          ,
                    event.eContNHits_x2_s1          ,
                    event.eContNHits_x3_s1          ,
                    event.eContNHits_x4_s1          ,
                    event.eContNHits_x5_s1          ,
                    event.eContXMean_x1_s1          ,
                    event.eContXMean_x2_s1          ,
                    event.eContXMean_x3_s1          ,
                    event.eContXMean_x4_s1          ,
                    event.eContXMean_x5_s1          ,
                    event.eContYMean_x1_s1          ,
                    event.eContYMean_x2_s1          ,
                    event.eContYMean_x3_s1          ,
                    event.eContYMean_x4_s1          ,
                    event.eContYMean_x5_s1          ,
                    event.eContLayerMean_x1_s1      ,
                    event.eContLayerMean_x2_s1      ,
                    event.eContLayerMean_x3_s1      ,
                    event.eContLayerMean_x4_s1      ,
                    event.eContLayerMean_x5_s1      ,
                    event.eContXStd_x1_s1           ,
                    event.eContXStd_x2_s1           ,
                    event.eContXStd_x3_s1           ,
                    event.eContXStd_x4_s1           ,
                    event.eContXStd_x5_s1           ,
                    event.eContYStd_x1_s1           ,
                    event.eContYStd_x2_s1           ,
                    event.eContYStd_x3_s1           ,
                    event.eContYStd_x4_s1           ,
                    event.eContYStd_x5_s1           ,
                    event.eContLayerStd_x1_s1       ,
                    event.eContLayerStd_x2_s1       ,
                    event.eContLayerStd_x3_s1       ,
                    event.eContLayerStd_x4_s1       ,
                    event.eContLayerStd_x5_s1       ,
                    event.eContEnergy_x1_s2         ,
                    event.eContEnergy_x2_s2         ,
                    event.eContEnergy_x3_s2         ,
                    event.eContEnergy_x4_s2         ,
                    event.eContEnergy_x5_s2         ,
                    event.eContNHits_x1_s2          ,
                    event.eContNHits_x2_s2          ,
                    event.eContNHits_x3_s2          ,
                    event.eContNHits_x4_s2          ,
                    event.eContNHits_x5_s2          ,
                    event.eContXMean_x1_s2          ,
                    event.eContXMean_x2_s2          ,
                    event.eContXMean_x3_s2          ,
                    event.eContXMean_x4_s2          ,
                    event.eContXMean_x5_s2          ,
                    event.eContYMean_x1_s2          ,
                    event.eContYMean_x2_s2          ,
                    event.eContYMean_x3_s2          ,
                    event.eContYMean_x4_s2          ,
                    event.eContYMean_x5_s2          ,
                    event.eContLayerMean_x1_s2      ,
                    event.eContLayerMean_x2_s2      ,
                    event.eContLayerMean_x3_s2      ,
                    event.eContLayerMean_x4_s2      ,
                    event.eContLayerMean_x5_s2      ,
                    event.eContXStd_x1_s2           ,
                    event.eContXStd_x2_s2           ,
                    event.eContXStd_x3_s2           ,
                    event.eContXStd_x4_s2           ,
                    event.eContXStd_x5_s2           ,
                    event.eContYStd_x1_s2           ,
                    event.eContYStd_x2_s2           ,
                    event.eContYStd_x3_s2           ,
                    event.eContYStd_x4_s2           ,
                    event.eContYStd_x5_s2           ,
                    event.eContLayerStd_x1_s2       ,
                    event.eContLayerStd_x2_s2       ,
                    event.eContLayerStd_x3_s2       ,
                    event.eContLayerStd_x4_s2       ,
                    event.eContLayerStd_x5_s2       ,
                    event.eContEnergy_x1_s3         ,
                    event.eContEnergy_x2_s3         ,
                    event.eContEnergy_x3_s3         ,
                    event.eContEnergy_x4_s3         ,
                    event.eContEnergy_x5_s3         ,
                    event.eContNHits_x1_s3          ,
                    event.eContNHits_x2_s3          ,
                    event.eContNHits_x3_s3          ,
                    event.eContNHits_x4_s3          ,
                    event.eContNHits_x5_s3          ,
                    event.eContXMean_x1_s3          ,
                    event.eContXMean_x2_s3          ,
                    event.eContXMean_x3_s3          ,
                    event.eContXMean_x4_s3          ,
                    event.eContXMean_x5_s3          ,
                    event.eContYMean_x1_s3          ,
                    event.eContYMean_x2_s3          ,
                    event.eContYMean_x3_s3          ,
                    event.eContYMean_x4_s3          ,
                    event.eContYMean_x5_s3          ,
                    event.eContLayerMean_x1_s3      ,
                    event.eContLayerMean_x2_s3      ,
                    event.eContLayerMean_x3_s3      ,
                    event.eContLayerMean_x4_s3      ,
                    event.eContLayerMean_x5_s3      ,
                    event.eContXStd_x1_s3           ,
                    event.eContXStd_x2_s3           ,
                    event.eContXStd_x3_s3           ,
                    event.eContXStd_x4_s3           ,
                    event.eContXStd_x5_s3           ,
                    event.eContYStd_x1_s3           ,
                    event.eContYStd_x2_s3           ,
                    event.eContYStd_x3_s3           ,
                    event.eContYStd_x4_s3           ,
                    event.eContYStd_x5_s3           ,
                    event.eContLayerStd_x1_s3       ,
                    event.eContLayerStd_x2_s3       ,
                    event.eContLayerStd_x3_s3       ,
                    event.eContLayerStd_x4_s3       ,
                    event.eContLayerStd_x5_s3       ,
                    # Photon RoC variables
                    event.gContEnergy_x1_s1         ,
                    event.gContEnergy_x2_s1         ,
                    event.gContEnergy_x3_s1         ,
                    event.gContEnergy_x4_s1         ,
                    event.gContEnergy_x5_s1         ,
                    event.gContNHits_x1_s1          ,
                    event.gContNHits_x2_s1          ,
                    event.gContNHits_x3_s1          ,
                    event.gContNHits_x4_s1          ,
                    event.gContNHits_x5_s1          ,
                    event.gContXMean_x1_s1          ,
                    event.gContXMean_x2_s1          ,
                    event.gContXMean_x3_s1          ,
                    event.gContXMean_x4_s1          ,
                    event.gContXMean_x5_s1          ,
                    event.gContYMean_x1_s1          ,
                    event.gContYMean_x2_s1          ,
                    event.gContYMean_x3_s1          ,
                    event.gContYMean_x4_s1          ,
                    event.gContYMean_x5_s1          ,
                    event.gContLayerMean_x1_s1      ,
                    event.gContLayerMean_x2_s1      ,
                    event.gContLayerMean_x3_s1      ,
                    event.gContLayerMean_x4_s1      ,
                    event.gContLayerMean_x5_s1      ,
                    event.gContXStd_x1_s1           ,
                    event.gContXStd_x2_s1           ,
                    event.gContXStd_x3_s1           ,
                    event.gContXStd_x4_s1           ,
                    event.gContXStd_x5_s1           ,
                    event.gContYStd_x1_s1           ,
                    event.gContYStd_x2_s1           ,
                    event.gContYStd_x3_s1           ,
                    event.gContYStd_x4_s1           ,
                    event.gContYStd_x5_s1           ,
                    event.gContLayerStd_x1_s1       ,
                    event.gContLayerStd_x2_s1       ,
                    event.gContLayerStd_x3_s1       ,
                    event.gContLayerStd_x4_s1       ,
                    event.gContLayerStd_x5_s1       ,
                    event.gContEnergy_x1_s2         ,
                    event.gContEnergy_x2_s2         ,
                    event.gContEnergy_x3_s2         ,
                    event.gContEnergy_x4_s2         ,
                    event.gContEnergy_x5_s2         ,
                    event.gContNHits_x1_s2          ,
                    event.gContNHits_x2_s2          ,
                    event.gContNHits_x3_s2          ,
                    event.gContNHits_x4_s2          ,
                    event.gContNHits_x5_s2          ,
                    event.gContXMean_x1_s2          ,
                    event.gContXMean_x2_s2          ,
                    event.gContXMean_x3_s2          ,
                    event.gContXMean_x4_s2          ,
                    event.gContXMean_x5_s2          ,
                    event.gContYMean_x1_s2          ,
                    event.gContYMean_x2_s2          ,
                    event.gContYMean_x3_s2          ,
                    event.gContYMean_x4_s2          ,
                    event.gContYMean_x5_s2          ,
                    event.gContLayerMean_x1_s2      ,
                    event.gContLayerMean_x2_s2      ,
                    event.gContLayerMean_x3_s2      ,
                    event.gContLayerMean_x4_s2      ,
                    event.gContLayerMean_x5_s2      ,
                    event.gContXStd_x1_s2           ,
                    event.gContXStd_x2_s2           ,
                    event.gContXStd_x3_s2           ,
                    event.gContXStd_x4_s2           ,
                    event.gContXStd_x5_s2           ,
                    event.gContYStd_x1_s2           ,
                    event.gContYStd_x2_s2           ,
                    event.gContYStd_x3_s2           ,
                    event.gContYStd_x4_s2           ,
                    event.gContYStd_x5_s2           ,
                    event.gContLayerStd_x1_s2       ,
                    event.gContLayerStd_x2_s2       ,
                    event.gContLayerStd_x3_s2       ,
                    event.gContLayerStd_x4_s2       ,
                    event.gContLayerStd_x5_s2       ,
                    event.gContEnergy_x1_s3         ,
                    event.gContEnergy_x2_s3         ,
                    event.gContEnergy_x3_s3         ,
                    event.gContEnergy_x4_s3         ,
                    event.gContEnergy_x5_s3         ,
                    event.gContNHits_x1_s3          ,
                    event.gContNHits_x2_s3          ,
                    event.gContNHits_x3_s3          ,
                    event.gContNHits_x4_s3          ,
                    event.gContNHits_x5_s3          ,
                    event.gContXMean_x1_s3          ,
                    event.gContXMean_x2_s3          ,
                    event.gContXMean_x3_s3          ,
                    event.gContXMean_x4_s3          ,
                    event.gContXMean_x5_s3          ,
                    event.gContYMean_x1_s3          ,
                    event.gContYMean_x2_s3          ,
                    event.gContYMean_x3_s3          ,
                    event.gContYMean_x4_s3          ,
                    event.gContYMean_x5_s3          ,
                    event.gContLayerMean_x1_s3      ,
                    event.gContLayerMean_x2_s3      ,
                    event.gContLayerMean_x3_s3      ,
                    event.gContLayerMean_x4_s3      ,
                    event.gContLayerMean_x5_s3      ,
                    event.gContXStd_x1_s3           ,
                    event.gContXStd_x2_s3           ,
                    event.gContXStd_x3_s3           ,
                    event.gContXStd_x4_s3           ,
                    event.gContXStd_x5_s3           ,
                    event.gContYStd_x1_s3           ,
                    event.gContYStd_x2_s3           ,
                    event.gContYStd_x3_s3           ,
                    event.gContYStd_x4_s3           ,
                    event.gContYStd_x5_s3           ,
                    event.gContLayerStd_x1_s3       ,
                    event.gContLayerStd_x2_s3       ,
                    event.gContLayerStd_x3_s3       ,
                    event.gContLayerStd_x4_s3       ,
                    event.gContLayerStd_x5_s3       ,
                    # Outside RoC variables
                    event.oContEnergy_x1_s1         ,
                    event.oContEnergy_x2_s1         ,
                    event.oContEnergy_x3_s1         ,
                    event.oContEnergy_x4_s1         ,
                    event.oContEnergy_x5_s1         ,
                    event.oContNHits_x1_s1          ,
                    event.oContNHits_x2_s1          ,
                    event.oContNHits_x3_s1          ,
                    event.oContNHits_x4_s1          ,
                    event.oContNHits_x5_s1          ,
                    event.oContXMean_x1_s1          ,
                    event.oContXMean_x2_s1          ,
                    event.oContXMean_x3_s1          ,
                    event.oContXMean_x4_s1          ,
                    event.oContXMean_x5_s1          ,
                    event.oContYMean_x1_s1          ,
                    event.oContYMean_x2_s1          ,
                    event.oContYMean_x3_s1          ,
                    event.oContYMean_x4_s1          ,
                    event.oContYMean_x5_s1          ,
                    event.oContLayerMean_x1_s1      ,
                    event.oContLayerMean_x2_s1      ,
                    event.oContLayerMean_x3_s1      ,
                    event.oContLayerMean_x4_s1      ,
                    event.oContLayerMean_x5_s1      ,
                    event.oContXStd_x1_s1           ,
                    event.oContXStd_x2_s1           ,
                    event.oContXStd_x3_s1           ,
                    event.oContXStd_x4_s1           ,
                    event.oContXStd_x5_s1           ,
                    event.oContYStd_x1_s1           ,
                    event.oContYStd_x2_s1           ,
                    event.oContYStd_x3_s1           ,
                    event.oContYStd_x4_s1           ,
                    event.oContYStd_x5_s1           ,
                    event.oContLayerStd_x1_s1       ,
                    event.oContLayerStd_x2_s1       ,
                    event.oContLayerStd_x3_s1       ,
                    event.oContLayerStd_x4_s1       ,
                    event.oContLayerStd_x5_s1       ,
                    event.oContEnergy_x1_s2         ,
                    event.oContEnergy_x2_s2         ,
                    event.oContEnergy_x3_s2         ,
                    event.oContEnergy_x4_s2         ,
                    event.oContEnergy_x5_s2         ,
                    event.oContNHits_x1_s2          ,
                    event.oContNHits_x2_s2          ,
                    event.oContNHits_x3_s2          ,
                    event.oContNHits_x4_s2          ,
                    event.oContNHits_x5_s2          ,
                    event.oContXMean_x1_s2          ,
                    event.oContXMean_x2_s2          ,
                    event.oContXMean_x3_s2          ,
                    event.oContXMean_x4_s2          ,
                    event.oContXMean_x5_s2          ,
                    event.oContYMean_x1_s2          ,
                    event.oContYMean_x2_s2          ,
                    event.oContYMean_x3_s2          ,
                    event.oContYMean_x4_s2          ,
                    event.oContYMean_x5_s2          ,
                    event.oContLayerMean_x1_s2      ,
                    event.oContLayerMean_x2_s2      ,
                    event.oContLayerMean_x3_s2      ,
                    event.oContLayerMean_x4_s2      ,
                    event.oContLayerMean_x5_s2      ,
                    event.oContXStd_x1_s2           ,
                    event.oContXStd_x2_s2           ,
                    event.oContXStd_x3_s2           ,
                    event.oContXStd_x4_s2           ,
                    event.oContXStd_x5_s2           ,
                    event.oContYStd_x1_s2           ,
                    event.oContYStd_x2_s2           ,
                    event.oContYStd_x3_s2           ,
                    event.oContYStd_x4_s2           ,
                    event.oContYStd_x5_s2           ,
                    event.oContLayerStd_x1_s2       ,
                    event.oContLayerStd_x2_s2       ,
                    event.oContLayerStd_x3_s2       ,
                    event.oContLayerStd_x4_s2       ,
                    event.oContLayerStd_x5_s2       ,
                    event.oContEnergy_x1_s3         ,
                    event.oContEnergy_x2_s3         ,
                    event.oContEnergy_x3_s3         ,
                    event.oContEnergy_x4_s3         ,
                    event.oContEnergy_x5_s3         ,
                    event.oContNHits_x1_s3          ,
                    event.oContNHits_x2_s3          ,
                    event.oContNHits_x3_s3          ,
                    event.oContNHits_x4_s3          ,
                    event.oContNHits_x5_s3          ,
                    event.oContXMean_x1_s3          ,
                    event.oContXMean_x2_s3          ,
                    event.oContXMean_x3_s3          ,
                    event.oContXMean_x4_s3          ,
                    event.oContXMean_x5_s3          ,
                    event.oContYMean_x1_s3          ,
                    event.oContYMean_x2_s3          ,
                    event.oContYMean_x3_s3          ,
                    event.oContYMean_x4_s3          ,
                    event.oContYMean_x5_s3          ,
                    event.oContLayerMean_x1_s3      ,
                    event.oContLayerMean_x2_s3      ,
                    event.oContLayerMean_x3_s3      ,
                    event.oContLayerMean_x4_s3      ,
                    event.oContLayerMean_x5_s3      ,
                    event.oContXStd_x1_s3           ,
                    event.oContXStd_x2_s3           ,
                    event.oContXStd_x3_s3           ,
                    event.oContXStd_x4_s3           ,
                    event.oContXStd_x5_s3           ,
                    event.oContYStd_x1_s3           ,
                    event.oContYStd_x2_s3           ,
                    event.oContYStd_x3_s3           ,
                    event.oContYStd_x4_s3           ,
                    event.oContYStd_x5_s3           ,
                    event.oContLayerStd_x1_s3       ,
                    event.oContLayerStd_x2_s3       ,
                    event.oContLayerStd_x3_s3       ,
                    event.oContLayerStd_x4_s3       ,
                    event.oContLayerStd_x5_s3
            ]

            self.events.append(evt)

        new_idx=np.random.permutation(np.arange(np.shape(self.events)[0]))
        self.events = np.array(self.events)
        np.take(self.events, new_idx, axis=0, out=self.events)
        print("Final Event Shape" + str(np.shape(self.events)))

    def constructTrainAndTest(self):
        self.train_x = self.events[0:int(len(self.events)*self.trainFrac)]
        self.test_x = self.events[int(len(self.events)*self.trainFrac):]

        self.train_y = np.zeros(len(self.train_x)) + (self.isSig == True)
        self.test_y = np.zeros(len(self.test_x)) + (self.isSig == True)

class mergedContainer:
    def __init__(self, sigContainer,bkgContainer):
        self.train_x = np.vstack((sigContainer.train_x,bkgContainer.train_x))
        self.train_y = np.append(sigContainer.train_y,bkgContainer.train_y)
        
        self.train_x[np.isnan(self.train_x)] = 0.000
        self.train_y[np.isnan(self.train_y)] = 0.000

        self.test_x  = np.vstack((sigContainer.test_x,bkgContainer.test_x))
        self.test_y  = np.append(sigContainer.test_y,bkgContainer.test_y)
        
        self.dtrain = xgb.DMatrix(self.train_x,self.train_y)
        self.dtest  = xgb.DMatrix(self.test_x,self.test_y)

if __name__ == "__main__":
    
    # Parse
    parser = OptionParser()
    parser.add_option('--seed', dest='seed',type="int",  default=2, help='Numpy random seed.')
    parser.add_option('--max_evt', dest='max_evt',type="int",  default=1500000, help='Max Events to load')
    parser.add_option('--train_frac', dest='train_frac',  default=.8, help='Fraction of events to use for training')
    parser.add_option('--eta', dest='eta',type="float",  default=0.023, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number',type="int",  default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth',type="int",  default=10, help='Max Tree Depth')
    parser.add_option('-b', dest='bkg_file', default='./bdt_0/bkg_train.root', help='name of background file')
    parser.add_option('-s', dest='sig_file', default='./bdt_0/sig_train.root', help='name of signal file')
    parser.add_option('-o', dest='out_name',  default='bdt_test', help='Output Pickle Name')
    (options, args) = parser.parse_args()

    # Seed numpy's randomness
    np.random.seed(options.seed)
   
    # Get BDT num
    bdt_num=0
    Check=True
    while Check:
        if not os.path.exists(options.out_name+'_'+str(bdt_num)):
            try:
                os.makedirs(options.out_name+'_'+str(bdt_num))
                Check=False
            except:
               Check=True
        else:
            bdt_num+=1

    # Print run info
    print( 'Random seed is = {}'.format(options.seed)             )
    print( 'You set max_evt = {}'.format(options.max_evt)         )
    print( 'You set tree number = {}'.format(options.tree_number) )
    print( 'You set max tree depth = {}'.format(options.depth)    )
    print( 'You set eta = {}'.format(options.eta)                 )

    # Make Signal Container
    print( 'Loading sig_file = {}'.format(options.sig_file) )
    sigContainer = sampleContainer(options.sig_file,options.max_evt,options.train_frac,True)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    # Make Background Container
    print( 'Loading bkg_file = {}'.format(options.bkg_file) )
    bkgContainer = sampleContainer(options.bkg_file,options.max_evt,options.train_frac,False)
    bkgContainer.root2PyEvents()
    bkgContainer.constructTrainAndTest()

    # Merge
    eventContainer = mergedContainer(sigContainer,bkgContainer)

    params = {
               'objective': 'binary:logistic',
               'eta': options.eta,
               'max_depth': options.depth,
               'min_child_weight': 20,
               # 'silent': 1,
               'subsample':.9,
               'colsample_bytree': .85,
               # 'eval_metric': 'auc',
               'eval_metric': 'error',
               'seed': 1,
               'nthread': 1,
               'verbosity': 1
               # 'early_stopping_rounds' : 10
    }

    # Train the BDT model
    evallist = [(eventContainer.dtest,'eval'), (eventContainer.dtrain,'train')]
    gbm = xgb.train(params, eventContainer.dtrain, options.tree_number, evallist, early_stopping_rounds = 10)

    # Store BDT
    output = open(options.out_name+'_'+str(bdt_num)+'/' + \
            options.out_name+'_'+str(bdt_num)+'_weights.pkl', 'wb')
    pkl.dump(gbm, output)

    # Plot feature importances
    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_name+'_'+str(bdt_num)+"/" + \
            options.out_name+'_'+str(bdt_num)+'_fimportance.png', # png file name
            dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
    
    # Closing statment
    print("Files saved in: ", options.out_name+'_'+str(bdt_num))
