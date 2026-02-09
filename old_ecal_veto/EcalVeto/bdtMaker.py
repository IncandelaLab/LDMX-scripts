#!/usr/bin/python
import argparse
import importlib
import os
import math
import sys
import random
import ROOT as r
import matplotlib as plt
import xgboost as xgb
import pickle as pkl
import numpy as np
from rootpy.tree import Tree, TreeModel, TreeChain, FloatCol, IntCol, BoolCol, FloatArrayCol
from rootpy.io import root_open

plt.use('Agg')
from collections import Counter
from array import array
from optparse import OptionParser
sys.path.insert(0, '../')
from sklearn import metrics

####################################################################################
class sampleContainer:
    def __init__(self, fn,maxEvts,trainFrac,isBkg,iseBkg,iseSig):
        print "Initializing Container!"
        #self.tin = r.TChain("EcalVeto")
        #self.tin.Add(fn)
        self.tfile = root_open(fn,'r+')
        self.tin = self.tfile.EcalVeto
        #self.tin.Print()

        self.maxEvts   = maxEvts
        self.trainFrac = trainFrac
        self.isBkg = isBkg
	self.iseBkg = iseBkg
	self.iseSig = iseSig


    def root2PyEvents(self):
        self.events =  []
        #print self.tin.GetEntries()
        for event in self.tin:
            if len(self.events) >= self.maxEvts:
                continue
            evt = []
            

################################### Features #######################################

            
            evt.append(event.nReadoutHits)
            evt.append(event.summedDet)
            evt.append(event.summedTightIso)
            evt.append(event.maxCellDep)
            evt.append(event.showerRMS)
            evt.append(event.xStd)
            evt.append(event.yStd)
            evt.append(event.avgLayerHit)
            evt.append(event.deepestLayerHit)
            evt.append(event.stdLayerHit)
            evt.append(event.ele68ContEnergy)
            evt.append(event.ele68x2ContEnergy)
            evt.append(event.ele68x3ContEnergy)
            evt.append(event.ele68x4ContEnergy)
            evt.append(event.ele68x5ContEnergy)
            evt.append(event.photon68ContEnergy)
            evt.append(event.photon68x2ContEnergy)
            evt.append(event.photon68x3ContEnergy)
            evt.append(event.photon68x4ContEnergy)
            evt.append(event.photon68x5ContEnergy)
            evt.append(event.outside68ContEnergy)
            evt.append(event.outside68x2ContEnergy)
            evt.append(event.outside68x3ContEnergy)
            evt.append(event.outside68x4ContEnergy)
            evt.append(event.outside68x5ContEnergy)
            evt.append(event.outside68ContNHits)
            evt.append(event.outside68x2ContNHits)
            evt.append(event.outside68x3ContNHits)
            evt.append(event.outside68x4ContNHits)
            evt.append(event.outside68x5ContNHits)
            evt.append(event.outside68ContXstd)
            evt.append(event.outside68x2ContXstd)
            evt.append(event.outside68x3ContXstd)
            evt.append(event.outside68x4ContXstd)
            evt.append(event.outside68x5ContXstd)
            evt.append(event.outside68ContYstd)
            evt.append(event.outside68x2ContYstd)
            evt.append(event.outside68x3ContYstd)
            evt.append(event.outside68x4ContYstd)
            evt.append(event.outside68x5ContYstd)
            evt.append(event.ecalBackEnergy)


######################################################################################
            self.events.append(evt)

            if (len(self.events)%10000 == 0 and len(self.events) > 0):
                print 'The shape of events = ', np.shape(self.events)


        new_idx=np.random.permutation(np.arange(np.shape(self.events)[0]))
        self.events = np.array(self.events)
        np.take(self.events, new_idx, axis=0, out=self.events)
        print "Final Event Shape", np.shape(self.events)
        self.tfile.Close()

    def constructTrainAndTest(self):
        self.train_x = self.events[0:int(len(self.events)*self.trainFrac)]
        self.test_x  = self.events[int(len(self.events)*self.trainFrac):]
        
        self.train_y = np.zeros(len(self.train_x)) + (self.isBkg == False)
        self.test_y  = np.zeros(len(self.test_x)) + (self.isBkg == False)

class mergedContainer:
    def __init__(self, sigContainer,bkgContainer):
        self.train_x = np.vstack((sigContainer.train_x,bkgContainer.train_x))
        self.train_y = np.append(sigContainer.train_y,bkgContainer.train_y)
        
        self.train_x[np.isnan(self.train_x)] = 0.000
        self.train_y[np.isnan(self.train_y)] = 0.000
        
        self.test_x  = np.vstack((sigContainer.test_x,bkgContainer.test_x))
        self.test_y  = np.append(sigContainer.test_y,bkgContainer.test_y)
        
        self.dtrain = xgb.DMatrix(self.train_x,self.train_y,\
                                    weight = self.getEventWeights(sigContainer.train_y,bkgContainer.train_y))
        self.dtest  = xgb.DMatrix(self.test_x,self.test_y)
    


    #Class for re-weighting bkg to signal
    def getEventWeights(self,sig,bkg):
        sigWgt = np.zeros(len(sig)) + 1
        bkgWgt = np.zeros(len(bkg)) + 1. * float(len(sig))/float(len(bkg))
        return np.append(sigWgt,bkgWgt)

if __name__ == "__main__":
    
    parser = OptionParser()


    parser.add_option('--seed', dest='seed',type="int",  default=2, help='Numpy random seed.')
    parser.add_option('--train_frac', dest='train_frac',  default=.8, help='Fraction of events to use for training')
    parser.add_option('--max_evt', dest='max_evt',type="int",  default=1250000, help='Max Events to load')
    parser.add_option('--out_name', dest='out_name',  default='bdt_gabrielle', help='Output Pickle Name')
    parser.add_option('--swdir', dest='swdir',  default='../ldmx-sw-install', help='ldmx-sw build directory')
    parser.add_option('--eta', dest='eta',type="float",  default=0.023, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number',type="int",  default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth',type="int",  default=10, help='Max Tree Depth')
    parser.add_option('--bkg_file', dest='bkg_file', default='bdttrain/bkg_bdttrain_tree.root', help='name of background file')
    parser.add_option('--sig_file', dest='sig_file', default='bdttrain/signal_bdttrain_tree.root', help='name of signal file')
  

    (options, args) = parser.parse_args()

    np.random.seed(options.seed)
    
    adds=0
    Check=True
    while Check:
        if not os.path.exists(options.out_name+'_'+str(adds)):
	    try:
                os.makedirs(options.out_name+'_'+str(adds))
	        Check=False
	    except:
	        Check=True
        else:
            adds+=1


    print "Random seed is = %s" % (options.seed)
    print "You set max_evt = %s" % (options.max_evt)
    print "You set train frac = %s" % (options.train_frac)
    print "You set tree number = %s" % (options.tree_number)
    print "You set max tree depth = %s" % (options.depth)
    print "You set eta = %s" % (options.eta)

    print "Loading library file from %s" % (options.swdir+"/lib/libEvent.so")
    r.gSystem.Load(options.swdir+"/lib/libEvent.so")


    print 'Loading sig_file = %s' % (options.sig_file)
    sigContainer = sampleContainer(options.sig_file,options.max_evt,options.train_frac,False,False,False)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    print 'Loading bkg_file = %s' % (options.bkg_file)
    bkgContainer = sampleContainer(options.bkg_file,options.max_evt,options.train_frac,True,False,False)
    bkgContainer.root2PyEvents()
    bkgContainer.constructTrainAndTest()

    eventContainer = mergedContainer(sigContainer,bkgContainer)

    params     = {"objective": "binary:logistic",
                "eta": options.eta,
                "max_depth": options.depth,
                "min_child_weight": 20,
                "silent": 1,
                "subsample": .9,
                "colsample_bytree": .85,
                #"eval_metric": 'auc',
                "eval_metric": 'error',
                "seed": 1,
                "nthread": 1,
                "verbosity": 1,
                "early_stopping_rounds" : 10}

    num_trees  = options.tree_number
    evallist  = [(eventContainer.dtest,'eval'), (eventContainer.dtrain,'train')]
    gbm       = xgb.train(params, eventContainer.dtrain, num_trees,evallist)


    preds = gbm.predict(eventContainer.dtest)
    fpr, tpr, threshold = metrics.roc_curve(eventContainer.test_y, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print 'Final Validation AUC = %s' % (roc_auc)
    np.savetxt(options.out_name+'_'+str(adds)+"/"+options.out_name+'_'+str(adds)+'_validation_preds.txt',preds)
    np.savetxt(options.out_name+'_'+str(adds)+"/"+options.out_name+'_'+str(adds)+'_validation_threetuples.txt',np.c_[fpr,tpr,threshold])
    output = open(options.out_name+'_'+str(adds)+"/"+options.out_name+'_'+str(adds)+"_weights"+'.pkl', 'wb')
    pkl.dump(gbm, output)

    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_name+'_'+str(adds)+"/"+options.out_name+'_'+str(adds)+'_fimportance.png', dpi=500, bbox_inches='tight', pad_inches=0.5)

print "Files saved in: ", options.out_name+'_'+str(adds)
