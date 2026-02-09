#!/usr/bin/python
import argparse
import importlib
import os
import math
import sys
import random
import ROOT as r
#import matplotlib as plt
from matplotlib import pyplot as plt
import xgboost as xgb
import pickle as pkl
import numpy as np
from rootpy.tree import Tree, TreeModel, TreeChain, FloatCol, IntCol, BoolCol, FloatArrayCol
from rootpy.io import root_open

from collections import Counter
from array import array
sys.path.insert(0, '../')
from sklearn import metrics

####################################################################################
class sampleContainer:
    def __init__(self, fns, outn, outd, model):
        print "Initializing Container!"
        #self.tin = r.TChain('EcalVeto')
        #for fn in fns:
        #    self.tin.Add(fn)
        #self.tfile = root_open(fn,'r+')
        #with root_open(fn,'r+') as f:
        #self.tin = self.tfile.EcalVeto
        #self.tin.Print()
        self.tin = TreeChain('EcalVeto',fns)
        self.tin.create_branches({'discValue_gabrielle' : 'F'})
        self.model = model

        self.outdir = outd
        self.outname = outn
        self.outfile = root_open(outn, 'RECREATE')
        self.tout = Tree('EcalVeto')
        self.tout.set_buffer(self.tin._buffer, create_branches=True)

        self.events =  []
        #print self.tin.GetEntries()
        for event in self.tin:
            #if len(self.events)>10:
            #    continue
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
            #new features
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

            evtarray = np.array([evt])
            pred = float(model.predict(xgb.DMatrix(evtarray))[0])
            #print pred
            event.discValue_gabrielle = pred
            self.tout.Fill()
######################################################################################
            self.events.append(evt)

            if (len(self.events)%10000 == 0 and len(self.events) > 0):
                print 'The shape of events = ', np.shape(self.events)

        self.outfile.cd()
        self.tout.Write()

        self.outfile.Close()
        #self.tfile.Close()
        print 'cp %s %s' % (self.outname,self.outdir)
        os.system('cp %s %s' % (self.outname,self.outdir))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Make tree with BDT result from inputs')


    parser.add_argument('--interactive', dest='interactive', action='store_true', help='Run in interactive mode [Default: False]')
    parser.add_argument('-o','--outfile', dest='out_file',  default='bdttest/bkg_bdteval.root', help='Output file name')
    parser.add_argument('--outdir', dest='outdir', default='/nfs/slac/g/ldmx/users/vdutta/test', help='Name of output directory')
    parser.add_argument('--swdir', dest='swdir',  default='/nfs/slac/g/ldmx/users/vdutta/ldmx-sw-1.7/ldmx-sw-install', help='ldmx-sw build directory')
    parser.add_argument('--signal', dest='issignal', action='store_true', help='Signal file [Default: False]')
    parser.add_argument('--eta', dest='eta',type=float,  default=0.023, help='Learning Rate')
    parser.add_argument('--tree_number', dest='tree_number',type=int,  default=1000, help='Tree Number')
    parser.add_argument('--depth', dest='depth',type=int,  default=10, help='Max Tree Depth')
    parser.add_argument('-i','--in_files', dest='in_files', nargs='*', default=['bdttrain/bkg_bdttest_subtree.root'], help='list of input files')
    parser.add_argument('-p','--pkl_file', dest='pkl_file', default='/nfs/slac/g/ldmx/users/vdutta/ldmx-sw/test/bdt_gabrielle/bdt_gabrielle_0_weights.pkl', help='name of BDT pkl file')
    parser.add_argument('--filelist', dest='filelist', default = '', help='Text file with list of input files')
  

    args = parser.parse_args()


    model = pkl.load(open(args.pkl_file,'rb'))
    #plt.figure()
    #xgb.plot_importance(model)
    #plt.show()

    inputfiles = []
    # If an input file list is provided, read from that
    if args.filelist != '':
        print 'Loading input files from',args.filelist
        with open(args.filelist,'r') as f:
            inputfiles = f.read().splitlines()
    else:
        inputfiles = args.in_files

    # Create output directory if it doesn't already exist
    print 'Creating output directory %s' % args.outdir
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
   
    # Create the scratch directory if it doesn't already exist
    scratch_dir = '%s/%s' % (os.getcwd(),os.environ['USER']) if args.interactive else '/scratch/%s' % os.environ['USER']
    print 'Using scratch path %s' % scratch_dir
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
  
    # Create a tmp directory that can be used to copy files into
    tmp_dir = '%s/%s' % (scratch_dir, 'tmp') if args.interactive else '%s/%s' % (scratch_dir, os.environ['LSB_JOBID'])
    print 'Creating tmp directory %s' % tmp_dir
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    os.chdir(tmp_dir)
   
    # Copy input files to the tmp directory
    print 'Copying input files into tmp directory'
    for f in inputfiles:
        os.system("ls %s" % f)
        os.system("cp %s ." % f )
    os.system("ls .")

    # Just get the file names without the full path
    localfiles = [f.split('/')[-1] for f in inputfiles]
  
    eventContainer = sampleContainer(localfiles,args.out_file,args.outdir,model)

    # Remove tmp directory
    print 'Removing tmp directory %s' % tmp_dir
    os.system('rm -rf %s' % tmp_dir)

print "Files saved in: ", args.out_file#+'_'+str(adds)
