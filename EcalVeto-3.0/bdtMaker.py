import os
import sys
import logging
import glob
import argparse
import ROOT as r
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib as plt
from array    import array
from optparse import OptionParser
r.gSystem.Load('libFramework.so')

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
plt.use('Agg')

def addBranch(tree, ldmx_class, branch_name):

    # Add a new branch to read from

    if tree == None:
        sys.exit('Please set tree!')

    if ldmx_class == 'EventHeader': branch = r.ldmx.EventHeader()
    elif ldmx_class == 'EcalVetoResult': branch = r.ldmx.EcalVetoResult()
    elif ldmx_class == 'HcalVetoResult': branch = r.ldmx.HcalVetoResult()
    elif ldmx_class == 'TriggerResult': branch = r.ldmx.TriggerResult()
    elif ldmx_class == 'SimParticle': branch = r.std.map(int, 'ldmx::'+ldmx_class)()
    else: branch = r.std.vector('ldmx::'+ldmx_class)()

    tree.SetBranchAddress(branch_name,r.AddressOf(branch)) # sets relevant branch addresses in the TTrees

    return branch

class sampleContainer:
    def __init__(self,filenames,maxEvts,trainFrac,isSig):

        print("Initializing Container!")
        self.tree = r.TChain("LDMX_Events") # initialize TChain of TTrees which have name "LDMX_Events"
        for file in filenames:
            self.tree.Add(file) # adds input file trees to TChain
        self.maxEvts = maxEvts
        self.trainFrac = trainFrac
        self.isSig   = isSig
        print(f'Assigning branch address to EcalVetoResult')
        self.ecalVeto = addBranch(self.tree, 'EcalVetoResult', 'EcalVeto_{}'.format('SegmipBDTReco'))
        if self.isSig:
            print(f'Assigning branch address to TriggerResult')
            self.trigger = addBranch(self.tree, 'TriggerResult', '2eTrigger_{}'.format('overlay'))

    def root2PyEvents(self):
        self.events =  []
        print(f'Loading {self.tree.GetEntries()} events')
        # TChain.GetEntries() should return all entries across all TTrees in the TChain
        for event_count in range(self.tree.GetEntries()): # iterates through no. of events
            
            # load event indexed by event_count
            self.tree.GetEntry(event_count)
            
            if len(self.events) >= self.maxEvts: # skips rest of for loop past maxEvts
                break
            
            if self.isSig: # skips signal events which failed the trigger
                if not self.trigger.passed(): 
                    print(f'Event {event_count} : failed trigger; skipped')
                    continue
            
            result = self.ecalVeto

            evt = [
                    # Base variables
                    result.getNReadoutHits(),
                    result.getSummedDet(),
                    result.getSummedTightIso(),
                    result.getMaxCellDep(),
                    result.getShowerRMS(),
                    result.getXStd(),
                    result.getYStd(),
                    result.getAvgLayerHit(),
                    result.getStdLayerHit(),
                    result.getDeepestLayerHit(),
                    result.getEcalBackEnergy(),
                    # MIP Tracking variables
                    result.getNStraightTracks(),
                    result.getFirstNearPhLayer(),
                    result.getNNearPhHits(),
                    result.getPhotonTerritoryHits(),
                    result.getEPSep(),
                    result.getEPDot(),
                    # Longitudinal segment variables
                    result.getEnergySeg()[0],
                    result.getXMeanSeg()[0],
                    result.getYMeanSeg()[0],
                    result.getLayerMeanSeg()[0],
                    result.getEnergySeg()[1],
                    result.getYMeanSeg()[2],
                    # Electron RoC variables
                    result.getEleContEnergy()[0][0],
                    result.getEleContEnergy()[1][0],
                    result.getEleContYMean()[0][0],
                    result.getEleContEnergy()[0][1],
                    result.getEleContEnergy()[1][1],
                    result.getEleContYMean()[0][1],
                    # Photon RoC variables
                    result.getPhContNHits()[0][0],
                    result.getPhContYMean()[0][0],
                    result.getPhContNHits()[0][1],
                    # Outside RoC variables
                    result.getOutContEnergy()[0][0],
                    result.getOutContEnergy()[1][0],
                    result.getOutContEnergy()[2][0],
                    result.getOutContNHits()[0][0],
                    result.getOutContXMean()[0][0],
                    result.getOutContYMean()[0][0],
                    result.getOutContYMean()[1][0],
                    result.getOutContYStd()[0][0],
                    result.getOutContEnergy()[0][1],
                    result.getOutContEnergy()[1][1],
                    result.getOutContEnergy()[2][1],
                    result.getOutContLayerMean()[0][1],
                    result.getOutContLayerStd()[0][1],
                    result.getOutContEnergy()[0][2],
                    result.getOutContLayerMean()[0][2],      
            ]

            self.events.append(evt)
            print(f'Event {event_count} : event data loaded successfully')

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
    parser.add_option('--seed', dest='seed',type="int",  default=1, help='Numpy random seed.')
    parser.add_option('--max_evt', dest='max_evt',type="int",  default=1500000, help='Max Events to load')
    parser.add_option('--train_frac', dest='train_frac',  default=.9, help='Fraction of events to use for training')
    parser.add_option('--eta', dest='eta',type="float",  default=0.15, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number',type="int",  default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth',type="int",  default=10, help='Max Tree Depth')
    parser.add_option('-b', dest='bkg_path', help='Absolute path to background file(s)')
    parser.add_option('-s', dest='sig_path', help='Absolute path to signal file(s)')
    parser.add_option('-o', dest='out_name',  default='bdt_test', help='Output Pickle Name (excluding file extension)')
    (options, args) = parser.parse_args()

    # pulls all files from the indicated input directories
    if os.path.isdir(options.bkg_path):
        bkg_files = glob.glob(os.path.join(options.bkg_path, '*.root'))
    elif os.path.isfile(options.bkg_path):
        bkg_files = [options.bkg_path]
    else:
        print(f'os.path cannot read bkg_path as either a file or directory')
        sys.exit(f'Problem reading bkg_path\nbkg_path = {options.bkg_path}\nPlease check inputted parameters!')
    if os.path.isdir(options.sig_path):
        sig_files = glob.glob(os.path.join(options.sig_path, '*.root'))
    elif os.path.isfile(options.sig_path):
        sig_files = [options.sig_path]
    else:
        print(f'os.path cannot read sig_path as either a file or directory')
        sys.exit(f'Problem reading sig_path\nsig_path = {options.sig_path}\nPlease check inputted parameters!')

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
    print( 'Loading sig_path = {}'.format(options.sig_path) )
    sigContainer = sampleContainer(sig_files,options.max_evt,options.train_frac,True)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    # Make Background Container
    print( 'Loading bkg_path = {}'.format(options.bkg_path) )
    bkgContainer = sampleContainer(bkg_files,options.max_evt,options.train_frac,False)
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
               'nthread': 30,
               'verbosity': 1
               # 'early_stopping_rounds' : 10
    }

    # Train the BDT model
    evallist = [(eventContainer.dtrain,'train'), (eventContainer.dtest,'eval')]
    gbm = xgb.train(params, eventContainer.dtrain, num_boost_round = options.tree_number, evals = evallist, early_stopping_rounds = 10)
    
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
