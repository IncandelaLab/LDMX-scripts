import os, sys, logging
import glob
import ROOT as r
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib as plt
import pandas as pd
from optparse import OptionParser
import shap
r.gSystem.Load('libFramework.so')

print('All packages loaded!')

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
            self.trigger = addBranch(self.tree, 'TriggerResult', '2eTrigger_{}'.format('2e-signal10mevOverlay-8gev-1M'))


    def root2PyEvents(self):
        """
        Loads events in the TChain up to maxEvts, formatting them as a Numpy array.
        """
        self.events =  []
        print(f'Loading {self.tree.GetEntries()} events')
        # TChain.GetEntries() should return all entries across all TTrees in the TChain
        for event_count in range(self.tree.GetEntries()): # iterates through no. of events
            
            # load event indexed by event_count
            self.tree.GetEntry(event_count)
            
            if len(self.events) >= self.maxEvts: # skips rest of for loop past maxEvts
                break

            if self.isSig: # skips signal events which failed the trigger
                # print(f'nElectrons = {self.trigger.getAlgoVar(3)}') # prints nElectrons counted by trigger
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
        """
        Splits data in self.events into training and testing subsets.
        """

        self.train_x = self.events[0:int(len(self.events)*self.trainFrac)]
        self.test_x = self.events[int(len(self.events)*self.trainFrac):]

        self.train_y = np.zeros(len(self.train_x)) + (self.isSig == True)
        self.test_y = np.zeros(len(self.test_x)) + (self.isSig == True)

class mergedContainer:
    """
    Merges signal and background sampleContainers into one.
    """
    def __init__(self, sigContainer,bkgContainer):
        # defines names of the features for better labeling
        self.feat_names = [
                # Base variables
                'nReadoutHits', # f0
                'summedDet', # f1
                'summedTightIso', # f2
                'maxCellDep', # f3
                'showerRMS', # f4
                'xStd', # f5
                'yStd', # f6
                'avgLayerHit', # f7
                'stdLayerHit', # f8
                'deepestLayerHit', # f9
                'ecalBackEnergy', # f10
                # MIP Tracking variables
                'nStraightTracks', # f11
                'firstNearPhLayer', # f12
                'nNearPhHits', # f13
                'photonTerritoryHits', # f14
                'epSep', # f15
                'epDot', # f16
                # Longitudinal segment variables
                'energy_s1', # f17
                'xMean_s1', # f18
                'yMean_s1', # f19
                'layerMean_s1', # f20
                'energy_s2', # f21
                'yMean_s2', # f22
                # Electron RoC variables
                'eContEnergy_x1_s1', # f23
                'eContEnergy_x2_s1', # f24
                'eContYMean_x1_s1', # f25
                'eContEnergy_x1_s2', # f26
                'eContEnergy_x2_s2', # f27
                'eContYMean_x1_s2', # f28
                # Photon RoC variables
                'gContNHits_x1_s1', # f29
                'gContYMean_x1_s1', # f30
                'gContNHits_x1_s2', # f31
                # Outside RoC variables
                'oContEnergy_x1_s1', # f32
                'oContEnergy_x2_s1', # f33
                'oContEnergy_x3_s1', # f34
                'oContNHits_x1_s1', # f35
                'oContXMean_x1_s1', # f36
                'oContYMean_x1_s1', # f37
                'oContYMean_x2_s1', # f38
                'oContYStd_x1_s1', # f39
                'oContEnergy_x1_s2', # f40
                'oContEnergy_x2_s2', # f41
                'oContEnergy_x3_s2', # f42
                'oContLayerMean_x1_s2', # f43
                'oContLayerStd_x1_s2', # f44
                'oContEnergy_x1_s3', # f45
                'oContLayerMean_x1_s3', # f46
            ]
        
        # merges train_x, train_y data
        self.train_x = np.vstack((sigContainer.train_x,bkgContainer.train_x))
        self.train_y = np.append(sigContainer.train_y,bkgContainer.train_y)
        
        # corrects for any NaN values in the data
        self.train_x[np.isnan(self.train_x)] = 0.000
        self.train_y[np.isnan(self.train_y)] = 0.000

        # merges test_x, test_y data
        self.test_x  = np.vstack((sigContainer.test_x,bkgContainer.test_x))
        self.test_y  = np.append(sigContainer.test_y,bkgContainer.test_y)
        
        # formats data for training
        self.dtrain = xgb.DMatrix(self.train_x,self.train_y,feature_names=self.feat_names)
        self.dtest  = xgb.DMatrix(self.test_x,self.test_y,feature_names=self.feat_names)

        # formats data for export as CSV
        self.train_x_df = pd.DataFrame(self.train_x, columns=self.feat_names)
        self.test_x_df = pd.DataFrame(self.test_x, columns=self.feat_names)

    def saveEventsAsCsv(self, out_dir):
        """
        Saves training and testing data as CSV files to the BDT output directory.
        """

        self.train_x_df.to_csv(os.path.join(out_dir, 'model_train_events.csv'), index=False)
        print( 'Exported training data at {}'.format(os.path.join(out_dir, 'model_train_events.csv')) )
        self.test_x_df.to_csv(os.path.join(out_dir, 'model_test_events.csv'), index=False)
        print( 'Exported training data at {}'.format(os.path.join(out_dir, 'model_test_events.csv')) )

def calcShap(evt_container:mergedContainer, model, out_dir:str):
    """
    Uses test data and model to calculate SHAP values for the BDT. 
    Saves SHAP values as a npy file, and returns a numpy array containing
    the values.
    """
    x_frame = evt_container.test_x_df

    # picks n rows at random from the dataframe to initalize shap.TreeExplainer container
    sample = x_frame.sample(n=100, random_state=42) # leaving random_state unset will make it truly random; enter value for repeatability
    
    explainer = shap.TreeExplainer(model, sample)
    print(f'shap.TreeExplainer object created successfully')

    print("Calculating shap values")
    out_path = os.path.join(out_dir, 'model_shapValues.npy')

    # calculates shap values and returns a numpy array, saving it to out_dir
    # for large datasets, this should be a rather time consuming process
    # 2 hrs for ~ 1M events
    shap_values = explainer.shap_values(x_frame)
    np.save(out_path, shap_values)

    print(f'Shap values calculated!\nSaved at : {out_path}')

    return shap_values

def plotShap(evt_container:mergedContainer, shap_values, out_dir:str):
    data = evt_container.test_x_df
    num_features = len(data.columns)

    shap_container = shap.Explanation(shap_values, data=data, feature_names=data.columns)

    bar_mean_outpath = os.path.join(out_dir, 'model_bar_shapMeanAbs.jpg')
    shap.plots.bar(shap_container, max_display=num_features, show=False)
    plt.pyplot.savefig(bar_mean_outpath, bbox_inches='tight', dpi=500)
    plt.pyplot.close()
    print(f'SHAP summary bar plot with {num_features} features saved at {bar_mean_outpath}')

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
    parser.add_option('--csv', action='store_true', dest='csv_bool', help='Whether or not to export the training and testing event features as a CSV file')
    parser.add_option('--shap', action='store_true', dest='shap_bool', help='Whether or not to plot the SHAP values summary for the models performance on the test data')
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
    outpath = options.out_name + '_'
    Check=True
    while Check:
        if not os.path.exists(outpath+str(bdt_num)):
            try:
                os.makedirs(outpath+str(bdt_num))
                Check=False
            except:
               Check=True
        else:
            bdt_num+=1
    
    outpath = outpath + str(bdt_num)

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
    print( 'Merging signal and background containers' )
    eventContainer = mergedContainer(sigContainer,bkgContainer)
    
    if options.csv_bool:
        print( 'Exporting training and testing events as CSV file' )
        eventContainer.saveEventsAsCsv(outpath)

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
    output = open(outpath+'/' + \
            outpath+'_weights.pkl', 'wb') # output file name
    pkl.dump(gbm, output)

    # Calculate and plot shap values
    if options.shap_bool:
        print( 'Calculating SHAP values...' )
        shap_values = calcShap(eventContainer, gbm, outpath)
        print( 'Plotting SHAP summary...' )
        plotShap(eventContainer, shap_values, outpath)


    # Plot feature importances
    figure_path = outpath+'_fimportance.png'
    xgb.plot_importance(gbm)
    plt.pyplot.savefig(outpath+"/" + \
            figure_path, # png file name
            dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
    

    print(f'Feature importances saved under {figure_path}')

    # Closing statement
    print("All files saved in: ", outpath)

if __name__ != "__main__":
    print(f'Imported {__file__}!')
