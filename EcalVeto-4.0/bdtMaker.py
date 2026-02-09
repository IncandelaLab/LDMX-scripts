import os, sys, logging
import glob
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib as plt
import pandas as pd
from optparse import OptionParser
import shap
import uproot
import awkward as ak

print('All packages loaded!')

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
plt.use('Agg')

class sampleContainer:
    def __init__(self,filenames,maxEvts,trainFrac,isSig):

        print("Initializing Container!")
        self.filenames = filenames
        self.maxEvts = maxEvts
        self.trainFrac = trainFrac
        self.isSig   = isSig
        
        # Define branch names based on signal/background
        if self.isSig:
            self.ecalVeto_branch = 'EcalVeto_test'
            self.trigger_branch = 'Trigger_sim'
            # self.recoil_fiducial_branch = 'RecoilTruthFiducialFlags_test'
            self.tagger_branch = 'TaggerTracksClean_test'
        else:
            self.ecalVeto_branch = 'EcalVeto_test'

    def root2PyEvents(self):
        """
        Loads events using uproot and awkward arrays, formatting them as a Numpy array.
        """
        print(f'Loading events from {len(self.filenames)} file(s)')
        
        # Load all files and concatenate
        all_events = []
        total_loaded = 0
        
        for file_path in self.filenames:
            if total_loaded >= self.maxEvts:
                break
                
            with uproot.open(file_path) as file:
                tree = file["LDMX_Events"]
                
                # Determine how many events to load from this file
                n_entries = tree.num_entries
                n_to_load = min(n_entries, self.maxEvts - total_loaded)
                
                print(f'Loading {n_to_load} events from {os.path.basename(file_path)}')
                
                # Define branches to load based on signal/background
                branches_to_load = [self.ecalVeto_branch]
                if self.isSig:
                    branches_to_load.extend([self.trigger_branch])
                
                # Load the branches as awkward arrays
                arrays = tree.arrays(branches_to_load, entry_stop=n_to_load, library="ak")
                
                # Apply trigger cuts for signal
                if self.isSig:
                    # Access trigger passed status
                    trigger_passed = arrays[self.trigger_branch].pass_
                    
                    # Apply cuts
                    mask = trigger_passed
                    arrays = arrays[mask]
                    
                    print(f'After trigger cut: {ak.num(arrays, axis=0)} events remain')
                
                # Extract EcalVeto features
                ecal = arrays[self.ecalVeto_branch]
                
                # Build event array with all features
                events_array = np.column_stack([
                    # Base variables
                    ak.to_numpy(ecal.n_readout_hits_),
                    ak.to_numpy(ecal.summed_det_),
                    ak.to_numpy(ecal.summed_tight_iso_),
                    ak.to_numpy(ecal.max_cell_dep_),
                    ak.to_numpy(ecal.shower_rms_),
                    ak.to_numpy(ecal.x_std_),
                    ak.to_numpy(ecal.y_std_),
                    ak.to_numpy(ecal.avg_layer_hit_),
                    ak.to_numpy(ecal.std_layer_hit_),
                    ak.to_numpy(ecal.deepest_layer_hit_),
                    ak.to_numpy(ecal.ecal_back_energy_),
                    ak.to_numpy(ecal.ep_sep_),
                    ak.to_numpy(ecal.ep_dot_),
                    # Longitudinal segment variables
                    ak.to_numpy(ecal.energy_seg_[:, 0]),
                    ak.to_numpy(ecal.x_mean_seg_[:, 0]),
                    ak.to_numpy(ecal.y_mean_seg_[:, 0]),
                    ak.to_numpy(ecal.layer_mean_seg_[:, 0]),
                    ak.to_numpy(ecal.energy_seg_[:, 1]),
                    ak.to_numpy(ecal.y_mean_seg_[:, 2]),
                    # Electron RoC variables
                    ak.to_numpy(ecal.e_cont_energy_[:, 0, 0]),
                    ak.to_numpy(ecal.e_cont_energy_[:, 1, 0]),
                    ak.to_numpy(ecal.e_cont_y_mean_[:, 0, 0]),
                    ak.to_numpy(ecal.e_cont_energy_[:, 0, 1]),
                    ak.to_numpy(ecal.e_cont_energy_[:, 1, 1]),
                    ak.to_numpy(ecal.e_cont_y_mean_[:, 0, 1]),
                    # Photon RoC variables
                    ak.to_numpy(ecal.g_cont_n_hits_[:, 0, 0]),
                    ak.to_numpy(ecal.g_cont_y_mean_[:, 0, 0]),
                    ak.to_numpy(ecal.g_cont_n_hits_[:, 0, 1]),
                    # Outside RoC variables
                    ak.to_numpy(ecal.o_cont_energy_[:, 0, 0]),
                    ak.to_numpy(ecal.o_cont_energy_[:, 1, 0]),
                    ak.to_numpy(ecal.o_cont_energy_[:, 2, 0]),
                    ak.to_numpy(ecal.o_cont_n_hits_[:, 0, 0]),
                    ak.to_numpy(ecal.o_cont_x_mean_[:, 0, 0]),
                    ak.to_numpy(ecal.o_cont_y_mean_[:, 0, 0]),
                    ak.to_numpy(ecal.o_cont_y_mean_[:, 1, 0]),
                    ak.to_numpy(ecal.o_cont_y_std_[:, 0, 0]),
                    ak.to_numpy(ecal.o_cont_energy_[:, 0, 1]),
                    ak.to_numpy(ecal.o_cont_energy_[:, 1, 1]),
                    ak.to_numpy(ecal.o_cont_energy_[:, 2, 1]),
                    ak.to_numpy(ecal.o_cont_layer_mean_[:, 0, 1]),
                    ak.to_numpy(ecal.o_cont_layer_std_[:, 0, 1]),
                    ak.to_numpy(ecal.o_cont_energy_[:, 0, 2]),
                    ak.to_numpy(ecal.o_cont_layer_mean_[:, 0, 2]),
                ])
                
                all_events.append(events_array)
                total_loaded += len(events_array)
                
                print(f'Loaded {len(events_array)} events, total: {total_loaded}')
        
        # Concatenate all events
        self.events = np.vstack(all_events) if len(all_events) > 1 else all_events[0]
        
        # Shuffle events
        new_idx = np.random.permutation(np.arange(len(self.events)))
        self.events = self.events[new_idx]
        
        print(f"Final Event Shape: {self.events.shape}")

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
                'epSep', # f11
                'epDot', # f12
                # Longitudinal segment variables
                'energy_s1', # f13
                'xMean_s1', # f14
                'yMean_s1', # f15
                'layerMean_s1', # f16
                'energy_s2', # f17
                'yMean_s2', # f18
                # Electron RoC variables
                'eContEnergy_x1_s1', # f19
                'eContEnergy_x2_s1', # f20
                'eContYMean_x1_s1', # f21
                'eContEnergy_x1_s2', # f22
                'eContEnergy_x2_s2', # f23
                'eContYMean_x1_s2', # f24
                # Photon RoC variables
                'gContNHits_x1_s1', # f25
                'gContYMean_x1_s1', # f26
                'gContNHits_x1_s2', # f27
                # Outside RoC variables
                'oContEnergy_x1_s1', # f28
                'oContEnergy_x2_s1', # f29
                'oContEnergy_x3_s1', # f30
                'oContNHits_x1_s1', # f31
                'oContXMean_x1_s1', # f32
                'oContYMean_x1_s1', # f33
                'oContYMean_x2_s1', # f34
                'oContYStd_x1_s1', # f35
                'oContEnergy_x1_s2', # f36
                'oContEnergy_x2_s2', # f37
                'oContEnergy_x3_s2', # f38
                'oContLayerMean_x1_s2', # f39
                'oContLayerStd_x1_s2', # f40
                'oContEnergy_x1_s3', # f41
                'oContLayerMean_x1_s3', # f42
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

    bar_mean_outpath = os.path.join(out_dir, 'model_bar_shapMeanAbs.png')
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
