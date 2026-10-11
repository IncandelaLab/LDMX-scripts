import os, sys, logging
import datetime
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
    def __init__(self,filenames,maxEvts,trainFrac,isSig,bkg_branch='EcalVeto_Recoil_Tracking',sig_branch='EcalVeto_reco_v1'):

        print("Initializing Container!")
        self.filenames = filenames
        self.maxEvts = maxEvts
        self.trainFrac = trainFrac
        self.isSig   = isSig

        # Define branch names based on signal/background
        if self.isSig:
            self.ecalVeto_branch = sig_branch
            self.trigger_branch = 'Trigger_sim'
        else:
            self.ecalVeto_branch = bkg_branch

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
                # Skip truncated/corrupt files (no LDMX_Events tree) instead of crashing
                if not any('LDMX_Events' in k for k in file.keys()):
                    print(f'[WARNING] Skipping corrupt file (no LDMX_Events): {os.path.basename(file_path)}')
                    continue
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
                    ak.to_numpy(ecal.n_tracking_hits_),
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

        frac = float(self.trainFrac)
        self.train_x = self.events[0:int(len(self.events)*frac)]
        self.test_x = self.events[int(len(self.events)*frac):]

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
                'n_tracking_hits', # f13
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

def plotLearningCurve(evals_result: dict, out_dir: str, metric: str = 'error'):
    """
    Plots the training and evaluation learning curves from the evals_result
    dict populated by xgb.train() and saves the figure to out_dir.
    Saves both a linear-scale and a log-scale version.

    Parameters
    ----------
    evals_result : dict filled in-place by xgb.train(evals_result=...)
                   structure: {'train': {metric: [...]}, 'eval': {metric: [...]}}
    out_dir      : directory where the PNG will be written
    metric       : the eval metric key to plot (default 'error', matches params)
    """
    import json
    train_metric = evals_result['train'][metric]
    eval_metric  = evals_result['eval'][metric]

    # Save raw history so it can be re-plotted later
    json_path = os.path.join(out_dir, f'evals_result_{metric}.json')
    with open(json_path, 'w') as jf:
        json.dump({'train': {metric: train_metric}, 'eval': {metric: eval_metric}}, jf)

    for log_scale in (False, True):
        fig, ax = plt.pyplot.subplots(figsize=(8, 5))
        ax.plot(train_metric, label='train', linewidth=1.5)
        ax.plot(eval_metric,  label='eval',  linewidth=1.5)
        ax.set_xlabel('Boosting Round')
        ax.set_ylabel(metric)
        suffix = '_log' if log_scale else ''
        ax.set_title(f'XGBoost Learning Curve ({metric}{"  —  log scale" if log_scale else ""})')
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')

        out_path = os.path.join(out_dir, f'model_learning_curve_{metric}{suffix}.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.pyplot.close(fig)
        print(f'Learning curve saved at {out_path}')


def saveRunSummary(options, params, sig_files, bkg_files,
                   sigContainer, bkgContainer, eventContainer,
                   gbm, out_dir: str, early_stopping_rounds: int):
    """
    Writes a human-readable run_summary.txt to out_dir capturing all
    relevant metadata about the training run.
    """
    out_path = os.path.join(out_dir, 'run_summary.txt')
    best_iter = getattr(gbm, 'best_iteration', None)
    best_score = getattr(gbm, 'best_score', None)

    sig_total   = len(sigContainer.events)
    sig_train   = len(sigContainer.train_x)
    sig_test    = len(sigContainer.test_x)
    bkg_total   = len(bkgContainer.events)
    bkg_train   = len(bkgContainer.train_x)
    bkg_test    = len(bkgContainer.test_x)

    lines = [
        '=' * 60,
        'BDT TRAINING RUN SUMMARY',
        f'Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 60,
        '',
        '--- Output ---',
        f'Output directory : {os.path.abspath(out_dir)}',
        '',
        '--- Input Paths ---',
        f'Signal path      : {options.sig_path}',
        f'Signal files     : {len(sig_files)}',
        f'Background path  : {options.bkg_path}',
        f'Background files : {len(bkg_files)}',
        '',
        '--- Branch Names ---',
        f'Signal EcalVeto branch     : {sigContainer.ecalVeto_branch}',
        f'Signal trigger branch      : {sigContainer.trigger_branch}',
        f'Background EcalVeto branch : {bkgContainer.ecalVeto_branch}',
        '',
        '--- Event Counts ---',
        f'Max events per sample (max_evt) : {options.max_evt}',
        f'Train fraction (train_frac)     : {options.train_frac}',
        '',
        f'  Signal   total : {sig_total:>10,}',
        f'  Signal   train : {sig_train:>10,}',
        f'  Signal   test  : {sig_test:>10,}',
        f'  Bkg      total : {bkg_total:>10,}',
        f'  Bkg      train : {bkg_train:>10,}',
        f'  Bkg      test  : {bkg_test:>10,}',
        '',
        f'  Combined train : {len(eventContainer.train_x):>10,}',
        f'  Combined test  : {len(eventContainer.test_x):>10,}',
        '',
        '--- XGBoost Parameters ---',
    ] + [f'  {k:<25}: {v}' for k, v in params.items()] + [
        '',
        '--- Training Options ---',
        f'  num_boost_round      : {options.tree_number}',
        f'  early_stopping_rounds: {early_stopping_rounds}',
        f'  numpy random seed    : {options.seed}',
        '',
        '--- Training Result ---',
        f'  Best iteration : {best_iter}',
        f'  Best score     : {best_score}',
        '',
        '--- Features ({} total) ---'.format(len(eventContainer.feat_names)),
    ] + [f'  f{i:<4} {name}' for i, name in enumerate(eventContainer.feat_names)] + [
        '',
        '=' * 60,
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Run summary saved at {out_path}')


if __name__ == "__main__":
    
    # Parse
    parser = OptionParser()
    parser.add_option('--seed', dest='seed',type="int",  default=1, help='Numpy random seed.')
    parser.add_option('--max_evt', dest='max_evt',type="int",  default=1500000, help='Max Events to load')
    parser.add_option('--bkg_max_evt', dest='bkg_max_evt',type="int",  default=-1, help='Max background events (default -1 = same as --max_evt). Set higher than --max_evt for a bkg:sig imbalance, e.g. 2x for a 2:1 ratio.')
    parser.add_option('--train_frac', dest='train_frac', type='float', default=.9, help='Fraction of events to use for training')
    parser.add_option('--eta', dest='eta',type="float",  default=0.15, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number',type="int",  default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth',type="int",  default=10, help='Max Tree Depth')
    parser.add_option('--early_stopping_rounds', dest='early_stopping_rounds',type="int",  default=30, help='Early Stopping Rounds')
    parser.add_option('--min_child_weight', dest='min_child_weight',type="int",  default=20, help='Minimum Child Weight')
    parser.add_option('--subsample', dest='subsample', type="float", default=0.9, help='Subsample ratio of training instances')
    parser.add_option('--colsample_bytree', dest='colsample_bytree', type="float", default=0.85, help='Subsample ratio of columns per tree')
    parser.add_option('--gamma', dest='gamma', type="float", default=0.0, help='Min loss reduction to make a split (min_split_loss)')
    parser.add_option('--reg_lambda', dest='reg_lambda', type="float", default=1.0, help='L2 regularization on weights')
    parser.add_option('--reg_alpha', dest='reg_alpha', type="float", default=0.0, help='L1 regularization on weights')
    parser.add_option('--scale_pos_weight', dest='scale_pos_weight', type="float", default=1.0, help='Weight of positive (signal) class relative to negative (background); set to n_bkg/n_sig to rebalance')
    parser.add_option('--bkg_branch', dest='bkg_branch', default='EcalVeto_Recoil_Tracking', help='Background EcalVeto branch name')
    parser.add_option('--sig_branch', dest='sig_branch', default='EcalVeto_reco_v1', help='Signal EcalVeto branch name')
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

    # Mass-balance: interleave signal files by mass point (round-robin) so a --max_evt
    # cap draws ~equally from ALL mass points instead of just the first ones (files are
    # glob-grouped by mass, so a sequential cap would otherwise miss low masses entirely).
    import re as _re
    from collections import defaultdict as _dd
    _bymass = _dd(list)
    for _f in sorted(sig_files):
        _m = _re.search(r'_(\d+)MeV_', os.path.basename(_f))
        _bymass[_m.group(1) if _m else 'X'].append(_f)
    if len(_bymass) > 1:
        _lists = [_bymass[k] for k in sorted(_bymass, key=lambda s: int(s) if s.isdigit() else 10**9)]
        _inter = []; _i = 0
        while any(_i < len(l) for l in _lists):
            for l in _lists:
                if _i < len(l): _inter.append(l[_i])
            _i += 1
        sig_files = _inter
        print(f'Interleaved {len(sig_files)} signal files across mass points {sorted(_bymass)} for balanced max_evt sampling')


    # Seed numpy's randomness
    random_seed = np.random.randint(1,999999)
   
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
    print( 'Random seed is = {}'.format(random_seed)             )
    print( 'You set max_evt = {}'.format(options.max_evt)         )
    print( 'You set tree number = {}'.format(options.tree_number) )
    print( 'You set max tree depth = {}'.format(options.depth)    )
    print( 'You set eta = {}'.format(options.eta)                 )

    # Make Signal Container
    print( 'Loading sig_path = {}'.format(options.sig_path) )
    sigContainer = sampleContainer(sig_files,options.max_evt,options.train_frac,True,sig_branch=options.sig_branch)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    # Make Background Container
    print( 'Loading bkg_path = {}'.format(options.bkg_path) )
    _bkg_cap = options.bkg_max_evt if options.bkg_max_evt and options.bkg_max_evt > 0 else options.max_evt
    print( 'Background cap (bkg_max_evt) = {}  [sig cap = {}]'.format(_bkg_cap, options.max_evt) )
    bkgContainer = sampleContainer(bkg_files,_bkg_cap,options.train_frac,False,options.bkg_branch)
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
               'min_child_weight': options.min_child_weight,
               'subsample': options.subsample,
               'colsample_bytree': options.colsample_bytree,
               'gamma': options.gamma,
               'lambda': options.reg_lambda,
               'alpha': options.reg_alpha,
               'scale_pos_weight': options.scale_pos_weight,
               'eval_metric': 'error',
               'seed': random_seed,
               'nthread': 30,
               'verbosity': 1
    }

    # Train the BDT model
    evallist = [(eventContainer.dtrain,'train'), (eventContainer.dtest,'eval')]
    evals_result = {}  # xgb.train fills this dict in-place with per-round metrics
    gbm = xgb.train(params, eventContainer.dtrain, num_boost_round = options.tree_number, evals = evallist, early_stopping_rounds = options.early_stopping_rounds, evals_result=evals_result)
    
    # Store BDT
    with open(outpath+'_weights.pkl', 'wb') as output:
        pkl.dump(gbm, output)

    # Plot learning curve (always saved)
    eval_metric_key = params.get('eval_metric', 'error')
    print( 'Plotting learning curve...' )
    plotLearningCurve(evals_result, outpath, metric=eval_metric_key)

    # Save run summary
    saveRunSummary(options, params, sig_files, bkg_files,
                   sigContainer, bkgContainer, eventContainer,
                   gbm, outpath, early_stopping_rounds=options.early_stopping_rounds)

    # Calculate and plot shap values
    if options.shap_bool:
        print( 'Calculating SHAP values...' )
        shap_values = calcShap(eventContainer, gbm, outpath)
        print( 'Plotting SHAP summary...' )
        plotShap(eventContainer, shap_values, outpath)


    # Plot feature importances
    figure_path = outpath+'_fimportance.png'
    xgb.plot_importance(gbm)
    plt.pyplot.savefig(figure_path,
            dpi=500, bbox_inches='tight', pad_inches=0.5)
    

    print(f'Feature importances saved under {figure_path}')

    # Closing statement
    print("All files saved in: ", outpath)

if __name__ != "__main__":
    print(f'Imported {__file__}!')
