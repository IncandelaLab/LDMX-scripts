import uproot
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
#trigger skim

def process_directory(directory_path, tree_name='LDMX_Events'):
    root_files = glob.glob(os.path.join(directory_path, '*.root'))
    all_selected_data = []

    for file_path in root_files:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            #print(tree.keys())
            trigger_skimmed = tree['TriggerSums20Layers_signal/pass_'].array(library='np')
            plot_data = tree['EcalVeto_signal/maxCellDep_'].array(library='np')
            summed_det = tree['summedDet_'].array(library='np')
            #selected_data = plot_data[(trigger_skimmed == 1) & (summed_det<3000)]
            selected_data = plot_data[(trigger_skimmed == 1)]
            all_selected_data.extend(selected_data)

    return np.array(all_selected_data)

def process_directory_p(directory_path, tree_name='LDMX_Events'):
    root_files = glob.glob(os.path.join(directory_path, '*.root'))
    all_selected_data = []

    for file_path in root_files:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            #print(tree.keys())
            trigger_skimmed = tree['Trigger_sim/pass_'].array(library='np')
            plot_data = tree['EcalVeto_sim/maxCellDep_'].array(library='np')
            summed_det = tree['summedDet_'].array(library='np')
            #selected_data = plot_data[(trigger_skimmed == 1)& (summed_det<3000)]
            selected_data = plot_data[(trigger_skimmed == 1)]
            all_selected_data.extend(selected_data)

    return np.array(all_selected_data)

dir_pn = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/pn'
data_pn = process_directory_p(dir_pn)

dir_1GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/1GeV'
data_1GeV = process_directory(dir_1GeV)

dir_01GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/0.1GeV'
data_01GeV = process_directory(dir_01GeV)

dir_001GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/0.01GeV'
data_001GeV = process_directory(dir_001GeV)

dir_0001GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/0.001GeV'
data_0001GeV = process_directory(dir_0001GeV)


#not trigger skim
'''def process_directory(directory_path, tree_name='LDMX_Events'):
    root_files = glob.glob(os.path.join(directory_path, '*.root'))
    all_summed_det = []

    for file_path in root_files:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            summed_det = tree['summedDet_'].array(library='np')
            all_summed_det.extend(summed_det)

    return np.array(all_summed_det)

dir_pn = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/pn'
summed_det_pn = process_directory(dir_pn)

dir_1GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/1GeV'
summed_det_1GeV = process_directory(dir_1GeV)

dir_01GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/0.1GeV'
summed_det_01GeV = process_directory(dir_01GeV)

dir_001GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/0.01GeV'
summed_det_001GeV = process_directory(dir_001GeV)

dir_0001GeV = '/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/8GeV_sample/0.001GeV'
summed_det_0001GeV = process_directory(dir_0001GeV)
'''

num_bins = 200 
x_min = 0  
x_max = 10000  
bins = np.linspace(x_min, x_max, num_bins + 1) 

plt.hist(data_pn, bins=bins, color='red', alpha=0.7, histtype='step', label='pn', density=True)
plt.hist(data_1GeV, bins=bins, color='green', alpha=0.7, histtype='step', label='1GeV', density=True)
plt.hist(data_01GeV, bins=bins, color='purple', alpha=0.7, histtype='step', label='0.1GeV', density=True)
plt.hist(data_001GeV, bins=bins, color='orange', alpha=0.7, histtype='step', label='0.01GeV', density=True)
plt.hist(data_0001GeV, bins=bins, color='blue', alpha=0.7, histtype='step', label='0.001GeV', density=True)


plt.xlabel('maxCellDep')
plt.ylabel('Normalized Counts')
plt.yscale('log')
#plt.ylim(10**(-4),1)
plt.title('Normalized Histogram of summedDet')
plt.legend()
plt.savefig('/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/8_gev_preselection_plot/try_maxCellDep_normalized_t-skimmed_histogram.png')