import os
import fastjet
import ROOT as r
import numpy as np
from array import array

# fastjet.ClusterSequence.set_fastjet_banner_stream(None)


def azimuthal_angle(x, y, z):
    phi = np.arctan2(y, x)
    return phi

def pseudorapidity(x, y, z):
    mag = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / mag)
    eta = -np.log(np.tan(theta / 2))
    return eta

def rapidity_to_theta(rapidity):
    return 2*np.arctan(  np.exp(-rapidity))

def distance_between_point_to_point(point1, point2):
    eta_diff = point1[0] - point2[0]
    phi_diff = np.arctan2(np.sin(point1[1] - point2[1]), np.cos(point1[1] - point2[1]) )

    return np.sqrt(eta_diff**2 + phi_diff**2)

class CalcVetoVariable:
    def __init__(self, skim_root_file, output_root_file, mass_list, jettiness_R, jettiness_beta, angularity_R, angularity_beta):
        self.skim_root_file = skim_root_file
        self.output_root_file = output_root_file

        self.mass_list = mass_list
        self.jettiness_R = jettiness_R
        self.jettiness_beta = jettiness_beta

        self.angularity_R = angularity_R
        self.angularity_beta = angularity_beta
        
    def calc_jettiness_ratio_for_single_key_name(self, key_name, jettiness_R, beta):

        root_file = r.TFile(self.skim_root_file, 'read')

        if key_name != 'deepPhotonFromTarget':
            tree = root_file.Get(f'{key_name}GeV')
            num_events = tree.GetEntries()
        else:
            tree = root_file.Get(key_name)
            num_events = tree.GetEntries()

        jettiness_ratio_array = np.zeros(num_events)

        for i in range(num_events):
            tree.GetEntry(i)

            x_pos_array = np.array(tree.rec_xpos)
            y_pos_array = np.array(tree.rec_ypos)
            z_pos_array = np.array(tree.rec_zpos) - 250
            E_array = np.array(tree.rec_energy)

            if len(x_pos_array) == 0:
                print(x_pos_array, y_pos_array, z_pos_array)

            # Rotated 
            eta_array = pseudorapidity(y_pos_array, z_pos_array, x_pos_array)
            phi_array = azimuthal_angle(y_pos_array, z_pos_array, x_pos_array)

            # Coarse gain data 200, -2, 2, 200, -3.5, 3.5
            eta_phi_hist = r.TH2D('eta_phi_hist', 'eta_phi_hist', 200, -2, 2, 200, -3.5, 3.5)
            for j in range(len(eta_array)):
                eta_phi_hist.Fill(eta_array[j], phi_array[j], np.sin(rapidity_to_theta(eta_array[j])) * E_array[j])

            pseudojet_data = []
            for j in range(1, 201):
                for k in range(1, 201):
                    if eta_phi_hist.GetBinContent(j, k) == 0: continue
                    eta_bin = eta_phi_hist.GetXaxis().GetBinCenter(j)
                    phi_bin = eta_phi_hist.GetYaxis().GetBinCenter(k)
                    pT_bin = eta_phi_hist.GetBinContent(j, k)
                    pseudojet_data.append(fastjet.PtYPhiM(pT_bin, eta_bin, phi_bin, 0))
            eta_phi_hist.Delete()

            jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, jettiness_R)

            N1_cluster = fastjet.ClusterSequence(pseudojet_data, jetdef)
            N2_cluster = fastjet.ClusterSequence(pseudojet_data, jetdef)
            N1_jet_output = N1_cluster.exclusive_jets(1)
            N2_jet_output = N2_cluster.exclusive_jets(2)

            N1_jet_info = []
            N2_jet_info = []
            for jet in N1_jet_output:
                N1_jet_info.append([jet.pseudorapidity(), jet.phi_std()])
            for jet in N2_jet_output:
                N2_jet_info.append([jet.pseudorapidity(), jet.phi_std()])

            tau1 = 0
            tau2 = 0
            for j in range(len(eta_array)):
                eta_hit = eta_array[j]
                phi_hit = phi_array[j]
                pT_hit =  np.sin(rapidity_to_theta(eta_hit)) * E_array[j]
                hit_coordinate = np.array([eta_hit, phi_hit])

                N1_minimum_distance = distance_between_point_to_point(hit_coordinate, np.array(N1_jet_info)[0,:])
                N2_minimum_distance = np.min( [distance_between_point_to_point(hit_coordinate, np.array(N2_jet_info)[0,:]), distance_between_point_to_point(hit_coordinate, np.array(N2_jet_info)[1,:]) ]    )

                tau1 += pT_hit * N1_minimum_distance**beta
                tau2 += pT_hit * N2_minimum_distance**beta

            jettiness_ratio = tau2 / tau1
            jettiness_ratio_array[i] = jettiness_ratio

        return jettiness_ratio_array

    def calc_jettiness_ratio_for_all_key_name(self):
        jettiness_ratio_dict = {}
        print(f'Computing Jettiness Ratio with R = {self.jettiness_R:.2f} and beta = {self.jettiness_beta:.2f}')

        for key_name in self.mass_list + ['deepPhotonFromTarget']:
            jettiness_ratio_dict[key_name] = self.calc_jettiness_ratio_for_single_key_name(key_name, self.jettiness_R, self.jettiness_beta)

        return jettiness_ratio_dict

    def calc_angularity_for_single_key_name(self, key_name, angularity_R, beta):   

        root_file = r.TFile(self.skim_root_file, 'read')

        if key_name != 'deepPhotonFromTarget':
            tree = root_file.Get(f'{key_name}GeV')
            num_events = tree.GetEntries()
        else:
            tree = root_file.Get(key_name)
            num_events = tree.GetEntries()
    
        angularity_array = np.zeros(num_events)

        for i in range(num_events):
            tree.GetEntry(i)

            x_pos_array = np.array(tree.rec_xpos)
            y_pos_array = np.array(tree.rec_ypos)
            z_pos_array = np.array(tree.rec_zpos) - 250
            E_array = np.array(tree.rec_energy)

            eta_array = pseudorapidity(y_pos_array, z_pos_array, x_pos_array)
            phi_array = azimuthal_angle(y_pos_array,z_pos_array,x_pos_array)

            # Coarse gain data  50, -2, 8, 50, -3.5, 3.5
            eta_phi_hist = r.TH2D('eta_phi_hist', 'eta_phi_hist', 200, -2, 2, 200, -3.5, 3.5)
            for j in range(len(eta_array)):
                eta_phi_hist.Fill(eta_array[j], phi_array[j], np.sin(rapidity_to_theta(eta_array[j])) * E_array[j])

            pseudojet_data = []

            for j in range(1, 201):
                for k in range(1, 201):
                    if eta_phi_hist.GetBinContent(j, k) == 0: continue
                    eta_bin = eta_phi_hist.GetXaxis().GetBinCenter(j)
                    phi_bin = eta_phi_hist.GetYaxis().GetBinCenter(k)
                    pT_bin = eta_phi_hist.GetBinContent(j, k)

                    pseudojet_data.append(fastjet.PtYPhiM(pT_bin, eta_bin, phi_bin, 0))

            eta_phi_hist.Delete()

            jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, angularity_R)

            jet_cluster = fastjet.ClusterSequence(pseudojet_data, jetdef)
            jet_output = jet_cluster.inclusive_jets()

            jet_info = []
            for jet in jet_output:
                jet_info.append([jet.pt(), jet.pseudorapidity(), jet.phi_std()])

            max_indices = np.argsort(np.array(jet_info)[:, 0])[-1:]
            jet_coords = np.array(jet_info)[max_indices, 1:]
            
            total_pT = 0
            angularity = 0
            for j in range(len(eta_array)):
                eta_hit = eta_array[j]
                phi_hit = phi_array[j]
                pT_hit =  np.sin(rapidity_to_theta(eta_hit)) * E_array[j]
                hit_coordinate = np.array([eta_hit, phi_hit])
                delta_R =  distance_between_point_to_point(hit_coordinate, jet_coords[0,:]) 
                if delta_R > angularity_R: continue
                
                total_pT += pT_hit
                angularity += pT_hit * delta_R**beta

            angularity_array[i] = angularity / total_pT

        return angularity_array

    def calc_angularity_for_all_key_name(self):
        angularity_dict = {}
        print(f'Computing Angularity with R = {self.angularity_R:.2f} and beta = {self.angularity_beta:.2f}')

        for key_name in self.mass_list + ['deepPhotonFromTarget']:
            angularity_dict[key_name] = self.calc_angularity_for_single_key_name(key_name, self.angularity_R, self.angularity_beta)

        return angularity_dict

    def retrieve_EcalVeto_variable_for_single_key_name(self, key_name, ecalVeto_var_name):

        root_file = r.TFile(self.skim_root_file, 'read')

        if key_name != 'deepPhotonFromTarget':
            tree = root_file.Get(f'{key_name}GeV')
            num_events = tree.GetEntries()
        else:
            tree = root_file.Get(key_name)
            num_events = tree.GetEntries()

        ecalVeto_var_array = np.zeros(num_events)

        for i in range(num_events):
            tree.GetEntry(i)
            ecalVeto_var_array[i] = getattr(tree, ecalVeto_var_name)
        
        return ecalVeto_var_array

    def retrieve_EcalVeto_variable_for_all_key_name(self, ecalVeto_var_name):

        ecalVeto_var_dict = {}
        print(f'Retrieving branch {ecalVeto_var_name}.')

        for key_name in self.mass_list + ['deepPhotonFromTarget']:
            ecalVeto_var_dict[key_name] = self.retrieve_EcalVeto_variable_for_single_key_name(key_name, ecalVeto_var_name)

        return ecalVeto_var_dict
        
    def calc_all_variables(self):

        self.n_readout_hits_dict = self.retrieve_EcalVeto_variable_for_all_key_name('nReadoutHits')
        self.avg_layer_hit_dict = self.retrieve_EcalVeto_variable_for_all_key_name('avgLayerHit')
        self.std_layer_hit_dict = self.retrieve_EcalVeto_variable_for_all_key_name('stdLayerHit')
        self.jettiness_ratio_dict = self.calc_jettiness_ratio_for_all_key_name()
        self.angularity_dict  = self.calc_angularity_for_all_key_name()

        out_file = r.TFile(self.output_root_file, 'recreate')

        nReadoutHits_array     = array('i', [0])
        avgLayerHit_array      = array('d', [0])
        stdLayerHit_array      = array('d', [0])
        jettiness_ratio_array  = array('d', [0])
        angularity_array       = array('d', [0])

        for key_name in self.mass_list + ['deepPhotonFromTarget']:
            num_events = len(self.n_readout_hits_dict[key_name])

            if key_name != 'deep_gammaFromTarget':
                root_tree =  r.TTree(f'{key_name}GeV', f'{key_name}GeV')
            else:
                root_tree = r.TTree('deepPhotonFromTarget', 'deepPhotonFromTarget')

            
            root_tree.Branch('nReadoutHits',        nReadoutHits_array,     'nReadoutHits/I')
            root_tree.Branch('avgLayerHit',         avgLayerHit_array,      'avgLayerHit/D')
            root_tree.Branch('stdLayerHit',         stdLayerHit_array,      'stdLayerHit/D')
            root_tree.Branch('jettinessRatio',      jettiness_ratio_array,  'jettinessRatio/D')
            root_tree.Branch('angularity',          angularity_array,       'angularity/D')

            for i in range(num_events):
                nReadoutHits_array[0]       = int(self.n_readout_hits_dict[key_name][i])
                avgLayerHit_array[0]        = self.avg_layer_hit_dict[key_name][i]
                stdLayerHit_array[0]        = self.std_layer_hit_dict[key_name][i]
                jettiness_ratio_array[0]    = self.jettiness_ratio_dict[key_name][i]
                angularity_array[0]         = self.angularity_dict[key_name][i]

                root_tree.Fill()  

            root_tree.Write()

        out_file.Close()


if __name__ == '__main__':

    cwd = os.getcwd()

    analysis = CalcVetoVariable(cwd + '/skimmed.root', cwd + '/vetoVar.root', [0.005, 0.025, 0.05, 0.5], 0.4, 2.5, 0.15, 5)
    analysis.calc_all_variables()






