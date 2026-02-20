import os
import csv
import logging
import argparse
import ROOT as r
import numpy as np
from array import array

print('Loading gSystem')
r.gSystem.Load('libFramework.so')

def bkg_decay_variables(sim_particles_map, TSP_hits):
    for sim_particle in sim_particles_map:
        track_id = sim_particle.first
        particle = sim_particle.second
        pdgID = particle.getPdgID()
        energy = particle.getEnergy()
        parent_track_ids = particle.getParents()

        energy_thresh = 5000
        for parent_track_id in parent_track_ids:
            # Search for hard brem emitted from beam electron (track id = 1)
            if (parent_track_id == 1) and (energy >= energy_thresh) and (pdgID == 22):
                daughter_track_ids = particle.getDaughters()
                decay_vertex = np.array([particle.getEndPoint()[0], particle.getEndPoint()[1], particle.getEndPoint()[2]] )
                brem_momentum = np.array([particle.getMomentum()[0], particle.getMomentum()[1], particle.getMomentum()[2]])
    
    # Some events have hard brem with daughters where the daughter particle info is not stored 
    electron_momentum = np.array([0, 0, 0])
    positron_momentum = np.array([0, 0, 0])
    for daughter_track_id in daughter_track_ids:
        daughter = sim_particles_map[daughter_track_id]
        if not np.abs(daughter.getVertex()[2] - decay_vertex[2]) < 0.1: continue
        if daughter.getPdgID() == 11:
            electron_momentum = np.array([daughter.getMomentum()[0], daughter.getMomentum()[1], daughter.getMomentum()[2]])
        elif daughter.getPdgID() == -11:
            positron_momentum = np.array([daughter.getMomentum()[0], daughter.getMomentum()[1], daughter.getMomentum()[2]])
    zero_momentum = np.array([0., 0., 0.])
    
    if (electron_momentum == zero_momentum).all() or (positron_momentum == zero_momentum).all() :
        opening_angle = -1
    else:
        opening_angle =  np.rad2deg( np.arccos( np.dot(electron_momentum, positron_momentum) / (np.linalg.norm(electron_momentum) * np.linalg.norm(positron_momentum)) ) )

    for hit in TSP_hits:
        if hit.getTrackID() != 1: continue
        if hit.getPosition()[2] > 0:
            recoil_momentum = np.array([hit.getMomentum()[0], hit.getMomentum()[1], hit.getMomentum()[2]])
            vertex_TSP = np.array([hit.getPosition()[0], hit.getPosition()[1], hit.getPosition()[2]])

    recoil_momentum_unit_vec = recoil_momentum / np.linalg.norm(recoil_momentum)
    scale = - vertex_TSP[2] / recoil_momentum_unit_vec[2]
    vertex_at_target = vertex_TSP + scale*recoil_momentum_unit_vec

    return decay_vertex, opening_angle, recoil_momentum, vertex_at_target, electron_momentum, positron_momentum, brem_momentum


def signal_decay_variables(sim_particles_map, TSP_hits):
    for sim_particle in sim_particles_map:
        track_id = sim_particle.first
        particle = sim_particle.second

        # Find ALP
        if particle.getGenStatus() == 2 and particle.getPdgID() == 622: 
            alp_momentum = np.array([particle.getMomentum()[0], particle.getMomentum()[1], particle.getMomentum()[2]])

        # Filter that only leaves electron at target and decay e+ e- 
        if particle.getGenStatus() == 1:
            if particle.getVertex()[2] == 0:
                recoil_momentum = np.array([particle.getMomentum()[0], particle.getMomentum()[1], particle.getMomentum()[2]])
            else:
                if particle.getPdgID() == 11:
                    decay_vertex = np.array([particle.getVertex()[0], particle.getVertex()[1], particle.getVertex()[2]])
                    electron_momentum = np.array([particle.getMomentum()[0], particle.getMomentum()[1], particle.getMomentum()[2]])
                elif particle.getPdgID() == -11:
                    positron_momentum = np.array([particle.getMomentum()[0], particle.getMomentum()[1], particle.getMomentum()[2]])
    
    opening_angle =  np.rad2deg( np.arccos( np.dot(electron_momentum, positron_momentum) / (np.linalg.norm(electron_momentum) * np.linalg.norm(positron_momentum)) ) )

    max_energy = 0
    for hit in TSP_hits:
        if hit.getPdgID()!= 11: continue
        if hit.getPosition()[2] < 0: continue
        if hit.getEnergy() > max_energy:
            max_energy = hit.getEnergy()
            recoil_momentum = np.array([hit.getMomentum()[0], hit.getMomentum()[1], hit.getMomentum()[2]])
            vertex_TSP = np.array([hit.getPosition()[0], hit.getPosition()[1], hit.getPosition()[2]])
    
    # Events where the electrons are so soft they leave no track, veto
    if max_energy == 0:
        vertex_TSP = np.array([0, 0, 0])
        opening_angle = -1

    recoil_momentum_unit_vec = recoil_momentum / np.linalg.norm(recoil_momentum)
    scale = - vertex_TSP[2] / recoil_momentum_unit_vec[2]
    vertex_at_target = vertex_TSP + scale*recoil_momentum_unit_vec
    
    return decay_vertex, opening_angle, recoil_momentum, vertex_at_target, electron_momentum, positron_momentum, alp_momentum


class EcalVetoDataSet:
    def __init__(self):
        self.signal_file_dict = {}

    def add_signal_file(self, signal_file_list, mass):
        signal_chain = r.TChain('LDMX_Events')
        for file in signal_file_list: signal_chain.Add(file)
        self.signal_file_dict[mass] = signal_chain

    def add_background_file(self, bkg_file_list):
        print('Adding background file')
        self.bkg_chain = r.TChain('LDMX_Events')
        for file in bkg_file_list: self.bkg_chain.Add(file)

    def load_data(self, root_file_name, num_signal_events, test_run = False):

        self.root_file_name = root_file_name
        out_file = r.TFile(self.root_file_name, 'recreate')

        # Rec Variables
        max_num_hits = 1000
        nReadoutHits_array = array('i', [0])
        decay_x_array           = array('d', [0])
        decay_y_array           = array('d', [0])
        decay_z_array           = array('d', [0])
        opening_angle_array     = array('d', [0])
        avgLayerHit_array       = array('d', [0])
        stdLayerHit_array       = array('d', [0])
        recoil_px_array         = array('d', [0])
        recoil_py_array         = array('d', [0])
        recoil_pz_array         = array('d', [0])
        target_vertex_x_array   = array('d', [0])
        target_vertex_y_array   = array('d', [0])
        target_vertex_z_array   = array('d', [0])
        electron_px_array       = array('d', [0])
        electron_py_array       = array('d', [0])
        electron_pz_array       = array('d', [0])
        positron_px_array       = array('d', [0])
        positron_py_array       = array('d', [0])
        positron_pz_array       = array('d', [0])

        brem_px_array           = array('d', [0])
        brem_py_array           = array('d', [0])
        brem_pz_array           = array('d', [0])
        alp_px_array            = array('d', [0])
        alp_py_array            = array('d', [0])
        alp_pz_array            = array('d', [0])

        rec_energy_array   = array('d', max_num_hits * [0])
        rec_xpos_array     = array('d', max_num_hits * [0])
        rec_ypos_array     = array('d', max_num_hits * [0])
        rec_zpos_array     = array('d', max_num_hits * [0])
        
        # Load background data 
        if test_run:
            n_events = np.floor( self.bkg_chain.GetEntries() * 0.01 )
        else:  
            n_events = self.bkg_chain.GetEntries()
        
        logging.info(f'Loading background sample (found {n_events} events)')
        
        # Rec Variables
        root_tree = r.TTree('deepPhotonFromTarget', 'deepPhotonFromTarget')

        root_tree.Branch('nReadoutHits',        nReadoutHits_array,     'nReadoutHits/I')
        root_tree.Branch('decay_x',             decay_x_array,          'decay_x/D')
        root_tree.Branch('decay_y',             decay_y_array,          'decay_y/D')
        root_tree.Branch('decay_z',             decay_z_array,          'decay_z/D')
        root_tree.Branch('opening_angle',       opening_angle_array,    'opening_angle/D')
        root_tree.Branch('avgLayerHit',         avgLayerHit_array,      'avgLayerHit/D')
        root_tree.Branch('stdLayerHit',         stdLayerHit_array,      'stdLayerHit/D')
        root_tree.Branch('recoil_px',           recoil_px_array,        'recoil_px/D')
        root_tree.Branch('recoil_py',           recoil_py_array,        'recoil_py/D')
        root_tree.Branch('recoil_pz',           recoil_pz_array,        'recoil_pz/D')
        root_tree.Branch('target_vertex_x',     target_vertex_x_array,  'target_vertex_x/D')
        root_tree.Branch('target_vertex_y',     target_vertex_y_array,  'target_vertex_y/D')
        root_tree.Branch('target_vertex_z',     target_vertex_z_array,  'target_vertex_z/D')
        root_tree.Branch('electron_px',         electron_px_array,      'electron_px/D')
        root_tree.Branch('electron_py',         electron_py_array,      'electron_py/D')
        root_tree.Branch('electron_pz',         electron_pz_array,      'electron_pz/D')
        root_tree.Branch('positron_px',         positron_px_array,      'positron_px/D')
        root_tree.Branch('positron_py',         positron_py_array,      'positron_py/D')
        root_tree.Branch('positron_pz',         positron_pz_array,      'positron_pz/D')
        root_tree.Branch('brem_px',             brem_px_array,          'brem_px/D')
        root_tree.Branch('brem_py',             brem_py_array,          'brem_py/D')
        root_tree.Branch('brem_pz',             brem_pz_array,          'brem_pz/D')

        root_tree.Branch('rec_energy',   rec_energy_array,   'rec_energy[nReadoutHits]/D')
        root_tree.Branch('rec_xpos',     rec_xpos_array,     'rec_xpos[nReadoutHits]/D')
        root_tree.Branch('rec_ypos',     rec_ypos_array,     'rec_ypos[nReadoutHits]/D')
        root_tree.Branch('rec_zpos',     rec_zpos_array,     'rec_zpos[nReadoutHits]/D')

        for event_num in range(n_events):

            self.bkg_chain.GetEntry(event_num)
            decay_vertex, opening_angle, recoil_momentum, vertex_at_target, electron_momentum, positron_momentum, brem_momentum = bkg_decay_variables(self.bkg_chain.SimParticles_v14_deepPhotonFromTarget, self.bkg_chain.TargetScoringPlaneHits_v14_deepPhotonFromTarget)
            
            #Only keep background events where decay_z > 350
            #if decay_vertex[2] <= 350: continue
            
            # Demand that the recoil electron has a postive z-momenutm
            if (recoil_momentum[2] <= 0) or (opening_angle == -1): continue

            decay_x_array[0]            = decay_vertex[0]
            decay_y_array[0]            = decay_vertex[1]
            decay_z_array[0]            = decay_vertex[2]
            opening_angle_array[0]      = opening_angle
            recoil_px_array[0]          = recoil_momentum[0]
            recoil_py_array[0]          = recoil_momentum[1]
            recoil_pz_array[0]          = recoil_momentum[2]
            target_vertex_x_array[0]    = vertex_at_target[0]
            target_vertex_y_array[0]    = vertex_at_target[1]
            target_vertex_z_array[0]    = vertex_at_target[2]
            electron_px_array[0]        = electron_momentum[0]
            electron_py_array[0]        = electron_momentum[1]
            electron_pz_array[0]        = electron_momentum[2]
            positron_px_array[0]        = positron_momentum[0]
            positron_py_array[0]        = positron_momentum[1]
            positron_pz_array[0]        = positron_momentum[2]
            brem_px_array[0]            = brem_momentum[0]
            brem_py_array[0]            = brem_momentum[1]
            brem_pz_array[0]            = brem_momentum[2]

            nReadoutHits_array[0]   = len(self.bkg_chain.EcalRecHits_v14_deepPhotonFromTarget)
            avgLayerHit_array[0]    = self.bkg_chain.EcalVeto_v14_deepPhotonFromTarget.getAvgLayerHit()
            stdLayerHit_array[0]    = self.bkg_chain.EcalVeto_v14_deepPhotonFromTarget.getStdLayerHit()

            hit_count = 0
            for hit in self.bkg_chain.EcalRecHits_v14_deepPhotonFromTarget:
                
                # ECal Rec Hits
                rec_energy_array[hit_count] = hit.getEnergy()
                rec_xpos_array[hit_count]   = hit.getXPos()
                rec_ypos_array[hit_count]   = hit.getYPos()
                rec_zpos_array[hit_count]   = hit.getZPos()

                hit_count += 1

            root_tree.Fill()

        root_tree.Write()

        # Load signal data 
        for mass, signal_chain in self.signal_file_dict.items():

            if test_run:
                n_events = int(np.floor( signal_chain.GetEntries() * 0.01 ))
            else:  
                n_events =  signal_chain.GetEntries()

            print(f'Loading m = {mass} GeV: Total sample (found {n_events} events)')

            root_tree = r.TTree(f'{mass}GeV', f'{mass}GeV')
          
            # ECal Veto Data
            root_tree.Branch('nReadoutHits',        nReadoutHits_array,     'nReadoutHits/I')
            root_tree.Branch('decay_x',             decay_x_array,          'decay_x/D')
            root_tree.Branch('decay_y',             decay_y_array,          'decay_y/D')
            root_tree.Branch('decay_z',             decay_z_array,          'decay_z/D')
            root_tree.Branch('opening_angle',       opening_angle_array,    'opening_angle/D')
            root_tree.Branch('avgLayerHit',         avgLayerHit_array,      'avgLayerHit/D')
            root_tree.Branch('stdLayerHit',         stdLayerHit_array,      'stdLayerHit/D')
            root_tree.Branch('recoil_px',           recoil_px_array,        'recoil_px/D')
            root_tree.Branch('recoil_py',           recoil_py_array,        'recoil_py/D')
            root_tree.Branch('recoil_pz',           recoil_pz_array,        'recoil_pz/D')
            root_tree.Branch('target_vertex_x',     target_vertex_x_array,  'target_vertex_x/D')
            root_tree.Branch('target_vertex_y',     target_vertex_y_array,  'target_vertex_y/D')
            root_tree.Branch('target_vertex_z',     target_vertex_z_array,  'target_vertex_z/D')
            root_tree.Branch('electron_px',         electron_px_array,      'electron_px/D')
            root_tree.Branch('electron_py',         electron_py_array,      'electron_py/D')
            root_tree.Branch('electron_pz',         electron_pz_array,      'electron_pz/D')
            root_tree.Branch('positron_px',         positron_px_array,      'positron_px/D')
            root_tree.Branch('positron_py',         positron_py_array,      'positron_py/D')
            root_tree.Branch('positron_pz',         positron_pz_array,      'positron_pz/D')
            root_tree.Branch('alp_px',              alp_px_array,           'alp_px/D')
            root_tree.Branch('alp_py',              alp_py_array,           'alp_py/D')
            root_tree.Branch('alp_pz',              alp_pz_array,           'alp_pz/D')
            
            root_tree.Branch('rec_energy',   rec_energy_array,   'rec_energy[nReadoutHits]/D')
            root_tree.Branch('rec_xpos',     rec_xpos_array,     'rec_xpos[nReadoutHits]/D')
            root_tree.Branch('rec_ypos',     rec_ypos_array,     'rec_ypos[nReadoutHits]/D')
            root_tree.Branch('rec_zpos',     rec_zpos_array,     'rec_zpos[nReadoutHits]/D')


            for event_num in range(n_events):
                signal_chain.GetEntry(event_num)

                decay_vertex, opening_angle, recoil_momentum, vertex_at_target, electron_momentum, positron_momentum, alp_momentum = signal_decay_variables(signal_chain.SimParticles_alp, signal_chain.TargetScoringPlaneHits_alp)

                # Check if decay_z is in valid ranges
                # z_pos = decay_vertex[2]
                # Defined ranges for signal keeping
                # valid_z_ranges = [
                #     (350, 354),
                #     (360, 363),
                #     (369, 371),
                #     (379, 384),
                #     (391, 394),
                #     (399, 401),
                #     (409, 410)
                # ]
                
                # is_valid_z = False
                # for z_min, z_max in valid_z_ranges:
                #     if z_min <= z_pos <= z_max:
                #         is_valid_z = True
                #         break
                
                # if not is_valid_z:
                #     continue

                if (recoil_momentum[2] <= 0) or  (opening_angle == -1): 
                    continue

                nReadoutHits_array[0]       = len(signal_chain.EcalRecHits_alp)
                avgLayerHit_array[0]        = signal_chain.EcalVeto_alp.getAvgLayerHit()
                stdLayerHit_array[0]        = signal_chain.EcalVeto_alp.getStdLayerHit()
                decay_x_array[0]            = decay_vertex[0]
                decay_y_array[0]            = decay_vertex[1]
                decay_z_array[0]            = decay_vertex[2]
                opening_angle_array[0]      = opening_angle
                recoil_px_array[0]          = recoil_momentum[0]
                recoil_py_array[0]          = recoil_momentum[1]
                recoil_pz_array[0]          = recoil_momentum[2]
                target_vertex_x_array[0]    = vertex_at_target[0]
                target_vertex_y_array[0]    = vertex_at_target[1]
                target_vertex_z_array[0]    = vertex_at_target[2]
                electron_px_array[0]        = electron_momentum[0]
                electron_py_array[0]        = electron_momentum[1]
                electron_pz_array[0]        = electron_momentum[2]
                positron_px_array[0]        = positron_momentum[0]
                positron_py_array[0]        = positron_momentum[1]
                positron_pz_array[0]        = positron_momentum[2]
                alp_px_array[0]             = alp_momentum[0]
                alp_py_array[0]             = alp_momentum[1]
                alp_pz_array[0]             = alp_momentum[2]

                # --- VETO CUT ---
                if nReadoutHits_array[0] < 10:
                    continue

                hit_count = 0
                for hit in signal_chain.EcalRecHits_alp:
                    # ECal Rec Hits
                    rec_energy_array[hit_count] = hit.getEnergy()
                    rec_xpos_array[hit_count]   = hit.getXPos()
                    rec_ypos_array[hit_count]   = hit.getYPos()
                    rec_zpos_array[hit_count]   = hit.getZPos()

                    hit_count += 1

                root_tree.Fill()   
            root_tree.Write()

        out_file.Close()

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Dark Photon Generator (Gamma Skimmer)')
    # Replace -min/-max with -dl for the new gamma data structure
    parser.add_argument('-dl', '--decay_length', type = int, required=True, help = 'Lab frame decay length for Gamma distribution (signal).')
    parser.add_argument('--bkg_decay_z', type = int, required=True, help = 'Lab frame decay z position (bkg).')

    args = parser.parse_args()

    # Configuring the logger
    logging.basicConfig(format = '[ Production ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    cwd = os.getcwd()
    ecal_veto_data = EcalVetoDataSet()
    test_run_bool = False
    
    # Add signal data - [MODIFIED FOR GAMMA FOLDER PATH]
    alp_mass_list = [0.005, 0.025, 0.05, 0.5, 1.0]
    for mass in alp_mass_list:
        file_list = []
        
        signal_mass_dir = cwd + f'/ALP_LDMX_Directory/gamma_decay{str(args.decay_length)}/{str(mass)}GeV'
        
        if os.path.exists(signal_mass_dir):
            for file_name in os.listdir(signal_mass_dir):
                if file_name.endswith('.root'): # Only add .root files
                    file_list.append(os.path.join(signal_mass_dir, file_name))
            ecal_veto_data.add_signal_file(file_list, mass)
        else:
            logging.warning(f'Signal directory not found: {signal_mass_dir}')
    
    # Add bkg data
    bkg_file_list = []
    bkg_dir = cwd + f'/deepPhotonLDMX/decay{str(args.bkg_decay_z)}'
    if os.path.exists(bkg_dir):
        for file_name in os.listdir(bkg_dir):
            if file_name.endswith('.root'):
                bkg_file_list.append(os.path.join(bkg_dir, file_name))
        ecal_veto_data.add_background_file(bkg_file_list)
    else:
         logging.warning(f'Background directory not found: {bkg_dir}')

    # Save to a specific name so it doesn't overwrite your old skimmed files
    output_skimmed_file = cwd + f'/skimmed_gamma_DL{args.decay_length}.root'
    ecal_veto_data.load_data(output_skimmed_file, test_run_bool)
    logging.info(f'Skimming complete! Output saved to {output_skimmed_file}')
