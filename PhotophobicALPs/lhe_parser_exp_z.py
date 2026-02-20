import os
import time
import pylhe
import logging
import argparse
import numpy as np
from scipy.stats import uniform

def format_event_info(event_info_dict, n_particles):
    keys = ["pid", "weight", "scale", "aqed", "aqcd"]
    value_list = [event_info_dict[key] for key in keys]
    format_string = ' {:<4.0f} {:>3.0f} {:<+14.8e} {:<+14.8e} {:<+14.8e} {:<+14.8e}\n'
    return format_string.format(n_particles, *value_list)

def format_single_particle_info(single_particle_info_list):
    format_string = ' {:<4} {:>3.0f} {:>2.0f}    {:.0f}    {:.0f}    {:.0f}    {:.0f} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e}\n'
    return format_string.format(' ', *single_particle_info_list)

def format_ALP_particle_info(particle_info_list):
    output_str = ''
    key_1 = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2']
    key_2 = ['px', 'py', 'pz', 'e']
    key_3 = ['m', 'lifetime', 'spin']
    
    three_momentum = np.array([0.0, 0.0, 0.0])
    lifetime = 0.0

    for particle_dict in particle_info_list:
        value_list = [particle_dict[key] for key in key_1] + [particle_dict['vector'][key] for key in key_2] + [particle_dict[key] for key in key_3]
        output_str += format_single_particle_info(value_list)

        if particle_dict['id'] == 622:
            three_momentum_dict = particle_dict['vector']
            three_momentum = np.array([three_momentum_dict['px'], three_momentum_dict['py'], three_momentum_dict['pz']])
            lifetime = particle_dict['lifetime']

    return output_str, three_momentum, lifetime

def format_decay_particle_info(particle_info_list):
    output_str = ''
    key_1 = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2']
    key_2 = ['px', 'py', 'pz', 'e']
    key_3 = ['m', 'lifetime', 'spin']

    for particle_dict in particle_info_list:
        value_list = [particle_dict[key] for key in key_1] + [particle_dict['vector'][key] for key in key_2] + [particle_dict[key] for key in key_3]
        value_list[2] = -1
        value_list[3] = -1
        output_str += format_single_particle_info(value_list)
    return output_str

def format_vertex(x_, y_, z_, t_):
    format_string = '#vertex {} {} {} {}\n'
    return format_string.format(x_, y_, z_, t_)


def find_decay_vertex(three_momentum, beam_vertex, mass, decay_min_z, decay_max_z, ctau_lifetime):
    c = 299_792_458 * 1000 # [c] = mm / s
    
    # 1. Kinematics
    p_mag = np.linalg.norm(three_momentum)
    p_z = three_momentum[2]
    
    # Forcefully override decay length: Hardcode decay length, regardless of input.
    lab_decay_length = 10.0 
    
    # Debug print to confirm code execution, uses end='\r' to prevent log spamming.
    print(f"DEBUG: Force Lambda = {lab_decay_length} mm | Window = {decay_min_z}-{decay_max_z}", end='\r')

    # 2. Geometry Scale Factor (Z -> Path Length L)
    if p_z <= 0:
        scale_factor_z_to_L = 1e9 
    else:
        scale_factor_z_to_L = p_mag / p_z

    # 3. Define Sampling Window
    z_window_width = decay_max_z - decay_min_z
    max_flight_dist_in_box = z_window_width * scale_factor_z_to_L
    
    # 4. Exponential Sampling Logic, P(d < max) = 1 - exp(-max / lambda)
    prob_max_in_window = 1.0 - np.exp(-max_flight_dist_in_box / lab_decay_length)
    
    # Uniform random number scaled to the CDF range
    u = uniform.rvs(0, 1)
    target_prob = u * prob_max_in_window
    
    # Inverse CDF
    if target_prob >= 1.0:
        sampled_local_dist = max_flight_dist_in_box
    else:
        # Math: dist = -lambda * ln(1 - target_prob)
        sampled_local_dist = -lab_decay_length * np.log(1.0 - target_prob)

    # 5. Reconstruct Global Position
    dist_to_entry = (decay_min_z - beam_vertex[2]) * scale_factor_z_to_L
    total_distance = dist_to_entry + sampled_local_dist
    
    unit_vec = three_momentum / p_mag
    vertex_pos = beam_vertex + unit_vec * total_distance

    # 6. Time
    energy = np.sqrt(p_mag**2 + mass**2)
    gamma = energy / mass
    velo_vec = three_momentum * c / (gamma * mass)
    speed = np.linalg.norm(velo_vec)
    
    if speed > 0:
        lab_frame_time = total_distance * 10**9 / speed # ns
    else:
        lab_frame_time = 0

    return format_vertex(*vertex_pos, lab_frame_time)

def format_event_block(event_info_line, particle_info_block, vertex_line):
    return '<event>\n' + event_info_line + particle_info_block + vertex_line + '</event>\n'

def main():
    parser = argparse.ArgumentParser(description = 'ALP LHE Parser (Hardcoded Exponential).')
    parser.add_argument('-i',   '--input_lhe',                      type = str,     help = 'Input LHE file.')
    parser.add_argument('-o',   '--output_dir',                     type = str,     help = 'Output directory.')
    parser.add_argument('-m',   '--mass',                           type = float,   help = 'Simulated ALP mass.')
    parser.add_argument('-min', '--decay_min_z',                    type = int,     help = 'Min Z.')
    parser.add_argument('-max', '--decay_max_z',                    type = int,     help = 'Max Z.')
    parser.add_argument('-s',   '--seed',                           type = int,     help = 'Random seed.')
    parser.add_argument('-n_A', '--n_ALP',          default = 5,    type = int,     help = 'N ALP.')
    parser.add_argument('-m_D', '--n_decay',        default = 2,    type = int,     help = 'N Decay.')

    args = parser.parse_args()

    logging.basicConfig(format = '[ Parser ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    input_file = args.input_lhe
    ALP_file = args.output_dir + f'/alp_to_e+e-_{str(args.mass)}GeV_seed{str(args.seed)}_min{str(args.decay_min_z)}_max{str(args.decay_max_z)}_prod.lhe'
    decay_file = args.output_dir + f'/alp_to_e+e-_{str(args.mass)}GeV_seed{str(args.seed)}_min{str(args.decay_min_z)}_max{str(args.decay_max_z)}_decay.lhe'

    n_events = pylhe.read_num_events(input_file)
    logging.info(f'Found {n_events} events.')
    
    logging.warning('!!! HARDCODED MODE: DECAY LENGTH FIXED TO 20.0 mm !!!')

    with open(input_file, 'r') as input_lhe, open(ALP_file, 'w') as ALP_lhe, open(decay_file, 'w') as decay_lhe:
        for line in input_lhe:
            if line == '</init>\n':
                ALP_lhe.write(line)
                decay_lhe.write(line)
                break
            ALP_lhe.write(line)
            decay_lhe.write(line)
        
        events = pylhe.to_awkward(pylhe.read_lhe(input_file))
        for event in events:
            ALP_event_info_line = format_event_info(event['eventinfo'], args.n_ALP)
            decay_event_info_line = format_event_info(event['eventinfo'], args.n_decay)

            ALP_particles = event['particles'][0:args.n_ALP]
            decay_particles = event['particles'][args.n_ALP:(args.n_decay + args.n_ALP)]

            ALP_particles_block, ALP_three_momentum, ALP_lifetime = format_ALP_particle_info(ALP_particles)

            beam_x = uniform.rvs(-10, 20)
            beam_y = uniform.rvs(-40, 80)
            beam_z = 0
            beam_t = 0
            ALP_vertex = format_vertex(beam_x, beam_y, beam_z, beam_t)
            ALP_lhe.write(format_event_block(ALP_event_info_line, ALP_particles_block, ALP_vertex))

            decay_particles_block = format_decay_particle_info(decay_particles)
            decay_vertex = find_decay_vertex(ALP_three_momentum, np.array([beam_x, beam_y, beam_z]), args.mass, args.decay_min_z, args.decay_max_z, ALP_lifetime)
            decay_lhe.write(format_event_block(decay_event_info_line, decay_particles_block, decay_vertex))

        ALP_lhe.write('</LesHouchesEvents>\n')
        decay_lhe.write('</LesHouchesEvents>\n')

if __name__ == '__main__':
    main()
