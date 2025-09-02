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

# Format individual particle information
def format_single_particle_info(single_particle_info_list):
    format_string = ' {:<4} {:>3.0f} {:>2.0f}    {:.0f}    {:.0f}    {:.0f}    {:.0f} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e} {:<+17.10e}\n'
    
    return format_string.format(' ', *single_particle_info_list)

# Format set of ALP production particles. Also find ALP three-momentum
def format_ALP_particle_info(particle_info_list):
    output_str = ''
    key_1 = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2']
    key_2 = ['px', 'py', 'pz', 'e']
    key_3 = ['m', 'lifetime', 'spin']

    for particle_dict in particle_info_list:
        value_list = [particle_dict[key] for key in key_1] + [particle_dict['vector'][key] for key in key_2] + [particle_dict[key] for key in key_3]

        output_str += format_single_particle_info(value_list)

        if particle_dict['id'] == 622:
            three_momentum_dict = particle_dict['vector']

            three_momentum = np.array([three_momentum_dict['px'], three_momentum_dict['py'], three_momentum_dict['pz']])

    return output_str, three_momentum

# Format set of decay particles.
def format_decay_particle_info(particle_info_list):
    output_str = ''
    key_1 = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2']
    key_2 = ['px', 'py', 'pz', 'e']
    key_3 = ['m', 'lifetime', 'spin']

    for particle_dict in particle_info_list:
        value_list = [particle_dict[key] for key in key_1] + [particle_dict['vector'][key] for key in key_2] + [particle_dict[key] for key in key_3]

        # Replace mother id 1 and 2 with -1
        value_list[2] = -1
        value_list[3] = -1

        output_str += format_single_particle_info(value_list)
    return output_str

# Format vertex spacetime
def format_vertex(x_, y_, z_, t_):
    format_string = '#vertex {} {} {} {}\n'
    return format_string.format(x_, y_, z_, t_)

# Compute vertex given ALP 3-momentum
def find_decay_vertex(three_momentum, beam_vertex, mass, decay_min, decay_max):
    c = 299_792_458 * 1000 # [c] = mm / s
    hbar = 6.582119569 * 10**(-16) * 10**(-9) # [hbar] = GeV s

    decay_window = decay_max - decay_min
    decay_length = uniform.rvs(decay_min, decay_window)

    mag = np.linalg.norm(three_momentum)
    unit_vec = three_momentum / mag
    scale = decay_length / unit_vec[2]

    vertex_pos = beam_vertex + unit_vec * scale

    energy = np.sqrt(np.linalg.norm(three_momentum)**2 + mass )
    gamma = energy / mass
    velo_vec = three_momentum  * c / (gamma * mass)
    speed =  np.linalg.norm(velo_vec)
    distance = np.linalg.norm(unit_vec * scale)
    lab_frame_time = distance * 10**9 / speed # [t] = ns

    return format_vertex(*vertex_pos, lab_frame_time)

# Format whole event block
def format_event_block(event_info_line, particle_info_block, vertex_line):
    return '<event>\n' + event_info_line + particle_info_block + vertex_line + '</event>\n'

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'ALP LHE Parser (Uniform Decay).')

    parser.add_argument('-i',   '--input_lhe',                      type = str,     help = 'Input LHE file with full ALP decay process.')
    parser.add_argument('-o',   '--output_dir',                     type = str,     help = 'Output directory.')
    parser.add_argument('-m',   '--mass',                           type = float,   help = 'Simulated ALP mass.')
    parser.add_argument('-min', '--decay_min_z',                    type = int,     help = 'Lab frame minimum decay z position.')
    parser.add_argument('-max', '--decay_max_z',                    type = int,     help = 'Lab frame maximum decay z position.')
    parser.add_argument('-s',   '--seed',                           type = int,     help = 'MadGraph random seed.')
    parser.add_argument('-n_A', '--n_ALP',          default = 5,    type = int,     help = 'Number of particles in ALP production.')
    parser.add_argument('-m_D', '--n_decay',        default = 2,    type = int,     help = 'Number of particles in decay.')

    args = parser.parse_args()

    # Configuring the logger
    logging.basicConfig(format = '[ Parser ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    if not os.path.exists(args.output_dir):
        logging.info('Output directory does not exist and now will be created.')
        os.makedirs(args.output_dir)
    
    input_file = args.input_lhe
    ALP_file = args.output_dir + f'/alp_to_e+e-_{str(args.mass)}GeV_seed{str(args.seed)}_min{str(args.decay_min_z)}_max{str(args.decay_max_z)}_prod.lhe'
    decay_file = args.output_dir + f'/alp_to_e+e-_{str(args.mass)}GeV_seed{str(args.seed)}_min{str(args.decay_min_z)}_max{str(args.decay_max_z)}_decay.lhe'

    # General information from input LHE file
    n_events = pylhe.read_num_events(input_file)
    logging.info(f'Found {n_events} events.')

    with open(input_file, 'r') as input_lhe, open(ALP_file, 'w') as ALP_lhe, open(decay_file, 'w') as decay_lhe:

        # Copy everything until </init>
        for line in input_lhe:

            # Break from initialization header when seeing </init>
            if line == '</init>\n':
                ALP_lhe.write(line)
                decay_lhe.write(line)
                break

            ALP_lhe.write(line)
            decay_lhe.write(line)
        
        # Utilize pylhe to edit events
        events = pylhe.to_awkward(pylhe.read_lhe(input_file))
        for event in events:
            ALP_event_info_line = format_event_info(event['eventinfo'], args.n_ALP)
            decay_event_info_line = format_event_info(event['eventinfo'], args.n_decay)

            ALP_particles = event['particles'][0:args.n_ALP] # List of ALP production particles
            decay_particles = event['particles'][args.n_ALP:(args.n_decay + args.n_ALP)] # List of decay particles

            # Create ALP production event block
            ALP_particles_block, ALP_three_momentum = format_ALP_particle_info(ALP_particles)

            # Beam spot smear. x = [-10, 10], y = [-40, 40]
            beam_x = uniform.rvs(-10, 20)
            beam_y = uniform.rvs(-40, 80)
            beam_z = 0
            beam_t = 0
            ALP_vertex = format_vertex(beam_x, beam_y, beam_z, beam_t)
            ALP_lhe.write(format_event_block(ALP_event_info_line, ALP_particles_block, ALP_vertex))

            # Create decay event block
            decay_particles_block = format_decay_particle_info(decay_particles)
            decay_vertex = find_decay_vertex(ALP_three_momentum, np.array([beam_x, beam_y, beam_z]), args.mass, args.decay_min_z, args.decay_max_z)
            decay_lhe.write(format_event_block(decay_event_info_line, decay_particles_block, decay_vertex))

        ALP_lhe.write('</LesHouchesEvents>\n')
        decay_lhe.write('</LesHouchesEvents>\n')

if __name__ == '__main__':
    main()

