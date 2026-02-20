import os
import time
import pylhe
import logging
import argparse
import subprocess
import numpy as np

from jinja2 import Environment, FileSystemLoader

def write_ldmx_config(config_dir, config_tpl, seed, mass_alp, decay_min_z, decay_max_z):
    # ldmx python config file
    ldmx_config_file = '%sGeV_seed%i_min%s_max%s.py' % (str(mass_alp), seed, str(decay_min_z), str(decay_max_z)) 

    # Result root file
    res_root_file = 'alp_to_e+e-_%sGeV_seed%i_min%s_max%s.root' % (str(mass_alp), seed, str(decay_min_z), str(decay_max_z)) 

    # Find LHE files
    prod_lhe_file = '%s/alpLHE/parsedLHE/%sGeV/alp_to_e+e-_%sGeV_seed%i_min%s_max%s_prod.lhe' % (config_dir, str(mass_alp), str(mass_alp), seed, str(decay_min_z), str(decay_max_z))
    decay_lhe_file = '%s/alpLHE/parsedLHE/%sGeV/alp_to_e+e-_%sGeV_seed%i_min%s_max%s_decay.lhe' % (config_dir, str(mass_alp), str(mass_alp), seed, str(decay_min_z), str(decay_max_z))

    n_events = pylhe.read_num_events(prod_lhe_file)

    # Load configuration file template
    file_loader = FileSystemLoader(config_dir)
    env = Environment(loader = file_loader)
    template = env.get_template(config_tpl)

    # Writing configuration file
    logging.info('Writing LDMX python configuration file.')
    with open(ldmx_config_file, 'w') as file:
        file.write(template.render(
            root_file = res_root_file,
            n_events = n_events,
            seed = seed,
            prod_file = prod_lhe_file,
            decay_file = decay_lhe_file
        ))

    return ldmx_config_file, res_root_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'ALP ldmx Config Generator')

    parser.add_argument('--config_tpl',                             type = str, help = 'Name of ldmx config file template (needs to be in working directory).')
    parser.add_argument('--seed',                                   type = int, help = 'ldmx random seed.')
    parser.add_argument('-min',             '--decay_min_z',        type = int, help = 'Lab frame minimum decay z position.')
    parser.add_argument('-max',             '--decay_max_z',        type = int, help = 'Lab frame maximum decay z position.')
    parser.add_argument('-t', action = 'store_true', dest = 'test',             help = 'Run the job locally instead of submitting it to the batch.')

    args = parser.parse_args()

    # Configuring the logger
    logging.basicConfig(format = '[ Production ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    cwd = os.getcwd()
    config_tpl_path = cwd + '/' + args.config_tpl
    if not config_tpl_path: 
        parser.error('ldmx config template file not found.')

    # Create log output and MG config output 
    output_dir = cwd + '/alpLDMX/decay_min%s_max%s' % (str(args.decay_min_z), str(args.decay_max_z))
    if not os.path.exists(output_dir):
        logging.info('Output directory does not exist and now will be created.')
        os.makedirs(output_dir)

    log_dir = cwd + '/alpLDMX/log' 
    if not os.path.exists(log_dir):
        logging.info('Log output directory does not exist and now will be created.')
        os.makedirs(log_dir)

    ldmx_config_dir = cwd + '/alpLDMX/ldmxConfig' 
    if not os.path.exists(ldmx_config_dir):
        logging.info('ldmx config file output directory does not exist and now will be created.')
        os.makedirs(ldmx_config_dir)

    job_config_dir = cwd + '/alpLDMX/jobConfig' 
    if not os.path.exists(job_config_dir):
        logging.info('Sbatch job config file output directory does not exist and now will be created.')
        os.makedirs(job_config_dir)

    # Start ALP LDMX Production
    logging.getLogger().handlers[0].stream.write("-" * 150 + '\n')

    alp_mass_list = [0.005, 0.025, 0.05, 0.5, 1.0]
    # alp_mass_list = [0.005]

    for mass in alp_mass_list:
        logging.info('Generating ldmx config file for mass %s GeV.' % str(mass))

        mass_dir = output_dir + '/%sGeV' % str(mass) 
        if not os.path.exists(mass_dir):
            logging.info('%s GeV output directory does not exist and now will be created.' % str(mass))
            os.makedirs(mass_dir)

        # Write configuration file replacing template parameters   
        config_file, output_root_file = write_ldmx_config(cwd, args.config_tpl, args.seed, mass, args.decay_min_z, args.decay_max_z)
        logging.info('Copying config file %s to %s.' % (config_file, ldmx_config_dir) )
        os.system('mv %s %s' % (config_file, ldmx_config_dir))

        logging.info('Generating job script for mass %s GeV.' % str(mass))
        with open('%s/sbatch_%sGeV_seed%i_min%s_max%s.sh' % (job_config_dir, str(mass), args.seed, str(args.decay_min_z), str(args.decay_max_z)), 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#SBATCH --job-name=%sGeV_seed%i_min%s_max%s\n' % (str(mass), args.seed, str(args.decay_min_z), str(args.decay_max_z)))
            f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
            f.write('#SBATCH --ntasks-per-node=1\n')
            f.write('#SBATCH -o %s/%sGeV_seed%i_min%s_max%s.out\n' % (log_dir, str(mass), args.seed, str(args.decay_min_z), str(args.decay_max_z)))
            f.write('#SBATCH --mail-type=all\n')
            # f.write('#SBATCH --mail-user=ajige@ucsb.edu\n')
            f.write('#SBATCH -t 12:00:00\n')
            f.write('#SBATCH -D %s\n\n' % cwd)

            f.write('module load apptainer\n')
            f.write('export APPTAINER_USERNS=1\n')
            f.write('denv fire %s\n' % (ldmx_config_dir + '/' + config_file))
            f.write('mv %s %s\n' % (output_root_file, mass_dir))

        # Format submission command and submit if this is not a test
        submit_command = 'sbatch %s/sbatch_%sGeV_seed%i_min%s_max%s.sh' % (job_config_dir, str(mass), args.seed, str(args.decay_min_z), str(args.decay_max_z))
        logging.info('Job submit command: %s\n' % submit_command)

        if not args.test: 
            subprocess.Popen(submit_command, shell=True).communicate()
            time.sleep(3)

if __name__ == '__main__':
    main()
