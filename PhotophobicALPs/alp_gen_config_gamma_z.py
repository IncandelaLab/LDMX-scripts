import os
import time
import logging
import argparse
import subprocess
import numpy as np

from jinja2 import Environment, FileSystemLoader

def calc_lambda(mass, decay_length):
    return 1000 * (decay_length / 150)**(1/2) * mass

def calc_decay_width(mass, lambda_e):
    mass_e = 0.000511
    return mass * (mass_e / lambda_e)**2 * np.sqrt(1 - 4*mass_e**2/mass**2) / (8*np.pi)

def write_mg_config(config_dir, config_tpl, n_events, seed, mass, decay_length):

    # Writing MG run script text file
    mg_run_script = '%sGeV_seed%i_decay%s_gamma.txt' % (str(mass), seed,  str(decay_length)) 

    # Result MG directory
    result_mg_dir = 'AlpToe+e-_%sGeV_seed%i_decay%s_gamma' % (str(mass), seed,  str(decay_length)) 

    # Load configuration file template
    fileLoader = FileSystemLoader(config_dir)
    env = Environment(loader = fileLoader)
    template = env.get_template(config_tpl)

    lambda_coupling = calc_lambda(mass, decay_length)
    width = calc_decay_width(mass, lambda_coupling)

    logging.info('Lambda coupling: %s' % str(lambda_coupling))
    logging.info('Decay width: %s' % str(width))

    # Writing configuration file
    logging.info('Writing MG configuration file.')
    with open(mg_run_script, 'w') as file:
        file.write(template.render(
            dir_name = result_mg_dir,
            mass_alp = mass,
            n_events = n_events,
            seed = seed,
            width = width,
            lambda_e = lambda_coupling
        ))

    return mg_run_script, result_mg_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'ALP Library Generator')

    parser.add_argument('--config_tpl',   type = str,                   help = 'Name of MG run script template (needs to be in working directory).')
    parser.add_argument('--parser',       type = str,                   help = 'Name of LHE file parser (needs to be in working directory).')
    parser.add_argument('--n_events',     type = int,                   help = 'Number of events to generate.')
    parser.add_argument('--seed',         type = int,                   help = 'MadGraph random seed.')
    parser.add_argument('--decay_length', type = int,                   help = 'Lab frame decay length.')
    parser.add_argument('-t', action = 'store_true', dest = 'test',     help = 'Run the job locally instead of submitting it to the batch.')

    args = parser.parse_args()

    # Configuring the logger
    logging.basicConfig(format = '[ Production ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    cwd = os.getcwd()
    config_tpl_path = cwd + '/' + args.config_tpl

    if not config_tpl_path: 
        parser.error('MG config template file not found.')

    parser_path = cwd + '/' + args.parser

    if not parser_path: 
        parser.error('LHE parser file not found.')

    # Create log output and MG config output 
    output_dir = cwd + '/ALP_MG_Directory/gamma_decay%s' % str(args.decay_length)
    if not os.path.exists(output_dir):
        logging.info('Output directory does not exist and now will be created.')
        os.makedirs(output_dir)

    log_out = output_dir + '/log_output' 
    if not os.path.exists(log_out):
        logging.info('Log output directory does not exist and now will be created.')
        os.makedirs(log_out)

    mg_config_out = output_dir + '/mg_config_output' 
    if not os.path.exists(mg_config_out):
        logging.info('MG config file output directory does not exist and now will be created.')
        os.makedirs(mg_config_out)

    job_script_out = output_dir + '/job_script_output' 
    if not os.path.exists(job_script_out):
        logging.info('Sbatch job script output directory does not exist and now will be created.')
        os.makedirs(job_script_out)

    lhe_dir = cwd + '/ALP_LHE_Directory/gamma_decay%s' % str(args.decay_length)
    if not os.path.exists(lhe_dir):
        logging.info('Parsed LHE file directory does not exist and now will be created.')
        os.makedirs(lhe_dir)

    ALP_mass_list = [0.005, 0.025, 0.05, 0.5, 1.0]

    for mass in ALP_mass_list:
        logging.info('Generating MG config file for mass %s GeV' % str(mass))

        mass_dir = output_dir + '/%sGeV' % str(mass) 
        if not os.path.exists(mass_dir):
            logging.info('%s GeV output directory does not exist and now will be created ' % str(mass))
            os.makedirs(mass_dir)

        lhe_mass_dir = lhe_dir  + '/%sGeV' % str(mass) 
        if not os.path.exists(lhe_mass_dir):
            logging.info('%s GeV LHE output directory does not exist and now will be created ' % str(mass))
            os.makedirs(lhe_mass_dir)

        # Write configuration file replacing template parameters   
        config_file, mg_output_dir = write_mg_config(cwd, args.config_tpl, args.n_events, args.seed,  mass, args.decay_length)

        logging.info('Copying config file %s to %s' % (config_file, mg_config_out) )
        os.system('mv %s %s' % (config_file, mg_config_out))

        lhe_file_path = '%s/%s/Events/run_01/unweighted_events.lhe' % (mass_dir, mg_output_dir)

        logging.info('Generating job script for mass %s GeV.' % str(mass))
        with open('%s/sbatch_%sGeV_seed%i_gamma.sh' % (job_script_out, str(mass), args.seed), 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#SBATCH --job-name=%sGeV_seed%i\n' % (str(mass), args.seed))
            f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
            f.write('#SBATCH --ntasks-per-node=1\n')
            f.write('#SBATCH -o %s/%sGeV_seed%i.out\n' % (log_out, str(mass), args.seed))
            f.write('#SBATCH --mail-type=all\n')
            # f.write('#SBATCH --mail-user=$EMAIL_ADDRESS\n')
            f.write('#SBATCH -t 12:00:00\n')
            f.write('#SBATCH -D %s\n\n' % cwd)

            f.write('module load apptainer\n')
            f.write('export APPTAINER_USERNS=1\n')
            f.write('apptainer exec madpydel_latest.sif /madgraph/MG5_aMC_v3_5_4/bin/mg5_aMC %s\n' % (mg_config_out + '/' + config_file))
            f.write('mv %s %s\n' % (mg_output_dir, mass_dir))

            # Parse LHE file
            f.write('gzip -d %s \n' % (lhe_file_path + '.gz'))
            f.write('python %s -i %s -o %s -m %f  -s %i' % (parser_path, lhe_file_path, lhe_mass_dir, mass, args.seed) )

        # Format submission command and submit if this is not a test
        submit_command = 'sbatch %s/sbatch_%sGeV_seed%i_gamma.sh' % (job_script_out, str(mass), args.seed)
        logging.info('Job submit command: %s\n' % submit_command)

        if not args.test: 
            subprocess.Popen(submit_command, shell=True).communicate()
            time.sleep(3)

if __name__ == '__main__':
    main()
