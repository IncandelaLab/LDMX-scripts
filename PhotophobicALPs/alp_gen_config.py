import os
import time
import logging
import argparse
import subprocess
import numpy as np

from jinja2 import Environment, FileSystemLoader

def calc_decay_width(mass_alp):
    mass_e = 0.000511 # [GeV]
    ref_lambda_e = 1 # [GeV]
    return mass_alp * (mass_e / ref_lambda_e)**2 * np.sqrt(1 - 4*mass_e**2/mass_alp**2) / (8*np.pi)

def write_mg_config(config_dir, config_tpl, n_events, seed, mass_alp):
    # MG run script text file
    mg_run_script = '%sGeV_seed%i.txt' % (str(mass_alp), seed) 

    # Resulting MG directory
    mg_res_dir = 'alp_to_e+e-_%sGeV_seed%i' % (str(mass_alp), seed) 

    # Load configuration file template
    file_loader = FileSystemLoader(config_dir)
    env = Environment(loader = file_loader)
    template = env.get_template(config_tpl)

    width = calc_decay_width(mass_alp)
    logging.info('ALP Width at Reference Lambda_e = 1 GeV: %s' % str(width))

    # Writing configuration file
    logging.info('Writing MG configuration file.')
    with open(mg_run_script, 'w') as file:
        file.write(template.render(
            dir_name = mg_res_dir,
            mass_alp = mass_alp,
            n_events = n_events,
            seed = seed,
            width = width,
            lambda_e = 1 # Note that we are utilizing a reference lambda_e value of 1 GeV 
        ))
    return mg_run_script, mg_res_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'ALP Library Generator')

    parser.add_argument('--mg_config_tpl',          type = str,                 help = 'Name of MG run script template (needs to be in working directory).')
    parser.add_argument('--parser',                 type = str,                 help = 'Name of LHE file parser (needs to be in working directory).')
    parser.add_argument('--n_events',               type = int,                 help = 'Number of events to generate.')
    parser.add_argument('--seed',                   type = int,                 help = 'MadGraph random seed.')
    parser.add_argument('-min', '--decay_min_z',    type = int,                 help = 'Lab frame minimum decay z position.')
    parser.add_argument('-max', '--decay_max_z',                    type = int, help = 'Lab frame maximum decay z position.')
    parser.add_argument('-t', action = 'store_true', dest = 'test',             help = 'Run the job locally instead of submitting it to the batch.')

    args = parser.parse_args()

    # Configuring the logger
    logging.basicConfig(format = '[ Production ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    cwd = os.getcwd()
    mg_config_tpl_path = cwd + '/' + args.mg_config_tpl
    if not mg_config_tpl_path: 
        parser.error('MG config template file not found.')

    parser_path = cwd + '/' + args.parser
    if not parser_path: 
        parser.error('LHE parser file not found.')

    # Create log output and MG config output 
    output_dir = cwd + '/alpLHE'
    if not os.path.exists(output_dir):
        logging.info('Output directory does not exist and now will be created.')
        os.makedirs(output_dir)

    log_dir = output_dir + '/log' 
    if not os.path.exists(log_dir):
        logging.info('Log output directory does not exist and now will be created.')
        os.makedirs(log_dir)

    mg_config_dir = output_dir + '/mgConfig' 
    if not os.path.exists(mg_config_dir):
        logging.info('MG config file output directory does not exist and now will be created.')
        os.makedirs(mg_config_dir)

    job_config_dir = output_dir + '/jobConfig' 
    if not os.path.exists(job_config_dir):
        logging.info('Sbatch job config file output directory does not exist and now will be created.')
        os.makedirs(job_config_dir)

    mg_lhe_dir = output_dir + '/mgLHE'
    if not os.path.exists(mg_lhe_dir):
        logging.info('MG LHE file directory does not exist and now will be created.')
        os.makedirs(mg_lhe_dir)
    
    parsed_lhe_dir = output_dir + '/parsedLHE'
    if not os.path.exists(parsed_lhe_dir):
        logging.info('Parsed LHE file directory does not exist and now will be created.')
        os.makedirs(parsed_lhe_dir)

    # Start ALP MadGraph Production
    logging.getLogger().handlers[0].stream.write("-" * 150 + '\n')

    alp_mass_list = [0.005, 0.025, 0.05, 0.5, 1.0]
    
    for mass in alp_mass_list:
        logging.info('Generating MG config file for mass %s GeV.' % str(mass))

        mg_mass_dir = mg_lhe_dir + '/%sGeV' % str(mass) 
        if not os.path.exists(mg_mass_dir):
            logging.info('%s GeV MG LHE file output directory does not exist and now will be created.' % str(mass))
            os.makedirs(mg_mass_dir)

        parsed_mass_dir = parsed_lhe_dir  + '/%sGeV' % str(mass) 
        if not os.path.exists(parsed_mass_dir):
            logging.info('%s GeV parsed LHE file output directory does not exist and now will be created.' % str(mass))
            os.makedirs(parsed_mass_dir)

        # Write configuration file replacing template parameters   
        config_file, mg_output_dir = write_mg_config(cwd, args.mg_config_tpl, args.n_events, args.seed,  mass)
        logging.info('Copying MG config file %s to %s.' % (config_file, mg_config_dir) )
        os.system('mv %s %s' % (config_file, mg_config_dir))

        mg_lhe_file_path = '%s/%s/Events/run_01/unweighted_events.lhe' % (cwd, mg_output_dir)
        mg_lhe_file_name = 'alp_to_e+e-_%sGeV_seed%i.lhe' % (str(mass), args.seed)

        logging.info('Generating job script for mass %s GeV.' % str(mass))
        with open('%s/sbatch_%sGeV_seed%i.sh' % (job_config_dir, str(mass), args.seed), 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#SBATCH --job-name=%sGeV_seed%i\n' % (str(mass), args.seed))
            f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
            f.write('#SBATCH --ntasks-per-node=1\n')
            f.write('#SBATCH -o %s/%sGeV_seed%i.out\n' % (log_dir, str(mass), args.seed))
            f.write('#SBATCH --mail-type=all\n')
            # f.write('#SBATCH --mail-user=$EMAIL_ADDRESS\n')
            f.write('#SBATCH -t 12:00:00\n')
            f.write('#SBATCH -D %s\n\n' % cwd)

            f.write('module load apptainer\n')
            f.write('apptainer exec madpydel_latest.sif /madgraph/MG5_aMC_v3_5_4/bin/mg5_aMC %s\n' % (mg_config_dir + '/' + config_file))
            
            # Clean file name and MG directory
            f.write('gzip -d %s \n' % (mg_lhe_file_path + '.gz'))
            f.write('mv %s %s\n' % (mg_lhe_file_path, mg_mass_dir))
            f.write('mv %s %s\n' % (mg_mass_dir + '/unweighted_events.lhe', mg_mass_dir + '/' + mg_lhe_file_name))
            f.write('rm -rf  %s\n' % (mg_output_dir))

            # Parse LHE file
            f.write('python3 %s -i %s -o %s -m %f -min %i -max %i -s %i' % (parser_path, mg_mass_dir + '/' + mg_lhe_file_name, parsed_mass_dir, mass, args.decay_min_z, args.decay_min_z, args.seed) )

        # Format submission command and submit if this is not a test
        submit_command = 'sbatch %s/sbatch_%sGeV_seed%i.sh' % (job_config_dir, str(mass), args.seed)
        logging.info('Job submit command: %s\n' % submit_command)

        if not args.test: 
            subprocess.Popen(submit_command, shell=True).communicate()
            time.sleep(3)

if __name__ == '__main__':
    main()
