import os
import time
import logging
import argparse
import subprocess
import numpy as np

from jinja2 import Environment, FileSystemLoader

def write_mg_config(config_dir, config_tpl, n_events, seed, decay_z):
    # ldmx python config file
    ldmx_config_file = 'deep_photon_conv_ldmx_seed%i_decay%s.py' % (seed, str(decay_z)) 

    # Result root file
    result_root_file = 'deep_photon_conv_ldmx_seed%i_decay%s.root' % (seed,  str(decay_z)) 

    # Load configuration file template
    fileLoader = FileSystemLoader(config_dir)
    env = Environment(loader = fileLoader)
    template = env.get_template(config_tpl)

    # Writing configuration file template
    logging.info('Writing LDMX python configuration file.')
    with open(ldmx_config_file, 'w') as file:
        file.write(template.render(
            root_file_name = result_root_file,
            seed = seed,
            decay_length = float(decay_z),
            n_events = n_events
        ))

    return ldmx_config_file, result_root_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Deep Photon Generator')

    parser.add_argument('--config_tpl', type = str,                 help = 'Name of ldmx config template (needs to be in working directory).')
    parser.add_argument('--n_events',   type = int,                 help = 'Number of total events to generate.')
    parser.add_argument('--decay_z',    type = int,                 help = 'Lab frame decay z position')
    parser.add_argument('--seed',       type = int,                 help = 'Starting ldmx  random seed.')
    parser.add_argument('--n_jobs',     type = int,                 help = 'Number of sbatch jobs.')
    parser.add_argument('-t', action = 'store_true', dest = 'test', help = 'Run the job locally instead of submitting it to the batch.')

    args = parser.parse_args()

    # Configuring the logger
    logging.basicConfig(format = '[ Production ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    cwd = os.getcwd()
    config_tpl_path = cwd + '/' + args.config_tpl

    if not config_tpl_path: 
        parser.error('ldmx config template file not found.')

    # Create log output and ldmx config output 
    output_dir = cwd + '/deepPhotonLDMX/decay%s' % str(args.decay_z)
    if not os.path.exists(output_dir):
        logging.info('Output directory does not exist and now will be created.')
        os.makedirs(output_dir)

    log_dir = cwd + '/deepPhotonLDMX/log' 
    if not os.path.exists(log_dir):
        logging.info('Log output directory does not exist and now will be created.')
        os.makedirs(log_dir)

    ldmx_config_dir = cwd + '/deepPhotonLDMX/ldmxConfig' 
    if not os.path.exists(ldmx_config_dir):
        logging.info('ldmx config file output directory does not exist and now will be created.')
        os.makedirs(ldmx_config_dir)

    job_config_dir  = cwd + '/deepPhotonLDMX/jobConfig' 
    if not os.path.exists(job_config_dir ):
        logging.info('Sbatch job config file output directory does not exist and now will be created.')
        os.makedirs(job_config_dir )
    
    # Start deep photon Production
    logging.getLogger().handlers[0].stream.write("-" * 150 + '\n')

    events_per_job = int(args.n_events / args.n_jobs)

    for i in range(args.n_jobs):
        logging.info('Generating ldmx config file for run %s' % str(args.seed + i))

        # Write configuration file replacing template parameters   
        config_file, output_root_file = write_mg_config(cwd, args.config_tpl, events_per_job, args.seed + i,  args.decay_z)

        logging.info('Copying config file %s to %s.' % (config_file, ldmx_config_dir) )
        os.system('mv %s %s' % (config_file, ldmx_config_dir))

        # root_file_path = '%s/%s' % (cwd, output_root_file)

        logging.info('Generating job script for seed %i.' % (args.seed + i))
        with open('%s/sbatch_seed%i_decay%s.sh' % (job_config_dir, args.seed + i, str(args.decay_z)), 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#SBATCH --job-name=seed%i_decay%s\n' % (args.seed + i, str(args.decay_length)))
            f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
            f.write('#SBATCH --ntasks-per-node=1\n')
            f.write('#SBATCH -o %s/seed%i_decay%s.out\n' % (log_dir, args.seed + i, str(args.decay_length)))
            f.write('#SBATCH --mail-type=all\n')
            # f.write('#SBATCH --mail-user=ajige@ucsb.edu\n')
            f.write('#SBATCH -t 7-12:00:00\n')
            f.write('#SBATCH -D %s\n\n' % cwd)

            f.write('module load apptainer\n')
            f.write('denv fire %s\n' % (ldmx_config_dir + '/' + config_file))
            f.write('mv %s %s\n' % (output_root_file, output_dir))

        # Format submission command and submit if this is not a test
        submit_command = 'sbatch %s/sbatch_seed%i_decay%s.sh' % (job_config_dir, args.seed + i, str(args.decay_z))
        logging.info('Job submit command: %s\n' % submit_command)

        if not args.test: 
            subprocess.Popen(submit_command, shell=True).communicate()
            time.sleep(3)

if __name__ == '__main__':
    main()
