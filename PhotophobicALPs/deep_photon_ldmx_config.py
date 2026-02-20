import os
import time
import logging
import argparse
import subprocess
import numpy as np

from jinja2 import Environment, FileSystemLoader

def write_mg_config(config_dir, config_tpl, n_events, seed, decay_z):
    # Define LDMX python config filename
    ldmx_config_file = 'deep_photon_conv_ldmx_seed%i_decay%s.py' % (seed, str(decay_z)) 

    # Define result root filename
    result_root_file = 'deep_photon_conv_ldmx_seed%i_decay%s.root' % (seed,  str(decay_z)) 

    # Load configuration file template
    fileLoader = FileSystemLoader(config_dir)
    env = Environment(loader = fileLoader)
    template = env.get_template(config_tpl)

    # Render and write the LDMX configuration file
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

    # Configure logging
    logging.basicConfig(format = '[ Production ][ %(levelname)s ]: %(message)s', level = logging.DEBUG)

    cwd = os.getcwd()
    config_tpl_path = cwd + '/' + args.config_tpl

    if not config_tpl_path: 
        parser.error('ldmx config template file not found.')

    # Create output directory for decay configuration
    output_dir = cwd + '/deepPhotonLDMX/decay%s' % str(args.decay_z)
    if not os.path.exists(output_dir):
        logging.info('Output directory does not exist and now will be created.')
        os.makedirs(output_dir)

    # Create log directory
    log_dir = cwd + '/deepPhotonLDMX/log' 
    if not os.path.exists(log_dir):
        logging.info('Log output directory does not exist and now will be created.')
        os.makedirs(log_dir)

    # Create LDMX config directory
    ldmx_config_dir = cwd + '/deepPhotonLDMX/ldmxConfig' 
    if not os.path.exists(ldmx_config_dir):
        logging.info('ldmx config file output directory does not exist and now will be created.')
        os.makedirs(ldmx_config_dir)

    # Create job config directory
    job_config_dir  = cwd + '/deepPhotonLDMX/jobConfig' 
    if not os.path.exists(job_config_dir ):
        logging.info('Sbatch job config file output directory does not exist and now will be created.')
        os.makedirs(job_config_dir )
    
    # Initialize production
    logging.getLogger().handlers[0].stream.write("-" * 150 + '\n')

    events_per_job = int(args.n_events / args.n_jobs)

    for i in range(0, args.n_jobs, 8):
        batch_id = int(i / 8) # Calculate current batch ID
        logging.info('Generating batch job script for batch %i (indices %i-%i).' % (batch_id, args.seed + i, args.seed + i + 7))

        # Define SLURM script path
        slurm_script_name = '%s/sbatch_batch%i_decay%s.sh' % (job_config_dir, batch_id, str(args.decay_z))

        with open(slurm_script_name, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#SBATCH --job-name=batch%i_decay%s\n' % (batch_id, str(args.decay_z)))
            f.write('#SBATCH --nodes=1\n')
            # Configure resources for 8 parallel tasks
            f.write('#SBATCH --ntasks-per-node=8\n')
            f.write('#SBATCH --ntasks=8\n')
            f.write('#SBATCH --cpus-per-task=1\n')
            f.write('#SBATCH --mem=32000M\n')
            f.write('#SBATCH -o %s/batch%i_decay%s.out\n' % (log_dir, batch_id, str(args.decay_z)))
            f.write('#SBATCH --mail-type=FAIL\n')
            f.write('#SBATCH -t 7-12:00:00\n')
            f.write('#SBATCH -D %s\n\n' % cwd)

            f.write('module load apptainer\n')
            f.write('export APPTAINER_USERNS=1\n')
            
            # Inner loop to handle 8 sub-tasks
            for k in range(8):
                # Calculate current job index and seed
                current_job_idx = i + k
                
                # Prevent overflow if total jobs are not a multiple of 8
                if current_job_idx >= args.n_jobs:
                    break
                
                current_seed = args.seed + current_job_idx

                logging.info('Generating ldmx config file for run %s' % str(current_seed))

                # Generate specific configuration file
                config_file, output_root_file = write_mg_config(cwd, args.config_tpl, events_per_job, current_seed,  args.decay_z)

                logging.info('Copying config file %s to %s.' % (config_file, ldmx_config_dir) )
                os.system('mv %s %s' % (config_file, ldmx_config_dir))

                # Construct commands to run generation and move output in parallel
                cmd_run = 'denv fire %s' % (ldmx_config_dir + '/' + config_file)
                cmd_mv = 'mv %s %s' % (output_root_file, output_dir)
                
                # Write command block to run in background
                f.write('( %s && %s ) &\n' % (cmd_run, cmd_mv))

            # Wait for all background tasks to complete
            f.write('wait\n')        
        
        # Submit the SLURM job
        submit_command = 'sbatch %s' % slurm_script_name
        logging.info('Job submit command: %s\n' % submit_command)

        if not args.test: 
            subprocess.Popen(submit_command, shell=True).communicate()
            time.sleep(3)

if __name__ == '__main__':
    main()
