#!/usr/bin/env python

import argparse
import logging
import platform
import os
import subprocess
import time
import yaml
import uuid

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config',
                        help="YAML file used to configure the job submission.")
    parser.add_argument('-s', action='store', dest='seed', 
                        help='Initial value of seed to use for when submitting jobs.')
    parser.add_argument("-t", action='store_true', dest='test',
                        help="Run the job locally instead of submitting it to the batch.")
    args = parser.parse_args()

    # If a configuration file was not specified, warn the user and exit.
    if not args.config :
        parser.error('A configuration file needs to be specified.')

    # Configure the logger
    logging.basicConfig(format='[ submitJobs ][ %(levelname)s ]: %(message)s', 
            level=logging.DEBUG)
   
    # Parse the configuration file.
    logging.info('Parsing configuration located at %s' % args.config.strip())
    config = yaml.load(open(args.config.strip(), 'r')) #, Loader=yaml.FullLoader)

    # Check for required parameters
    jobs = 0
    input_files = []
    input_dir = config['input_dir'].strip()

    if 'input_dir' in config :
        input_files = os.listdir(input_dir)
        logging.info('Getting input files from %s' % input_dir)
        jobs = len(input_files)
    
    if 'jobs' in config: 
        # jobs is minimum of available input files and number passed
        if 'input_dir' in config : jobs = min( jobs , int(config['jobs']) )
        else : jobs = int(config['jobs'])
        logging.info('Submitting %s jobs.' % jobs)
    elif 'input_dir' not in config : 
        logging.warning('Need to specify the number of jobs (jobs) or a list of input files (input_dir).')
        exit()

    config_template = ''

    if 'config' in config:
        config_template = config['config']
        logging.info('Config template located at \'%s\'' % config_template)
    else:
        logging.warning('A python config template is required.')
        exit()

    if 'event_output' not in config and 'histogram_output' not in config:
        logging.warning('Either \'event_output\' or \'histogram_output\' must be given.')
        exit()

    # Check for optional parameters and build command to run executable

    # Use the user specified run script if it has been specified instead.
    run_script = '%s/run_ana.py' % os.getcwd()
    if 'run_script' in config: run_script = os.path.realpath(config['run_script'])

    command = 'python %s %s ' %( run_script , config_template )

    # Get prefix for output file names
    oprefix = ''
    if 'prefix' in config:
        oprefix = config['prefix'].strip()
    command += '--prefix %s ' % ( oprefix )

    # Get the path where all files will be stored. If the path doesn't exist, create it.
    if 'event_output' in config:
        eventDir = config['event_output'].strip()
        if not os.path.exists(eventDir): 
            logging.info('Output directory does not exist and will be created.')
            os.makedirs(eventDir)
        logging.info('All event files will be save to %s' % eventDir)
        command += '--eventOut %s ' % ( eventDir )
    else :
        logging.info( 'Event files will not be saved.' )

    if 'histogram_output' in config:
        histogramDir = config['histogram_output'].strip()
        if not os.path.exists(histogramDir): 
            logging.info('Output directory does not exist and will be created.')
            os.makedirs(histogramDir)
        logging.info('All histogram files will be save to %s' % histogramDir)
        command += '--histOut %s ' % ( histogramDir )
    else :
        logging.info( 'Histogram files will not be saved.' )

    # Define command to submit to batch
    batch_command = 'sbatch'
    if 'batch_command' in config:
        batch_command = config['batch_command'].strip()

    # Set starting seed number. Should be incremented with each batch submission (only needed for simulation jobs)
    seed = 1
    if args.seed: 
        seed = int(args.seed)
    logging.info('Starting seeds from %s' % seed)

    # Create the path where the job configs will be stored
    jobdir = config['job_dir'].strip()
    logdir = '%s/logs' % (jobdir)
    if not os.path.exists(jobdir):
        logging.info('Creating jobs directory.')
        os.makedirs(jobdir)
    if not os.path.exists(logdir):
        logging.info('Creating log files directory.')
        os.makedirs(logdir)
    logging.info('Job configs will be save to %s' % jobdir)
    logging.info('Log files will be save to %s' % logdir)

    # Submit the jobs

    if args.test:
        logging.info('Testing setup, jobs will not be submitted automatically')

    for job in range(0, jobs):
        # Add arguments to python command
        if 'input_dir' in config:
            specific_command += '--inputFile %s/%s ' % ( input_dir, input_files[job] )

        specific_command += ' --run %s' % (job+1) 

        # Set seed and increment it for the next set of tasks
        specific_command += ' --seed %s' % seed
        seed += 2

        # Write job submission script
        with open('%s/slurm_submit_%d.job' % (jobdir,job), 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
            f.write('#SBATCH --error=%s/slurm-%%A_%%a.err\n' % logdir)
            f.write('#SBATCH --output=%s/slurm-%%A_%%a.out\n' % logdir)
            f.write('#SBATCH --time=12:00:00\n\n')
            f.write('cd $SLURM_SUBMIT_DIR\n\n')
            f.write('/bin/hostname\n\n')
            f.write('cat /etc/centos-release\n\n')
            f.write ('module load singularity\n\n')
            f.write('%s\n' % specific_command)

        # Format submission command and submit if this is not a test
        submit_command = '%s %s/slurm_submit_%d.job' % (batch_command,jobdir,job)
        logging.info('Job submit command: %s' % submit_command)
        if args.test:
            if job==jobs-1:
                logging.info('Test command: %s --test' % specific_command)
                subprocess.Popen(specific_command + ' --test', shell=True).wait()
        else:
            subprocess.Popen(submit_command, shell=True).wait()
            time.sleep(1)

if __name__ == "__main__" : 
    main()
    
