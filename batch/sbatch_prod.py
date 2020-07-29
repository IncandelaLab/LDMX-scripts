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
    parser.add_argument('-n', action='store', type=int, dest='npertask', default=40,
                        help='Number of jobs per slurm task.')
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

    # If a path to a directory containing files to process if specified, use 
    # the number of files to determine the number of jobs to submit. Otherwise, 
    # make sure the user has specified the number of jobs to run. 
    jobs = 0
    files = []
    if 'input_path' in config:
        (_, _, files) = next(os.walk(config['input_path'].strip()))
        jobs = len(files)
    elif 'jobs' in config: 
        jobs = int(config['jobs'])
    else: parser.error('An input path or the total number of jobs needs to be specified.') 
    logging.info('Preparing to run %s jobs' % jobs)

    # Determine how many jobs to submit at a time.  By default, 1000 jobs will
    # be submitted to the batch.  Once all jobs are running, another array
    # of jobs will be submitted. 
    # Not currently used
    #job_array = 1000
    #if 'job_array' in config: 
    #    job_array = int(config['job_array'])
    #logging.info('Submitting jobs in batches of %s' % job_array)

    # Get the command that will be used to submit jobs to the batch system
    batch_command = config['batch_command']

    # Build the command that will be executed for each job
    command = 'python %s' % config['command']['script']
    for key, value in config['command']['arguments'].items(): 
        command += ' --%s %s' % (key, value)
    logging.info('Command to be executed: %s' % command)

    # Set starting seed number. Should be incremented with each batch submission
    seed = 0
    if args.seed: 
        seed = int(args.seed)
    logging.info('Starting seeds from %s' % seed)

    # Get the path where all files will be stored. If the path doesn't
    # exist, create it.
    odir = config['command']['arguments']['output_path'].strip()
    if not os.path.exists(odir): 
        logging.info('Output directory does not exist and will be created.')
        os.makedirs(odir)
    logging.info('All files will be save to %s' % odir)

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

    # LHE input submission (not yet set up)
    if len(files) != 0:
        for job in range(0, jobs):
            submit_command = command + ' --input %s/%s' % (config['input_path'], files[job])
            logging.info('Processing file ( %s ) %s/%s' % (job, config['input_path'], files[job]))
        if args.seed:
            submit_command += ' --seed %s' % seed
            seed += 2

        if not args.test:
            submit_command = '%s %s' % (batch_command, submit_command)
   
        subprocess.Popen(submit_command, shell=True).wait()

    # Standard submission
    elif jobs > 0: 
        # Calculated number of jobs, setting number of tasks per job to npertask
        njobs, nlast = int(jobs)/args.npertask + 1, int(jobs)%args.npertask
        if nlast == 0:
            njobs -= 1

        # Set up slurm job submission scripts
        for job in range(0, njobs):
            logging.info('Job %s' % job)

            arraymax = args.npertask-1
            if job==njobs-1 and nlast>0:
                arraymax=nlast-1

            python_command = command + ' --run %s' % (job+1) 

            # Set seed and increment it for the next set of tasks
            if args.seed:
                python_command += ' --seed %s' % seed
                seed += args.npertask*2

            # Write job submission script
            with open('%s/slurm_submit_%d.job' % (jobdir,job), 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
                f.write('#SBATCH --array=0-%d\n' % arraymax)
                f.write('#SBATCH --error=%s/slurm-%%A_%%a.err\n' % logdir)
                f.write('#SBATCH --output=%s/slurm-%%A_%%a.out\n' % logdir)
                f.write('#SBATCH --time=12:00:00\n\n')
                f.write('cd $SLURM_SUBMIT_DIR\n\n')
                f.write('/bin/hostname\n\n')
                f.write('cat /etc/centos-release\n\n')
                f.write ('module load singularity\n\n')
                f.write('%s\n' % python_command)

            # Format submission command and submit if this is not a test
            submit_command = '%s %s/slurm_submit_%d.job' % (batch_command,jobdir,job)
            logging.info('Job submit command: %s' % submit_command)
            if not args.test: 
                subprocess.Popen(submit_command, shell=True).communicate()
                time.sleep(3)


if __name__ == "__main__" : 
    main()
    
