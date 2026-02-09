#!/usr/bin/env python

#DON'T KNOW IF THESE ARE NEEDED? (imported in other scripts but no clear use)
import math
import platform
import uuid

# IMPORT MODULES NEEDED
import argparse
import glob
import logging
import os
import subprocess
import time


def main():

    # PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s','--script', action='store', dest='script', help='Eval script to run, absolute path')
    parser.add_argument('-i','--indirs', nargs='+', action='store', dest='indirs', help='Director(y/ies) of input files, absolute path(s)')
    parser.add_argument('-g','-groupls', action='store', dest='group_labels', default='', help='Human readable sample labels e.g. for legends')
    parser.add_argument('-o','--outdir', action='store', dest='outdir', help='Directory of output files, absolute path')
    parser.add_argument('-t','--test', action='store_true', dest='test', help='Run the job locally instead of submitting it to the batch')
    args = parser.parse_args()


    # CHECK NECESSARY ARGUMENTS PROVIDED
    if not args.script:
        parser.error('provide eval script, -s or --script')

    if not args.indirs:
        parser.error('provide input directory, -i or --indirs')
    else:
        infiles = [glob.glob(indir + '/*.root') for indir in args.indirs]

    if not args.outdir:
        parser.error('provide output directory, -o or --outdir')

    script = args.script
    outdir = args.outdir

    # CONFIGURE THE LOGGER
    logging.basicConfig(format='[ submitJobs ][ %(levelname)s ]: %(message)s', level=logging.DEBUG)


    # SET BATCH SUBMISSION COMMAND AND SCRIPT RUN COMMAND
    batch_command = 'sbatch'
    command = 'ldmx python3 %s -o %s -m 10000 ' % (script, outdir)
    if args.group_labels:
        command += '-g %s ' % (args.group_labels)
        
    #singularity run --no-home $LDMX_ANALYSIS_IMG . python3 /home/aminali/analysis/running_eval/gabrielle_eval.py --indirs /home/aminali/analysis/running_eval -g kaons -o /home/aminali/analysis/running_eval/evaled
    #singularity run --no-home $LDMX_ANALYSIS_IMG . python3 /home/aminali/analysis/running_eval/gabrielle_eval.py -i /home/aminali/analysis/running_eval/EphotonMIP_2120435_4gev_1e_ecal_upkaons_v12_ldmx-det-v12_run0_hists.root -g kaons -o /home/aminali/analysis/running_eval/evaled

    # CHECK FOR (AND CREATE) JOB AND LOGS DIRECTORIES
    jobdir = outdir + '/jobs/'
    logdir = outdir + '/logs/'

    if not os.path.exists(jobdir):
        logging.info('Creating jobs directory.')
        os.makedirs(jobdir)
    if not os.path.exists(logdir):
        logging.info('Creating log files directory.')
        os.makedirs(logdir)
    logging.info('Job configs will be saved to %s' % jobdir)
    logging.info('Log files will be saved to %s' % logdir)


    #WRITE AND SUBMIT JOB SUBMISSION SCRIPTS
    if args.test:
        logging.info('Testing setup, jobs will not be submitted automatically')

    groups = len(infiles)
    a = 1
    for group in range(0,groups):
        jobs = len(infiles[group])
        for job in range(0,jobs):
            a =a+1
            specific_command = command + '-i %s' % (infiles[group][job])
            with open('%s/slurm_submit_%d.job' % (jobdir,job), 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write('#SBATCH --nodes=1 --ntasks-per-node=1\n')
                f.write('#SBATCH --error=%s/slurm-%%A_%%a.err\n' % logdir)
                f.write('#SBATCH --output=%s/slurm-%%A_%%a.out\n' % logdir)
                #f.write('#SBATCH --time=12:00:00\n\n')
                f.write('#SBATCH --mail-type=BEGIN,END,FAIL\n')
                f.write('#SBATCH --mail-user=ywang6@ucsb.edu\n')
                f.write('cd $SLURM_SUBMIT_DIR\n\n')
                f.write('/bin/hostname\n\n')
                f.write('cat /etc/centos-release\n\n')
                f.write('module load singularity\n\n')
                f.write('source /home/billy/ultimateLDMX/ldmx-sw/scripts/ldmx-env.sh\n\n')
                f.write('%s -g hellothere%s\n' % (specific_command,a))

            submit_command = '%s %s/slurm_submit_%d.job' % (batch_command, jobdir, job)
            logging.info('Job submit command: %s' % submit_command)
            if args.test:
                if job==jobs-1:
                    logging.info('Test command: %s' % specific_command)
                    subprocess.Popen(specific_command, shell=True).wait()
            else:
                subprocess.Popen(submit_command, shell=True).wait()
                time.sleep(2)


if __name__ == "__main__" : 
    main()
