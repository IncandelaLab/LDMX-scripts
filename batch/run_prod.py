#!/usr/bin/env python

import argparse
import glob
import json
import logging
import os
import platform
import random
import subprocess
import time
import sys
import shutil

from jinja2 import Environment, FileSystemLoader

def write_config(config, run, seed1, seed2, prefix, detector):
    
    config_path = '%s.py' % prefix.replace('.', '_') 
    logging.info('Writing configuration to %s' % config_path)

    rfile = '%s_%s_run%s_seeds_%s_%s.root' % (prefix, detector, run, seed1, seed2)
    logging.info('Saving file to %s' % rfile)

    file_loader = FileSystemLoader(searchpath='./')
    env = Environment(loader=file_loader)
    template = env.get_template(config)

    with open(config_path, 'w') as fh:
        fh.write(template.render(
                run=run, 
                detector=detector,
                seed1=seed1, 
                seed2=seed2,
                outputFile=rfile
        ))
   
    print fh

    return config_path, rfile

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--env',         help='Path to the environment.') 
    parser.add_argument('-o', '--output_path', help='Path to save output to.')
    parser.add_argument("-d", '--detector',    help="The detector to use to generate events.")
    parser.add_argument('-c', '--config',      help='Config template to use.')
    parser.add_argument('-p', '--prefix',      help='The file prefix.')
    parser.add_argument('-r', '--run',         help='The run number.') 
    parser.add_argument('-s', '--seed',        help='The seed number.') 
    args = parser.parse_args()

    #if not args.env: 
    #    parser.error('A setup script needs to be specified.') 

    if not args.output_path: 
        parser.error('Please specify a path to write the output files to.')

    if not args.config: 
        parser.error('Python configuration file needs to be specified.')

    # Configure the logger
    logging.basicConfig(format='[ Production ][ %(levelname)s ]: %(message)s', 
            level=logging.DEBUG)

    # Set run number and seeds
    run = int(args.run)
    seed1 = int(args.seed)

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        logging.info('SLURM_ARRAY_TASK_ID: %s', os.environ['SLURM_ARRAY_TASK_ID'])
        seed1 += int(os.environ['SLURM_ARRAY_TASK_ID'])
    seed2 = seed1 + 1
    logging.info('Using seeds %d, %d' % (seed1, seed2))

    # First, create a user directory within scratch or locally
    if os.access('/tmp/', os.W_OK): 
        scratch_dir = '/tmp/%s' % os.environ['USER']
    else:
        scratch_dir = '/home/%s/ldmx-prod/production/scratch' % os.environ['USER']
    logging.info('Using scratch path %s' % scratch_dir)
    if not os.path.exists(scratch_dir):
        logging.info('Scratch directory does not exist and will be created.')
        os.makedirs(scratch_dir)

    # Now, create a directory within the user directory in scratch
    if 'SLURM_JOBID' in os.environ:  
        tmp_dir = '%s/%s' % (scratch_dir, os.environ['SLURM_JOBID'])
        logging.info('SLURM_JOBID: %s' % os.environ['SLURM_JOBID'])
    else: 
        tmp_dir = '%s/%s' % (scratch_dir, 'test')
    logging.info('Using tmp directory %s' % tmp_dir)
    if not os.path.exists(tmp_dir):
        logging.info('tmp directory does not exist and will be created')
        os.makedirs(tmp_dir)

    # Move into the directory
    os.chdir(tmp_dir)
 
    # If the directory isn't empty, clear it
    if os.listdir(tmp_dir):
        logging.info('Cleaning directory %s' % tmp_dir)
        files = glob.glob('%s/*' % tmp_dir)
        for f in files: 
            logging.info('Removing %s' % f)
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
            else:
                logging.info('%s is not a file or directory' % f)

    # Create a link to the config template
    os.symlink(args.config, 'config.py')

    # Write config replacing template parameters
    config_path, rfile = write_config('config.py', run, seed1, seed2, args.prefix, args.detector )

    # Run the config
    print 'Using production image %s' % os.environ.get('LDMX_PRODUCTION_IMG')
    command = "singularity run --no-home %s . %s" % (os.environ.get('LDMX_PRODUCTION_IMG'),config_path) 
    subprocess.Popen(command, shell=True).wait()

    # Create output directory if needed and copy output file into it
    prod_dir = args.output_path
    if not os.path.exists(prod_dir):
        logging.info('Production directory does not exist and will be created.')
        os.makedirs(prod_dir)

    logging.info('Copying sim file to %s' % prod_dir)
    os.system('cp -r %s %s' % (rfile, prod_dir))

    # Clean up afterwards
    os.system('rm -rf %s' % tmp_dir)

    # Just a check to see what's in the scratch directory
    lsres = subprocess.Popen('ls %s' % scratch_dir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info('Contents of scratch directory: %s' % lsres.communicate()[0])

if __name__ == "__main__" : 
    main()
