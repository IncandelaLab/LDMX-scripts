#!/usr/bin/env python

import argparse
import glob
import logging
import os
import platform
import random
import subprocess
import time
import shutil

from string import Template

#####################################################
# Write Python Config Script for This Analysis
# Inputs
#   template_config ==> config template file
#   prefix          ==> prefix to attach to front of output files
#   job_id          ==> id number assigned to this job
#   input_file      ==> root file to be processed or lhe file to simulate over (in working directory)
# Outputs
#   config_path     ==> path to python config file to run this analysis
#   output_events   ==> output event root file (name and extension)
#   output_hists    ==> output histogram/tree root file (name and extension)
#
# The template_config is opened with the template python library.
# For a simulation/production run the following keys are searched for:
#   lheFile, outputEventFile, histogramFile, run, seed1, seed2
# For a analysis/recon run the following keys are searched for:
#   inputEventFile, outputEventFile, histogramFile
# To have a key substituted with its value, you should insert it in the python config like:
#   $key_name
def write_config(template_config, run, seed1, seed2, prefix, job_id, input_file ):
    
    job = int(job_id)

    filename = ''
    total_prefix = prefix + '_'
    if ( input_file == "" ) :
        # no input file or input file is lhe file ==> production/simulation run
        total_prefix += str(run)
    elif input_file.endswith('.lhe') :
        # input lhe file ==> lhe sim run
        filename = input_file[input_file.rfind('/') + 1: ]
        total_prefix += str(job_id) + '_' + filename[:-4] #remove .lhe
    elif input_file.endswith('.root') :
        # input root file ==> analysis/recon run
        filename = input_file[input_file.rfind('/') + 1: ]
        total_prefix += str(job_id) + '_' + filename[:-5] #remove .root
    else :
        logging.warning( 'Unknown configuration! The file \'%s\' was input, but it is not a lhe or root file.' % ( input_file ) )
        exit()

    config_path   = '%s_config.py'   % total_prefix
    output_events = '%s_events.root' % total_prefix
    output_hists  = '%s_hists.root'  % total_prefix

    logging.info('Writing configuration to %s' % config_path)
    logging.info('Saving event output to %s' % output_events )
    logging.info('Saving histogram/tree output to %s' % output_hists )

    with open(template_config) as configTemplateFile :
        configTemplate = Template( configTemplateFile.read() )
        # Template quietly skips keys that aren't in template
        outputConfig = configTemplate.substitute(
                inputEventFile  = filename,
                lheFile         = filename,
                outputEventFile = output_events,
                histogramFile   = output_hists,
                run             = run,
                seed1           = seed1,
                seed2           = seed2
        )
   
    with open(config_path, 'w') as fh:
        fh.write(outputConfig)

    return config_path, output_events, output_hists

###################################################################
#   run_ana main
#   Runs fire with inputs substituted into input template.
#   Outputs are copied to input output directories.
#   Inputs:
#       config      (required) full path to template config file to run
#       --test      (optional) should we actually run fire or just print out the command we would have run?
#       --prefix    (optional) prefix to attach to front of output files
#       --inputFile (optional) full path to root file to process or lhe file to simulate over
#       --eventOut  (optional) full path to directory to copy event output file to - must already exist
#       --histOut   (optional) full path to directory to copy histogram/tree output file to - must already exist
#       --envScript (optional) full path to environment script to run before fire (default is for Centos7+cvmfs)
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', 
            action='store_true',
            dest='test',
            help='Don\'t run fire.')
    parser.add_argument('--prefix', 
            default='',
            help='Output prefix to name output files.')
    parser.add_argument('--inputFile', 
            default='',
            help='Input file to process.')
    parser.add_argument('--eventOut', 
            default='',
            help='Path to save event output to.')
    parser.add_argument('--histOut', 
            default='',
            help='Path to save histogram/tree output to.')
    parser.add_argument('config', 
            type=str,
            help='Config template to use.')
    parser.add_argument('-r', '--run',
            help='The run number.') 
    parser.add_argument('-s', '--seed',
            help='The seed number.') 
    parser.add_argument('--detectorPath', 
            default='',
            help='Path to directory with detector files which will be symlinked to the working directory.')
    args = parser.parse_args() 

    # Configure the logger
    logging.basicConfig(format='[ run ][ %(levelname)s ]: %(message)s', 
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
            elif os.path.islink(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
            else:
                logging.info('%s is not a file or directory' % f)

    # Create a link to the config template
    os.symlink(args.config.strip(), 'config_template.py')

    # If an input file has been specified, copy it over.
    if args.inputFile:
        logging.info('Copying input file %s' % args.inputFile.strip())
        os.system('cp %s .' % args.inputFile.strip())

    # Get job id
    if 'SLURM_JOBID' in os.environ: 
        jobid = os.environ['SLURM_JOBID']
    else: jobid = 100

    # Write config file replacing template parameters
    config_path, eventsf, histsf = write_config('config_template.py', run, seed1, seed2, args.prefix, jobid, args.inputFile)

    # Run the config
    print 'Using production image %s' % os.environ.get('LDMX_PRODUCTION_IMG')
    command = "singularity run --no-home %s . %s" % (os.environ.get('LDMX_PRODUCTION_IMG'),config_path) 

    if args.test :
        logging.info( command )
    else :
        subprocess.Popen(command, shell=True).wait()

    # Copy output files to output directory
    if args.eventOut :
        if os.path.isfile( eventsf ) :
            eventOutputDir = args.eventOut.strip()
            logging.info('Copying event file to %s' % eventOutputDir)
            if not args.test :
                if not os.path.exists(eventOutputDir):
                    logging.info('Event output directory does not exist and will be created')
                    os.makedirs(eventOutputDir)
                logging.info('cp %s %s' % ( eventsf , os.path.join( eventOutputDir , eventsf ) ) )
                os.system('cp %s %s' % ( eventsf , os.path.join( eventOutputDir , eventsf ) ) )
        else :
            logging.info('No event file to copy.')
    else :
        logging.info( 'Not saving events file (if it exists).' )

    if args.histOut :
        if os.path.isfile( histsf ) :
            histOutputDir = args.histOut.strip()
            logging.info('Copying hist/tree file to %s' % histOutputDir)
            if not args.test :
                if not os.path.exists(histOutputDir):
                    logging.info('Histogram/tree output directory does not exist and will be created')
                    os.makedirs(histOutputDir)
                logging.info('cp %s %s' % ( histsf , os.path.join( histOutputDir , histsf ) ) )
                os.system('cp %s %s' % ( histsf , os.path.join( histOutputDir , histsf ) ) )
        else :
            logging.info('No hist/tree file to copy.')
    else :
        logging.info( 'Not saving histogram/tree file (if it exists).' )

    # Clean up afterwards
    if not args.test:
        os.system('rm -r %s' % tmp_dir)
    lsres = subprocess.Popen('ls %s' % scratch_dir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info('Contents of scratch directory: %s' % lsres.communicate()[0])

#if this python script is main, run it
if __name__ == "__main__" : 
    main()
