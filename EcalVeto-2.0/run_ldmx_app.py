#!/usr/bin/env python

import argparse
import glob
import logging
import os
import platform
import random
import subprocess
import time

from string import Template

###############################################
# Calculate unique integer from two integers
def cantor_pair(a, b): 
    return int(0.5*(a + b)*(a + b + 1) + b)

#####################################################
# Write Python Config Script for This Analysis
# Inputs
#   template_config ==> config template file
#   outPrefix       ==> prefix to attach to front of output files
#   job_id          ==> id number assigned to this job (if job_id <= 0, then no random seeds are generated ==> assumes analysis process instead of production)
#   inputFile       ==> root file to be processed or lhe file to simulate over (in working directory)
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
def write_config( template_config, outPrefix, job_id, inputFile ):
    
    job = int(job_id)

    run = '1'
    seed1 = 1234
    seed2 = 5678

    totalPrefix = outPrefix + '_'
    if ( inputFile == "" ) :
        #no input file or input file is lhe file ==> production/simulation run
        random.seed(time.time())
        seed1 = int(random.random()*100000000)
        seed2 = int(random.random()*100000000)
        
        seed1 = int(str(cantor_pair(seed1, job))[:6])
        seed2 = int(str(cantor_pair(seed2, job))[:6])
        
        logging.info('Using seeds %s %s' % (seed1, seed2))
    
        run = '%s%s' % (seed1, seed2)
        logging.info('Setting run number to %s' % run)
        
        totalPrefix += run
    elif inputFile.endswith('.lhe') :
        # input lhe file ==> lhe sim run
        filename = inputFile[inputFile.rfind('/') + 1: ]
        totalPrefix += str(job_id) + '_' + filename[:-4] #remove .lhe
    elif inputFile.endswith('.root') :
        # input root file ==> analysis/recon run
        filename = inputFile[inputFile.rfind('/') + 1: ]
        totalPrefix += str(job_id) + '_' + filename[:-5] #remove .root
    else :
        logging.warning( 'Unknown configuration! The file \'%s\' was input, but it is not a lhe or root file.' % ( inputFile ) )
        exit()
    #end if production or analysis job

    config_path   = '%s_config.py'   % totalPrefix
    output_events = '%s_events.root' % totalPrefix
    output_hists  = '%s_hists.root'  % totalPrefix

    logging.info('Writing configuration to %s' % config_path)
    logging.info('Saving event output to %s' % output_events )
    logging.info('Saving histogram/tree output to %s' % output_hists )

    with open(template_config) as configTemplateFile :
        configTemplate = Template( configTemplateFile.read() )
        #Template quietly skips keys that aren't in template
        outputConfig = configTemplate.substitute(
                inputEventFile  = inputFile,
                lheFile         = inputFile,
                outputEventFile = output_events,
                histogramFile   = output_hists,
                run             = run,
                seed1           = seed1,
                seed2           = seed2
        )
    #end open config template
   
    with open(config_path, 'w') as fh:
        fh.write(outputConfig)
    #end writing config

    return config_path, output_events, output_hists

###################################################################
#   run_ldmx_app main
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
    parser.add_argument('--envScript', 
            default='/nfs/slac/g/ldmx/users/$USER/local_setup_gcc8.3.1_cos7.sh', #TODO make default env script distribution dependent
            help='Environment script to run before running fire.')
    parser.add_argument('--detectorPath', 
            default='',
            help='Path to directory with detector files which will be symlinked to the working directory.')
    args = parser.parse_args() 

    # Configure the logger
    logging.basicConfig(format='[ run ][ %(levelname)s ]: %(message)s', 
            level=logging.DEBUG)

    # Setup all environmental variables.
    #command = ['bash', '-c', 'source %s && env' % (os.path.realpath(args.envScript))]
    #proc = subprocess.Popen(command, stdout=subprocess.PIPE)

    #for line in proc.stdout: 
    #    (key, _, value) = line.partition('=')
    #    os.environ[key] = value.strip()

    #proc.communicate()
 
    # First, create a user directory within scratch or locally
    if os.access('/scratch/', os.W_OK): 
        scratch_dir = '/scratch/%s' % os.environ['USER']
    else:
        scratch_dir = '%s/%s' % (os.getcwd(),os.environ['USER'])
    logging.info('Using scratch path %s' % scratch_dir)
    if not os.path.exists(scratch_dir):
        logging.info('Scratch directory does not exist and will be created.')
        os.makedirs(scratch_dir)

    # Now, create a directory within the user directory in scratch
    if 'LSB_JOBID' in os.environ:  
        tmp_dir = '%s/%s' % (scratch_dir, os.environ['LSB_JOBID'])
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
            os.remove(f)

    # Create a link to the config template
    os.symlink(args.config.strip(), 'config_template.py')

    # If an input file has been specified, copy it over.
    if args.inputFile:
        logging.info('Copying input w file %s' % args.inputFile.strip())
        os.system('cp %s .' % args.inputFile.strip())

    if args.detectorPath:
        fullPath = os.path.realpath( args.detectorPath )
        for fn in os.listdir( fullPath ) :
            logging.info('Sym-linking detector file %s' % fn )
            os.symlink( os.path.join( fullPath , fn ) , fn )
        #end loop over detector files
    #end if detector directory provided

    if 'LSB_JOBID' in os.environ: 
        jobid = os.environ['LSB_JOBID']
    else: jobid = 100

    #write config steering file
    config_path, eventsf, histsf = write_config('config_template.py', args.prefix, jobid, args.inputFile)
    #run fire and wait for it to finish
    command = "fire %s" % config_path
    if args.test :
        logging.info( command )
    else :
        subprocess.Popen(command, shell=True).wait()

    #copy outputs to output directory after fire is done
    if args.eventOut :
        if os.path.isfile( eventsf ) :
            eventOutputDir = args.eventOut.strip()
            logging.info('Copying event file to %s' % eventOutputDir)
            if not args.test :
                if not os.path.exists(eventOutputDir):
                    logging.info('event output directory does not exist and will be created')
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
                    logging.info('histogram/tree output directory does not exist and will be created')
                    os.makedirs(histOutputDir)
                logging.info('cp %s %s' % ( histsf , os.path.join( histOutputDir , histsf ) ) )
                os.system('cp %s %s' % ( histsf , os.path.join( histOutputDir , histsf ) ) )
        else :
            logging.info('No hist/tree file to copy.')
    else :
        logging.info( 'Not saving histogram/tree file (if it exists).' )

    if not args.test:
        os.system('rm -r %s' % tmp_dir)

#if this python script is main, run it
if __name__ == "__main__" : 
    main()
