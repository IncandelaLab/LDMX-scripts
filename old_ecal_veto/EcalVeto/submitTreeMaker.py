import os
import sys
from ConfigParser import SafeConfigParser
from collections import OrderedDict

# Set up and run using the config file provided
# Will create a shell script that can be run to submit the jobs
def main():
    # Default conf file, if running with a different one just provide it as an argument
    configFile = 'process.conf'
    args = sys.argv[1:]
    if len(args) >= 1:
        configFile = args[0]
    if os.path.exists(configFile):
        print 'running with config file', configFile
    else:
        print 'you are trying to use a config file (' + configFile + ') that does not exist!'
        sys.exit(1)

    # Parse the conf file
    configparser = SafeConfigParser(dict_type=OrderedDict)
    configparser.optionxform = str

    # Create jobs and prepare the submit script
    config = JobConfig(configFile, configparser)
    jobs = Jobs(config)
    jobs.submit()


# Class to parse the config file
class JobConfig:
    def __init__(self, conf_file, config_parser):
        self.conf_file = conf_file
        config_parser.read(self.conf_file)
        self.sigdirs = config_parser.get('config','signal_dir').replace(' ', '').split(',')
        self.bkgdirs = config_parser.get('config','background_dir').replace(' ', '').split(',')
        self.sigsuffix = config_parser.get('config','signal_suffix')
        self.bkgsuffix = config_parser.get('config','background_suffix')
        self.sigprefix = config_parser.get('config','signal_prefix')
        self.bkgprefix = config_parser.get('config','background_prefix')
        self.sigpoints = config_parser.get('signals','masspoints').replace(' ', '').split(',')
        self.runscript = config_parser.get('config','run_script')
        self.outdir = config_parser.get('config','output_dir')
        self.jobdir = config_parser.get('config','job_dir')
        self.outlabel = config_parser.get('config','output_label')
        self.filesperjob = config_parser.get('config','files_per_job')
        self.queuetime = config_parser.get('config','queue_time')
        self.submitfile = config_parser.get('config','submit_file')

# Class to create jobs processing all the files we want to run on
class Jobs:
    def __init__(self,config):
        self.config = config

        # Compile list of input files to run
        self.files = {}
        bkgprefix = self.bkgprefix if self.bkgprefix != 'None' else ''
        if self.bkgdirs[0] != 'dummy':
            self.files['bkg'] = [os.path.join(self.bkgdirs[0],f) for f in os.listdir(self.bkgdirs[0]) if f.startswith(bkgprefix) and f.endswith(self.bkgsuffix)]
            for idir in range(1,len(self.bkgdirs)):
                self.files['bkg'].extend([os.path.join(self.bkgdirs[idir],f) for f in os.listdir(self.bkgdirs[idir]) if f.startswith(bkgprefix) and f.endswith(self.bkgsuffix)])
        sigprefix = self.sigprefix if self.sigprefix != 'None' else ''
        isig = 0
        for s in self.sigpoints:
            if s != 'None':
                if len(self.sigdirs) == 1:
                    sigdir = self.sigdirs[0]
                    self.files[s] = [os.path.join(sigdir,f) for f in os.listdir(sigdir) if f.startswith(sigprefix) and s in f and f.endswith(self.sigsuffix)]
                else:
                    self.files[s] = [os.path.join(self.sigdirs[isig],f) for f in os.listdir(self.sigdirs[isig]) if f.startswith(sigprefix) and s in f and f.endswith(self.sigsuffix)]
                isig += 1

        self.jobcfgs = {}

        # Create the job configs
        for sample,files in self.files.iteritems():
            # Create one job per nfiles per sample
            nfiles = len(files)
            numjobs = int(nfiles)/int(self.filesperjob)
            rem = nfiles % int(self.filesperjob)

            if rem == 0: numjobs -= 1

            for ijob in range(numjobs+1):
                start = (ijob * int(self.filesperjob))
                end = (int(self.filesperjob) * (ijob + 1)) # -1
                print ijob,start,end
                # List of files for this job
                jobfiles = files[start:end]
                filelist = '%s/filelist_%s_%d.txt' % (self.jobdir,sample,ijob)
                if not os.path.exists(self.jobdir):
                    os.makedirs(self.jobdir)
                with open(filelist,'w') as f:
                    f.write('\n'.join(jobfiles))
                # Output file name for this job
                outfile = '_'.join([sample,self.outlabel,str(ijob)+'.root'])
                # Create the job config and add it to the list for this sample
                job = Job('signal' if sample in self.sigpoints else 'background',filelist,self.outdir,outfile)
                if self.jobcfgs.has_key(sample):
                    self.jobcfgs[sample].append(job)
                else:
                    self.jobcfgs[sample] = [job]

    # Create a submit script with all jobs
    def submit(self):
        with open(self.submitfile, 'w') as f:
            for sample,jobs in self.jobcfgs.iteritems():
                f.write('#'+sample+'\n')
                for job in jobs:
                    submitcmd = 'bsub -R rhel60 -W %s python %s --outdir %s --outfile %s --filelist %s' % (self.queuetime,self.runscript,job.outputdir,job.outputfile,job.filelist)
                    if sample in self.sigpoints:
                       submitcmd += ' --signal'
                    f.write(submitcmd + '\n')

    # Makes life easier
    def __getattr__(self, name):
        # Look up attributes in self.config as well.
        return getattr(self.config, name)


# Class to hold configuration for a job
class Job:
    def __init__(self,datatype,filelist,outputdir,outputfile):
        # datatype is signal or background
        self.datatype = datatype
        self.filelist = filelist
        self.outputdir = outputdir
        self.outputfile = outputfile


if __name__ == '__main__': main()
