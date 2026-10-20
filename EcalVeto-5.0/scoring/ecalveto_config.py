import argparse
from ast import arg
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--infile', action='store', dest='infile', nargs='+', help='List of input files followed by run number, in a single string')
parser.add_argument('-o','--outdir', action='store', dest='outdir', help='Directory of output file, absolute path')
parser.add_argument('-m','--maxEvent', type=int, action='store', dest='maxEvent', help='Maximum number of events to process')
parser.add_argument('-g','--group_label', action='store', dest='group_label', help='Sample label')
parser.add_argument('-r','--run_number', type=int, action='store', dest='run_number', help='Run number')
parser.add_argument('--bdt', action='store', dest='bdt', default=None,
                    help='BDT name for make_bdt_path (default: whatever vetos.py sets)')
args = parser.parse_args()
 
### ------------------LDMX CONFIG----------------- ###

from LDMX.Framework import ldmxcfg
import os
import sys

p = ldmxcfg.Process(args.group_label if args.group_label else 'Training_BDT')
p.max_events = args.maxEvent


# Parse input string: "file1,file2,...fileN,runNumber" or "file1 file2 ... runNumber"
raw_input = args.infile
input_args = []
for arg in raw_input:
    # Replace commas with spaces to handle both formats, then split
    cleaned_arg = arg.replace(',', ' ')
    input_args.extend(cleaned_arg.split())

runNumber = args.run_number # Currently not using
inputFiles = input_args

for f in inputFiles:
    print(f"  - {f}")

# Derive output name from input file, append group label
inputBasename = os.path.splitext(os.path.basename(inputFiles[0]))[0]
outputFilename = f'{inputBasename}_{args.group_label}.root'

os.makedirs(args.outdir, exist_ok=True)

outputName = os.path.join(args.outdir, outputFilename)

p.input_files = inputFiles
p.output_files = [outputName]


### ------------------------------------------ ###

p.sequence = []


# ECAL Veto only
import LDMX.Ecal.ecal_geometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos
ecalVeto = ecal_vetos.EcalVetoProcessor()

# False = SP , True = Digi(recoil).  
ecalVeto.recoil_from_tracking = True

if args.bdt:
    from LDMX.Ecal.make_path import make_bdt_path
    ecalVeto.bdt_file = make_bdt_path(args.bdt)
    print(f'BDT manual input: {ecalVeto.bdt_file}')

# p.logger.custom(ecalVeto, level = 0)
p.skip_corrupted_input_files = True

p.sequence.extend([ecalVeto])

