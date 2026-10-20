import argparse
from ast import arg
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--infile', action='store', dest='infile', nargs='+', help='List of input files followed by run number, in a single string')
parser.add_argument('-o','--outdir', action='store', dest='outdir', help='Directory of output file, absolute path')
parser.add_argument('-m','--maxEvent', type=int, action='store', dest='maxEvent', help='Maximum number of events to process')
parser.add_argument('-g','--group_label', action='store', dest='group_label', help='Sample label')
parser.add_argument('-r','--run_number', type=int, action='store', dest='run_number', help='Run number')
args = parser.parse_args()
 
### ------------------LDMX CONFIG----------------- ###

from LDMX.Framework import ldmxcfg
import os
import sys
p = ldmxcfg.Process('reco_v1')

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
from LDMX.Tracking import full_tracking_sequence
# These samples are v4.5.4 (v14 detector). full_tracking_sequence defaults the tracking
# geometry to v15, which segfaults the Acts propagator on v14 hits. Override to v14.
from LDMX.Tracking.geo import TrackersTrackingGeometryProvider as trackgeo
trackgeo.get_instance().set_detector('ldmx-det-v14-8gev')


# ECAL Veto only
import LDMX.Ecal.ecal_geometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos
ecalVeto = ecal_vetos.EcalVetoProcessor()

# Load the HCAL modules
import LDMX.Hcal.hcal_geometry
import LDMX.Hcal.hcal_hardcoded_conditions
import LDMX.Hcal.digi as hcal_digi
hcal_digi_reco = hcal_digi.HcalSimpleDigiAndRecProducer()

# Hcal Veto
import LDMX.Hcal.hcal as hcal
hcal_veto = hcal.HcalVetoProcessor()

# p.logger.custom(ecalVeto, level = 0)
p.skip_corrupted_input_files = True

p.sequence.extend(full_tracking_sequence.sequence)

p.sequence.extend([ecalVeto,
    hcal_digi_reco,
    hcal_veto,
    ])
