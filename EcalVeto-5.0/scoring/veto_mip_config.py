import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--infile', action='store', dest='infile', nargs='+')
parser.add_argument('-o','--outdir', action='store', dest='outdir')
parser.add_argument('-m','--maxEvent', type=int, action='store', dest='maxEvent')
parser.add_argument('-g','--group_label', action='store', dest='group_label')
parser.add_argument('-r','--run_number', type=int, action='store', dest='run_number')
args = parser.parse_args()

from LDMX.Framework import ldmxcfg

# EcalVeto + MIP tracking in one pass.

PASS = "mip_reco_v1"
p = ldmxcfg.Process(PASS)
p.max_events = args.maxEvent

input_args = []
for a in args.infile:
    input_args.extend(a.replace(',', ' ').split())
inputFiles = input_args
base = os.path.splitext(os.path.basename(inputFiles[0]))[0]
os.makedirs(args.outdir, exist_ok=True)
p.input_files = inputFiles
p.output_files = [os.path.join(args.outdir, f'{base}_{args.group_label}.root')]

p.sequence = []

import LDMX.Ecal.ecal_geometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos

ecalVeto = ecal_vetos.EcalVetoProcessor()
ecalVeto.recoil_from_tracking = True # Use recoil tracking

ecalMip = ecal_vetos.EcalMipProcessor()

# Read THIS pass's freshly produced objects.
ecalMip.ecal_pass_name = PASS
ecalMip.mip_pass_name  = PASS


p.skip_corrupted_input_files = True
p.sequence.extend([ecalVeto, ecalMip])
