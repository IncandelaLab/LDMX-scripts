import argparse
from ast import arg
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--infile', action='store', dest='infile', help='Directory of input file, absolute path')
parser.add_argument('-o','--outdir', action='store', dest='outdir', help='Directory of output file, absolute path')
parser.add_argument('-m','--maxEvent', type=int, action='store', dest='maxEvent', help='Maximum number of events to process')
parser.add_argument('-g','--groupl', action='store', dest='group_label', help='Sample label')
args = parser.parse_args()

from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('SegmipBDTReco')

p.maxTriesPerEvent = 10000

p.maxEvents = args.maxEvent

p.inputFiles = [args.infile]

outfile = os.path.join(args.outdir, args.group_label + '.root')
p.outputFiles = [outfile]
p.termLogLevel = 0

# EcalVeto
import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos

p.sequence = [ecal_vetos.EcalVetoProcessor()]
