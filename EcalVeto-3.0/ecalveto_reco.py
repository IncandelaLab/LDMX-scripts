import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--indir', action='store', dest='indir', help='Directory of input file, absolute path')
parser.add_argument('-o','--outdir', action='store', dest='outdir', help='Directory of output files, absolute path')
parser.add_argument('-m','--maxEvent', action='store', dest='maxEvent', help='Maximum number of events to process')
args = parser.parse_args()

from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('SegmipBDTReco')

p.maxTriesPerEvent = 10000

p.maxEvents = args.maxEvent

p.inputFiles = [args.indir]

p.outputFiles = [args.outdir]
p.termLogLevel = 0

# EcalVeto
import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos

p.sequence = [ecal_vetos.EcalVetoProcessor()]
