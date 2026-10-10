import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', nargs='+', dest='infile')
parser.add_argument('-o', '--outdir', dest='outdir')
parser.add_argument('-m', '--maxEvent', type=int, dest='maxEvent')
parser.add_argument('-g', '--group_label', dest='group_label')
parser.add_argument('-r', '--run_number', type=int, dest='run_number')
parser.add_argument('--bdt', dest='bdt', default=None)
args = parser.parse_args()

from LDMX.Framework import ldmxcfg

PASS = args.group_label if args.group_label else 'prerun'
p = ldmxcfg.Process(PASS)
p.maxEvents = args.maxEvent
p.skipCorruptedInputFiles = True
p.termLogLevel = 4

inputFiles = [f for x in args.infile for f in x.replace(',', ' ').split()]
base = os.path.splitext(os.path.basename(inputFiles[0]))[0]
os.makedirs(args.outdir, exist_ok=True)
p.inputFiles = inputFiles
p.outputFiles = [os.path.join(args.outdir, f'{base}_{PASS}.root')]

import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos
from LDMX.Recon.simpleTrigger import simpleTrigger 
from LDMX.Recon.ecalPreselectionSkimmer import EcalPreselectionSkimmer
from LDMX.Tracking import tracking

ecalVeto = ecal_vetos.EcalVetoProcessor()
if args.bdt:        
    from LDMX.Ecal.makePath import makeBDTPath
    ecalVeto.bdt_file = makeBDTPath(args.bdt)

ecalMip = ecal_vetos.EcalMipProcessor()
ecalMip.ecal_pass_name = PASS   
ecalMip.mip_pass_name = PASS      

presel = EcalPreselectionSkimmer()
presel.use_rechits = False    
presel.ecal_veto_pass = PASS     
presel.ecal_mip_pass = PASS

trackerVeto = tracking.TrackerVetoProcessor()

p.sequence = [simpleTrigger, ecalVeto, ecalMip, presel, trackerVeto]
