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
p.max_events = args.maxEvent
p.skip_corrupted_input_files = True

inputFiles = [f for x in args.infile for f in x.replace(',', ' ').split()]
base = os.path.splitext(os.path.basename(inputFiles[0]))[0]
os.makedirs(args.outdir, exist_ok=True)
p.input_files = inputFiles
p.output_files = [os.path.join(args.outdir, f'{base}_{PASS}.root')]

import LDMX.Ecal.ecal_geometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos
from LDMX.Recon.simple_trigger import simple_trigger 
from LDMX.Recon.ecal_preselection_skimmer import EcalPreselectionSkimmer
from LDMX.Recon.fiducial_flag import RecoilFiducialityProcessor
from LDMX.Tracking import tracking

ecalVeto = ecal_vetos.EcalVetoProcessor()
if args.bdt:        
    from LDMX.Ecal.make_path import make_bdt_path
    ecalVeto.bdt_file = make_bdt_path(args.bdt)

ecalMip = ecal_vetos.EcalMipProcessor()
ecalMip.ecal_pass_name = PASS   
ecalMip.mip_pass_name = PASS      

presel = EcalPreselectionSkimmer()
presel.use_rechits = False    
presel.ecal_veto_pass = PASS     
presel.ecal_mip_pass = PASS

trackerVeto = tracking.TrackerVetoProcessor()

# Truth-level recoil fiduciality -> RecoilTruthFiducialFlags.
# NOTE this is NOT the same quantity as EcalVetoResult.fiducial_, which is derived
# from the recoil position at the ECal face and is what the cutflow's Fiducial row
# uses. This processor reads Ecal/Hcal/Recoil SIM HITS, so it cannot run on
# dropsim samples that have had sim hits removed; drop it from the sequence there.
fiducial = RecoilFiducialityProcessor()

# Order matters: preselection reads EcalVeto + EcalMipInfo from this pass.
p.sequence = [simple_trigger, ecalVeto, ecalMip, presel, trackerVeto, fiducial]
