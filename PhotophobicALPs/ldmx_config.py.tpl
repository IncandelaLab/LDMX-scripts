from LDMX.Framework import ldmxcfg

nElectrons = 1
passName = 'alp'
p = ldmxcfg.Process(passName)
p.max_tries_per_event = 1
p.max_events = {{ n_events }}
p.logger.term_level = 2
p.log_frequency = 1000

# Set run parameters
p.run = {{ seed }}
beamEnergy = 8  #in GeV   

# Import all Processors
from LDMX.SimCore import generators
from LDMX.SimCore import simulator
from LDMX.Tracking import full_tracking_sequence

# Instantiate the simulator.
sim = simulator.simulator('sim')
detector = 'ldmx-det-v15-8gev'
sim.setDetector( detector , include_scoring_planes_minimal = True )
sim.description = 'ALP with ' + str(beamEnergy)+ ' GeV electron events'
sim.beamSpotSmear = [0., 0., 0.]
sim.time_shift_primaries = True

# Set ALP and decay generator
ALP_gen = generators.lhe('ALP Generator', '{{ prod_file }}')
decay_gen = generators.lhe('Decay Generator', '{{ decay_file }}')
sim.generators = [ALP_gen, decay_gen]

import LDMX.Ecal.ecal_hardcoded_conditions
from LDMX.Ecal import ecal_geometry

import LDMX.Hcal.hcal_hardcoded_conditions
import LDMX.Hcal.hcal_geometry

from LDMX.Ecal import digi as ecal_digi_reco
from LDMX.Ecal import vetos as ecal_vetos
from LDMX.Hcal import digi as hDigi
from LDMX.Hcal import hcal

from LDMX.Recon.simple_trigger import TriggerProcessor

from LDMX.TrigScint.trig_scint import TrigScintDigiProducer
from LDMX.TrigScint.trig_scint import TrigScintClusterProducer
from LDMX.TrigScint.trig_scint import trig_scint_track

# Ecal Digi Chain
ecal_digi   = ecal_digi_reco.EcalDigiProducer('ecal_digis')
ecal_reco   = ecal_digi_reco.EcalRecProducer('ecal_recon')
ecal_veto   = ecal_vetos.EcalVetoProcessor('ecal_veto')

# Hcal Digi Chain
hcal_digi   = hDigi.HcalDigiProducer('hcal_digis')
hcal_reco   = hDigi.HcalRecProducer('hcal_recon')                  
hcal_veto   = hcal.HcalVetoProcessor('hcal_veto')

tsDigisUp    = TrigScintDigiProducer.pad1()
tsDigisTag   = TrigScintDigiProducer.pad2()
tsDigisDown  = TrigScintDigiProducer.pad3()

tsClustersUp    = TrigScintClusterProducer.pad1()
tsClustersTag   = TrigScintClusterProducer.pad2()
tsClustersDown  = TrigScintClusterProducer.pad3()

p.sequence=[ sim,
        ecal_digi, ecal_reco,
        hcal_digi, hcal_reco]

p.sequence.extend(full_tracking_sequence.sequence)

p.sequence.extend([ecal_veto, hcal_veto])

#tp = TriggerProcessor("TriggerSumsLayer20", 8000.)
#tp.start_layer = 0
#tp.end_layer = 20
#tp.trigger_collection = "TriggerSums20Layers"
#p.sequence.append(tp)

p.output_files = [ '{{ root_file }}' ]
