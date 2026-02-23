from LDMX.Framework import ldmxcfg
p=ldmxcfg.Process("v15_deepPhotonFromTarget")

#from LDMX.Ecal import EcalGeometry
from LDMX.Ecal import ecal_geometry
#from LDMX.Hcal import HcalGeometry
import LDMX.Hcal.hcal_geometry
import LDMX.Ecal.ecal_hardcoded_conditions as ecal_conditions
import LDMX.Hcal.hcal_hardcoded_conditions as hcal_conditions
from LDMX.Biasing import ecal
from LDMX.SimCore import generators

from LDMX.Tracking import full_tracking_sequence

det = 'ldmx-det-v15-8gev'
mysim = ecal.deep_photo_nuclear(det, generators.single_8gev_e_upstream_tagger(), bias_threshold = 5010., processes=['conv'], ecal_min_z = {{ decay_length }}, require_photon_from_target = True)
#mysim = ecal.deep_photo_nuclear(det, generators.single_8gev_e_upstream_tagger(), bias_threshold = 5010., processes=['conv','phot)'], ecal_min_Z = 200., require_photon_fromTarget = True)
mysim.description = "ECal Deep Conversion Test Simulation"

from LDMX.Biasing import util
#mysim.actions.append( util.StepPrinter(1) )
#step = util.StepPrinter(track_id=1, depth=3)
#mysim.actions.extend([step])

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

# TS Digi + Clustering + Track Chain
tsDigisUp    = TrigScintDigiProducer.pad1()
tsDigisTag   = TrigScintDigiProducer.pad2()
tsDigisDown  = TrigScintDigiProducer.pad3()

tsClustersUp    = TrigScintClusterProducer.pad1()
tsClustersTag   = TrigScintClusterProducer.pad2()
tsClustersDown  = TrigScintClusterProducer.pad3()


p.output_files = ['{{ root_file_name }}']

p.max_tries_per_event = 1_000_000
p.max_events = {{ n_events }}
p.run = {{ seed }}
p.log_frequency = 1000
p.logger.term_level = 1

p.sequence=[ mysim,
        ecal_digi, ecal_reco, 
        hcal_digi, hcal_reco]

p.sequence.extend(full_tracking_sequence.sequence)

p.sequence.extend([ecal_veto, hcal_veto])

layers = [17, 20]
tList = []
for iLayer in range(len(layers)) :
    tp = TriggerProcessor("TriggerSumsLayer"+str(layers[iLayer]), 8000.)
    tp.start_layer = 0
    tp.end_layer = layers[iLayer]
    tp.trigger_collection = "TriggerSums"+str(layers[iLayer])+"Layers"
    tList.append(tp)
p.sequence.extend( tList ) 

