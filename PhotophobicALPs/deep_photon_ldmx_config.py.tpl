from LDMX.Framework import ldmxcfg
p=ldmxcfg.Process("v15_deepPhotonFromTarget")

from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions as ecal_conditions
import LDMX.Hcal.hcal_hardcoded_conditions as hcal_conditions
from LDMX.Biasing import ecal
from LDMX.SimCore import generators

from LDMX.Tracking import full_tracking_sequence

det = 'ldmx-det-v15-8gev'
mysim = ecal.deep_photo_nuclear(det, generators.single_8gev_e_upstream_tagger(), bias_threshold = 5010., processes = ['conv'], ecal_min_Z = {{ decay_length }}, require_photon_fromTarget = True)
#mysim = ecal.deep_photo_nuclear(det, generators.single_8gev_e_upstream_tagger(), bias_threshold = 5010., processes=['conv','phot)'], ecal_min_Z = 200., require_photon_fromTarget = True)
mysim.description = "ECal Deep Conversion Test Simulation"

from LDMX.Biasing import util
#mysim.actions.append( util.StepPrinter(1) )
#step = util.StepPrinter(track_id=1, depth=3)
#mysim.actions.extend([step])

from LDMX.Ecal import digi as eDigi
from LDMX.Ecal import vetos
from LDMX.Hcal import digi as hDigi
from LDMX.Hcal import hcal

from LDMX.Recon.simpleTrigger import TriggerProcessor

from LDMX.TrigScint.trigScint import TrigScintDigiProducer
from LDMX.TrigScint.trigScint import TrigScintClusterProducer
from LDMX.TrigScint.trigScint import trigScintTrack

# Ecal Digi Chain
ecalDigi   = eDigi.EcalDigiProducer('EcalDigis')
ecalReco   = eDigi.EcalRecProducer('ecalRecon')
ecalVeto   = vetos.EcalVetoProcessor('ecalVetoBDT')

# Hcal Digi Chain
hcalDigi   = hDigi.HcalDigiProducer('hcalDigis')
hcalReco   = hDigi.HcalRecProducer('hcalRecon')                  
hcalVeto   = hcal.HcalVetoProcessor('hcalVeto')

# TS Digi + Clustering + Track Chain
tsDigisUp    = TrigScintDigiProducer.pad1()
tsDigisTag   = TrigScintDigiProducer.pad2()
tsDigisDown  = TrigScintDigiProducer.pad3()

tsClustersUp    = TrigScintClusterProducer.pad1()
tsClustersTag   = TrigScintClusterProducer.pad2()
tsClustersDown  = TrigScintClusterProducer.pad3()


p.outputFiles = ['{{ root_file_name }}']

p.maxTriesPerEvent = 1_000_000
p.maxEvents = {{ n_events }}
p.run = {{ seed }}
p.logFrequency = 1000
p.termLogLevel = 2

p.sequence=[ mysim,
        ecalDigi, ecalReco, 
        hcalDigi, hcalReco]

p.sequence.extend(full_tracking_sequence.sequence)

p.sequence.extend([ecalVeto, hcalVeto])

layers = [17, 20]
tList = []
for iLayer in range(len(layers)) :
    tp = TriggerProcessor("TriggerSumsLayer"+str(layers[iLayer]), 8000.)
    tp.start_layer = 0
    tp.end_layer = layers[iLayer]
    tp.trigger_collection = "TriggerSums"+str(layers[iLayer])+"Layers"
    tList.append(tp)
p.sequence.extend( tList ) 

