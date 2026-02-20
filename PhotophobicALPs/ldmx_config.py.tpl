from LDMX.Framework import ldmxcfg

nElectrons = 1
passName = 'alp'
p = ldmxcfg.Process(passName)
p.maxTriesPerEvent = 1
p.maxEvents = {{ n_events }}
p.termLogLevel = 2
p.logFrequency = 1000

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
sim.time_shift_primaries = False

# Set ALP and decay generator
ALP_gen = generators.lhe('ALP Generator', '{{ prod_file }}')
ALP_gen.vertex = [ 0., 0., 0. ]
decay_gen = generators.lhe('Decay Generator', '{{ decay_file }}')
decay_gen.vertex = [ 0., 0., 0. ]
sim.generators = [ALP_gen, decay_gen]

#Ecal and Hcal hardwired/geometry stuff
import LDMX.Ecal.ecal_hardcoded_conditions
from LDMX.Ecal import EcalGeometry

from LDMX.Hcal import HcalGeometry
import LDMX.Hcal.hcal_hardcoded_conditions

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

p.sequence=[ sim,
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

p.outputFiles = [ '{{ root_file }}' ]
