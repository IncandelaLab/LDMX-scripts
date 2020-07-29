import os

from LDMX.Framework import ldmxcfg
from LDMX.SimCore import generators
from LDMX.SimCore import simulator
from LDMX.Biasing import filters
from LDMX.Ecal import digi
from LDMX.Ecal import vetos
from LDMX.EventProc import hcal
from LDMX.EventProc import trigScintDigis

p=ldmxcfg.Process("v12")

#
# These need to be imported after the process is defined for now
#
from LDMX.EventProc import simpleTrigger
from LDMX.EventProc import trackerHitKiller

sim = simulator.simulator("simulator")

#
# Set the detector to use and enable the scoring planes
#
sim.setDetector( '{{ detector }}' , True)

#
# Set run parameters
#
sim.runNumber = {{ run }}
sim.description = "ECal photo-nuclear, xsec bias 450"
sim.randomSeeds = [ {{ seed1 }} , {{ seed2 }} ]
sim.beamSpotSmear = [20., 80., 0]

#
# Fire an electron upstream of the tagger tracker
#
sim.generators.append(generators.single_4gev_e_upstream_tagger())

#
# Enable and configure the biasing
#
sim.biasing_enabled = True
sim.biasing_particle = 'gamma'
sim.biasing_process = 'photonNuclear'
sim.biasing_volume = 'ecal'
sim.biasing_threshold = 2500.
sim.biasing_factor = 450

#
# Configure the sequence in which user actions should be called.
#
sim.actions = [ filters.TaggerVetoFilter(), 
                filters.TargetBremFilter(),
                filters.EcalProcessFilter(), 
                filters.TrackProcessFilter.photo_nuclear()]

findableTrack = ldmxcfg.Producer("findable", "ldmx::FindableTrackProcessor")

trackerVeto = ldmxcfg.Producer("trackerVeto", "ldmx::TrackerVetoProcessor")

p.sequence=[ sim, digi.EcalDigiProducer(), digi.EcalRecProducer(), vetos.EcalVetoProcessor(), hcal.HcalDigiProducer(), hcal.HcalVetoProcessor(), trigScintDigis.TrigScintDigiProducer.up() , trigScintDigis.TrigScintDigiProducer.down() , trigScintDigis.TrigScintDigiProducer.tagger(), trackerHitKiller.TrackerHitKiller("trackerHitKiller",99.0), simpleTrigger.TriggerProcessor("simpleTrigger"), findableTrack, trackerVeto]

p.outputFiles=["{{ outputFile }}"]

p.maxEvents = 1000000
p.logFrequency = 5000
