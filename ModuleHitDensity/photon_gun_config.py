#!/bin/python

import sys
import os

# we need the ldmx configuration package to construct the object
from LDMX.Framework import ldmxcfg

# set a 'pass name'
passName="sim"
p=ldmxcfg.Process(passName)

# Set run parameters.
p.maxEvents = 10000
# p.maxEvents = 10
p.run = 1
p.maxTriesPerEvent = 1

#import all processors
from LDMX.SimCore import generators
from LDMX.SimCore import simulator
from LDMX.Biasing import filters

from LDMX.Detectors.makePath import *
from LDMX.SimCore import simcfg

# Instantiate the simulator.
sim = simulator.simulator("mySim")

# Set the path to the detector to use (pulled from job config)
detector='ldmx-det-v14-8gev'
sim.setDetector( detector, True )
sim.scoringPlanes = makeScoringPlanesPath(detector)
sim.beamSpotSmear = [20., 80., 0]


# Setup the multi-particle gun
mpgGen = generators.multi( "mgpGen" )                                                                           
mpgGen.vertex = [ 0., 0, 200. ] # mm                                                                                                                              
mpgGen.nParticles = 1
mpgGen.pdgID = 22
mpgGen.enablePoisson = False #True                                                                                     

# import math
# import numpy as np
# theta = math.radians(5.65)
# beamEnergyMeV=1000*beamEnergy
# px = beamEnergyMeV*math.sin(theta)
# py = 0.;
# pz= beamEnergyMeV*math.cos(theta)
px = 0.
py = 0.
pz= 2500.
mpgGen.momentum = [ px, py, pz ]

# Set the multiparticle gun as generator
sim.generators = [ mpgGen ]

#Ecal and Hcal geometry stuff
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions


# ecal digi chain
from LDMX.Ecal import digi as eDigi
from LDMX.DQM import dqm

from LDMX.Recon.electronCounter import ElectronCounter
from LDMX.Recon.simpleTrigger import TriggerProcessor
trigger = TriggerProcessor('trigger', 8000.)

avgGain = 0.3125/20.
ecalDigi   =eDigi.EcalDigiProducer('EcalDigis')
ecalDigi.avgNoiseRMS = 0.6*avgGain # default
# ecalDigi.avgNoiseRMS = 1.6*avgGain
ecalReco   =eDigi.EcalRecProducer('ecalRecon')
ecalDigiVerDQM = dqm.EcalDigiVerify()

# default is 2 (WARNING); but then logFrequency is ignored. level 1 = INFO.
p.logger.termLevel = 10
p.logFrequency = 100
p.sequence=[ sim, ecalDigi, ecalReco, trigger, ecalDigiVerDQM]

p.keep = [ "drop MagnetScoringPlaneHits", "drop TrackerScoringPlaneHits", "drop HcalScoringPlaneHits"]

p.outputFiles=["simoutput.root"]
p.histogramFile = f'hist.root'

p.skimDefaultIsDrop()
p.skimConsider(trigger.instanceName)

print("Simulation configured to produce output files:", p.outputFiles, "and histogram file:", p.histogramFile)
    
