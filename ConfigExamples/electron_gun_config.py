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
mpgGen.vertex = [ 0., 0., 200. ] # mm                                                                                                                              
mpgGen.nParticles = 1
mpgGen.pdgID = 11
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
pz= 3000.
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


ecalDigi   =eDigi.EcalDigiProducer('EcalDigis', si_thickness = 0.4)
ecalReco   =eDigi.EcalRecProducer('ecalRecon')
ecalDigiVerDQM = dqm.EcalDigiVerify()

# default is 2 (WARNING); but then logFrequency is ignored. level 1 = INFO.
p.logger.termLevel = 1 
p.logFrequency = 10
p.sequence=[ sim, ecalDigi, ecalReco, ecalDigiVerDQM]

p.keep = [ "drop MagnetScoringPlaneHits", "drop TrackerScoringPlaneHits", "drop HcalScoringPlaneHits"]

p.outputFiles=["simoutput.root"]
p.histogramFile = f'hist.root'

print("Simulation configured to produce output files:", p.outputFiles, "and histogram file:", p.histogramFile)
    
