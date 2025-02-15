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
pz= 3000.
mpgGen.momentum = [ px, py, pz ]

# Set the multiparticle gun as generator
sim.generators = [ mpgGen ]

#Ecal and Hcal geometry stuff
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
from LDMX.Ecal.ecal_hardcoded_conditions import EcalTrigPrimConditionsHardcode
from LDMX.Ecal.ecal_hardcoded_conditions import EcalReconConditionsHardcode
from LDMX.Conditions.SimpleCSVTableProvider import SimpleCSVIntegerTableProvider, SimpleCSVDoubleTableProvider

# this used to be 0.6
noise_in_ADC = 1.6

EcalHgcrocConditionsHardcode=SimpleCSVDoubleTableProvider("EcalHgcrocConditions", [
            "PEDESTAL",
            "NOISE",
            "MEAS_TIME",
            "PAD_CAPACITANCE",
            "TOT_MAX",
            "DRAIN_RATE",
            "GAIN",
            "READOUT_THRESHOLD",
            "TOA_THRESHOLD",
            "TOT_THRESHOLD"
        ])

EcalHgcrocConditionsHardcode.validForAllRows([
    50. , #PEDESTAL - ADC
    noise_in_ADC, #NOISE - ADC
    0.0, #MEAS_TIME - ns
    20., #PAD_CAPACITANCE - pF
    200., #TOT_MAX - ns - maximum time chip would be in TOT mode
    10240. / 200., #DRAIN_RATE - fC/ns
    320./1024/20., #GAIN - 320. fC / 1024. counts / 20 pF - conversion from ADC to mV
    50. + 3., #READOUT_THRESHOLD - 3 ADC counts above pedestal
    50.*320./1024/20. + 5 *37*0.1602/20., #TOA_THRESHOLD - mV - ~5  MIPs above pedestal
    50.*320./1024/20. + 50*37*0.1602/20., #TOT_THRESHOLD - mV - ~50 MIPs above pedestal
    ])

# ecal digi chain
from LDMX.Ecal import digi as eDigi
from LDMX.DQM import dqm


avgGain = 0.3125/20.
ecalDigi   =eDigi.EcalDigiProducer('EcalDigis')
ecalDigi.avgNoiseRMS = noise_in_ADC*avgGain
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
    
