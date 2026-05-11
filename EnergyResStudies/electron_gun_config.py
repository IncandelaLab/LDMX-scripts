#!/bin/python

import sys
import os

# we need the ldmx configuration package to construct the object
from LDMX.Framework import ldmxcfg

# set a 'pass name'
passName="sim"
p = ldmxcfg.Process(passName)

# Set run parameters.
p.max_events = 10000
p.run = 1
p.max_tries_per_event = 1

#import all processors
from LDMX.SimCore import generators
from LDMX.SimCore import simulator

# Instantiate the simulator.
sim = simulator.Simulator(instance_name = "mySim")

# Set the path to the detector to use (pulled from job config)
detector='ldmx-det-v15-8gev'
sim.set_detector(detector, include_scoring_planes_minimal=True)


# Setup the multi-particle gun
mpgGen = generators.Multi(instance_name = "mgpGen")
mpgGen.vertex = [ 0., 0., 200. ] # mm
mpgGen.n_particles = 1
mpgGen.pdg_id = 11
mpgGen.enable_poisson = False
mpgGen.beam_spot_smear = [20., 80., 0.]

# import math
# import numpy as np
# theta = math.radians(5.65)
# beamEnergyMeV=1000*beamEnergy
# px = beamEnergyMeV*math.sin(theta)
# py = 0.;
# pz= beamEnergyMeV*math.cos(theta)
px = 0.
py = 0.
pz = 3000.
mpgGen.momentum = [ px, py, pz ]

# Set the multiparticle gun as generator
sim.generators = [ mpgGen ]

#Ecal and Hcal geometry stuff
import LDMX.Ecal.ecal_geometry
import LDMX.Hcal.hcal_geometry
import LDMX.Ecal.ecal_hardcoded_conditions

# ecal digi chain
from LDMX.Ecal import digi as eDigi
from LDMX.DQM import dqm


ecalDigi   = eDigi.EcalDigiProducer()
ecalReco   = eDigi.EcalRecProducer()
ecalDigiVerDQM = dqm.EcalDigiVerify()

# default is 2 (WARNING); but then log_frequency is ignored. level 1 = INFO.
p.logger.term_level = 1
p.log_frequency = 10
p.sequence=[ sim, ecalDigi, ecalReco, ecalDigiVerDQM]

p.keep = [ "drop MagnetScoringPlaneHits", "drop TrackerScoringPlaneHits", "drop HcalScoringPlaneHits"]

p.output_files = ["simoutput.root"]
p.histogram_file = 'hist.root'

print("Simulation configured to produce output files:", p.output_files, "and histogram file:", p.histogram_file)
