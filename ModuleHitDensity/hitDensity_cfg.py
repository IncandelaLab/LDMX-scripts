import os
import sys
fileIn = sys.argv[1:]
fileName =  " ".join(sys.argv[1:])

from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('myAna')

import LDMX.Ecal.ecal_hardcoded_conditions
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import hcal
from LDMX.Ecal import vetos

# p.maxEvents = 1
p.maxEvents = -1
p.run = 2

print(fileIn)
p.inputFiles  = ["simoutput_electron.root"]
# p.inputFiles  = ["simoutput_photon_central_normal_noise.root"]
# p.inputFiles  = ["simoutput_100_photon_central_normal_noise.root"]
p.histogramFile = "histo_hit_density_electron.root"


myAna = ldmxcfg.Analyzer.from_file('HitDensity.cxx')


p.sequence = [myAna]


