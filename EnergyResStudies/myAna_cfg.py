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

p.maxEvents = -1
p.run = 2

print(fileIn)
p.inputFiles  = fileIn
p.histogramFile = "myHistoOnEventFile.root"


myAna = ldmxcfg.Analyzer.from_file('/sdf/home/t/tamasvami/ReEcalVeto/ldmx-sw/MyAna.cxx')


p.sequence = [myAna]


