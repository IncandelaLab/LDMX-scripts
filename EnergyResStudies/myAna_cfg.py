import os
import sys
fileIn = sys.argv[1:]
fileName =  " ".join(sys.argv[1:])

from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('myAna')

import LDMX.Ecal.ecal_hardcoded_conditions
from LDMX.Ecal import ecal_geometry
from LDMX.Hcal import hcal
from LDMX.Ecal import vetos

p.max_events = -1
p.run = 2

print(fileIn)
p.input_files  = fileIn
p.histogram_file = "myHistoOnEventFile.root"


my_ana = ldmxcfg.Analyzer.from_file('MyAna.cxx', needs=['Ecal_Event','Hcal_Event','Recon_Event','SimCore_Event','Tracking_Event', 'TrigScint_Event'])

p.sequence = [my_ana]
