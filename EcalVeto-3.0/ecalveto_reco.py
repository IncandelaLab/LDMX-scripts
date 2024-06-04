from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('SegmipBDTReco')

p.maxTriesPerEvent = 10000

p.maxEvents = 100

p.inputFiles = ['mc_v14-8gev-8.0GeV-1e-ecal_photonuclear_run100_t1714988693.root']

p.outputFiles = [f'mc_v14-8gev-8.0GeV-1e-ecal_photonuclear_run100_t1714988693_segmipBDTReco.root']
p.termLogLevel = 0

# EcalVeto
import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.vetos as ecal_vetos

p.sequence = [ecal_vetos.EcalVetoProcessor()]
