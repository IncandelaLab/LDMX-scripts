from LDMX.Framework import ldmxcfg

p=ldmxcfg.Process("v3_v13")   #this will be appended to your branch names
p.run = 14      #run number is now the seeding number


#importing overall simulation modules
from LDMX.SimCore import simulator
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions as ecal_conditions
import LDMX.Hcal.hcal_hardcoded_conditions as hcal_conditions

#importing and setting up the generator
from LDMX.Biasing import ecal    #has different filtering/biasing configurations
from LDMX.SimCore import generators
sim = ecal.photo_nuclear("ldmx-det-v13", generators.single_4gev_e_upstream_tagger())   #fires a 4Gev e- upstream of tagger tracker

#importing processing modules
import LDMX.Ecal.digi as ecal_digi
import LDMX.Hcal.digi as hcal_digi
import LDMX.Ecal.vetos as ecal_vetos
import LDMX.Hcal.hcal as hcal_py
from LDMX.Recon.simpleTrigger import simpleTrigger          #reasonable default parameters for trigger
from LDMX.Recon.electronCounter import ElectronCounter      #needed since software allows for multi-e samples

#setting up trigger scintillator
from LDMX.TrigScint.trigScint import TrigScintDigiProducer
from LDMX.TrigScint.trigScint import TrigScintClusterProducer
from LDMX.TrigScint.trigScint import trigScintTrack
tsDigisUp   = TrigScintDigiProducer.up()
tsDigisTag  = TrigScintDigiProducer.tagger()
tsDigisDown = TrigScintDigiProducer.down()

#this block is for trigger skimming
#first two lines required for trigger skimming to perform correctly
eCount = ElectronCounter(1, "ElectronCounter") #first argument is number of electrons in simulation
eCount.use_simulated_electron_number = True
#these actually lines turn on trigger skimming
p.skimDefaultIsDrop()
p.skimConsider(simpleTrigger.instanceName)


#setting the actions in order for simulation
p.sequence = [sim, ecal_digi.EcalDigiProducer(), ecal_digi.EcalRecProducer(),
              ecal_vetos.EcalVetoProcessor(),    #EcalVetoProcessor calculates all the Ecal  veto variables and applies the BDT
              hcal_digi.HcalDigiProducer(), hcal_digi.HcalRecProducer(),
              hcal_py.HcalVetoProcessor(),
              tsDigisUp, tsDigisTag, tsDigisDown,
              eCount, simpleTrigger]


#output parameters
p.outputFiles=["100k_pn_testprod.root"]
p.maxEvents = 100000
p.maxTriesPerEvent = 1    #if != 1, pay attention to last output line for how many events started
p.logFrequencency = 1000
print(p)
