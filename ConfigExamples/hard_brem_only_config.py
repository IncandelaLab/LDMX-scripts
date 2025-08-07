from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('test')

from LDMX.SimCore import simulator as sim
from LDMX.SimCore import generators as gen
from LDMX.Biasing import filters
mySim = sim.simulator( "mySim" )
det = 'ldmx-det-v14-8gev'
mySim.setDetector(det, True )
mySim.generators.append( gen.single_8gev_e_upstream_tagger() )
mySim.beamSpotSmear = [20.,80.,0.]
mySim.description = 'Basic test Simulation'
mySim.actions.clear()
mySim.actions.extend([
        filters.TaggerVetoFilter(thresh=2*3800.),
        filters.TargetBremFilter(recoil_max_p = 3000.,brem_min_e = 5000.),
])

##################################################################
# Below should be the same for all sim scenarios

import os
import sys

p.run = 5
p.maxEvents = 1
p.totalEvents = 10000

p.histogramFile = f'hist_{p.run}_hardbrem.root'
p.outputFiles = [f'events_{p.run}_hardbrem.root']

# Load the ECAL modules
import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Ecal.digi as ecal_digi


# Load the HCAL modules
import LDMX.Hcal.HcalGeometry
import LDMX.Hcal.hcal_hardcoded_conditions

from LDMX.Recon.simpleTrigger import TriggerProcessor
simpleTrig = TriggerProcessor('trigger', 8000.)

# Load ecal veto and use tracking in it
ecalDigi   =ecal_digi.EcalDigiProducer('EcalDigis')
ecalReco   =ecal_digi.EcalRecProducer('ecalRecon')

p.logger.termLevel = 5

p.sequence = [mySim,ecalDigi,ecalReco,simpleTrig]

p.skimDefaultIsDrop()
p.skimConsider(simpleTrig.instanceName)

# In case you are interested before the trigger too
# Add full tracking for both tagger and recoil trackers: digi, seeds, CFK, ambiguity resolution, GSF, DQM
# p.sequence.extend(full_tracking_sequence.sequence)
# p.sequence.extend(full_tracking_sequence.dqm_sequence)
# p.sequence.extend([
#         ecal_digi.EcalDigiProducer(),
#         ecal_digi.EcalRecProducer(), 
#         ecal_cluster.EcalClusterProducer(),
#         ecal_veto,
#         ecal_mip,
#         ecal_veto_pnet,
#         hcal_digi_reco,
#         hcal_veto,
#         *ts_digis,
#         *ts_clusters,
#         trigScintTrack,
#         count, TriggerProcessor('trigger', 8000.),
#         dqm.PhotoNuclearDQM(),
#         dqm.EcalClusterAnalyzer()
#         ])

# p.sequence.extend(dqm.all_dqm)
