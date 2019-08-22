#!/usr/bin/python

import sys
import os
import copy

# we need the ldmx configuration package to construct the object
from LDMX.Framework import ldmxcfg

# Setup producers with default templates
from LDMX.Ecal.ecalClusters import ecalClusters
from LDMX.EventProc.ecalDigis import ecalDigis
from LDMX.EventProc.multiElectronVeto import multiEleVeto

p = ldmxcfg.Process("recon")

p.libraries.append("libEcal.so")
p.libraries.append("libEventProc.so")

#ecalClusters.parameters["cutoff"] = 10.0
#ecalClusters.parameters["seedThreshold"] = 100.0
#ecalClusters.parameters["clusterCollName"] = "ecalClusters"
#ecalClusters.parameters["algoCollName"] = "ClusterAlgoResult"

#ecalClusters8 = copy.deepcopy(ecalClusters)
#ecalClusters8.parameters["cutoff"] = 8.0
#ecalClusters8.parameters["clusterCollName"] = "ecalClusters8"
#ecalClusters8.parameters["algoCollName"] = "ClusterAlgoResult8"

#ecalClusters6 = copy.deepcopy(ecalClusters)
#ecalClusters6.parameters["cutoff"] = 6.0
#ecalClusters6.parameters["clusterCollName"] = "ecalClusters6"
#ecalClusters6.parameters["algoCollName"] = "ClusterAlgoResult6"

#ecalClusters4 = copy.deepcopy(ecalClusters)
#ecalClusters4.parameters["cutoff"] = 4.0
#ecalClusters4.parameters["clusterCollName"] = "ecalClusters4"
#ecalClusters4.parameters["algoCollName"] = "ClusterAlgoResult4"

#ecalClusters2 = copy.deepcopy(ecalClusters)
#ecalClusters2.parameters["cutoff"] = 2.0
#ecalClusters2.parameters["clusterCollName"] = "ecalClusters2"
#ecalClusters2.parameters["algoCollName"] = "ClusterAlgoResult2"


multiEleVeto2 = copy.deepcopy(multiEleVeto)
multiEleVeto2.parameters["verbose"] = True

p.sequence=[multiEleVeto, multiEleVeto2]
p.maxEvents = 1000

p.inputFiles.append(sys.argv[1])
p.outputFiles.append(sys.argv[1].split("/")[-1].replace(".root", "_recon.root"))

#p.histogramFile = "THISISAHISTOFILE.root"
p.printMe()
