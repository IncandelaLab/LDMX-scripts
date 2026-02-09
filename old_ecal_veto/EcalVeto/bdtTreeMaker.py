import os
import argparse
import numpy as np
import ROOT as r
from rootpy.tree import Tree, TreeModel, TreeChain, FloatCol, IntCol, BoolCol, FloatArrayCol
from rootpy.io import root_open
#import rootpy.stl as stl
import math


# Will go in utils
cellMap = np.loadtxt('cellmodule.txt')
scoringPlaneZ = 220
ecalFaceZ = 223.8000030517578
cell_radius = 5
mcid = cellMap[:,0].tolist()

layerZs = [223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,
        266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,
        322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,
        375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,
        448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125]

#radius = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
#4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
#9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
#17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
#30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
#55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
#119.06854141, 121.20048803, 127.5236134, 121.99024095]

#from 2e (currently not used)
radius_beam_68 = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]

#from 1e
radius_recoil_68_p_0_500_theta_0_10 = [4.045666158618167, 4.086393662224346, 4.359141107602775, 4.666549994726691, 5.8569181911416015, 6.559716356124256, 8.686967529043072, 10.063482736354674, 13.053528344041274, 14.883496407943747, 18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406, 28.329169392203216, 29.468032123356345, 34.03271241527079, 35.03747443690781, 38.50748727211848, 39.41576583301171, 42.63622296033334, 45.41123601592071, 48.618139095742876, 48.11801717451056, 53.220539860213655, 58.87753380915155, 66.31550881539764, 72.94685877928593, 85.95506228335348, 89.20607201266672, 93.34370253818409, 96.59471226749734, 100.7323427930147, 103.98335252232795]

radius_recoil_68_p_500_1500_theta_0_10 = [4.081926458777424, 4.099431732299409, 4.262428482867968, 4.362017581473145, 4.831341579961153, 4.998346041276382, 6.2633736512415705, 6.588371889265881, 8.359969947444522, 9.015085558044309, 11.262722588206483, 12.250305471269183, 15.00547660437276, 16.187264014640103, 19.573764900578503, 20.68072032434797, 24.13797140783321, 25.62942209291236, 29.027596514735617, 30.215039667389316, 33.929540248019585, 36.12911729771914, 39.184563500620946, 42.02062468386282, 46.972125628650204, 47.78214816041894, 55.88428562462974, 59.15520134927332, 63.31816666637158, 66.58908239101515, 70.75204770811342, 74.022963432757, 78.18592874985525, 81.45684447449884]

radius_recoil_68_theta_10_20 = [4.0251896715647115, 4.071661598616328, 4.357690094817289, 4.760224640141712, 6.002480766325418, 6.667318981016246, 8.652513285172342, 9.72379373302137, 12.479492693251478, 14.058548828317289, 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302, 31.768946746958296, 35.79247330197688, 37.27810357669942, 41.657281051476545, 42.628141392692626, 47.94208483539388, 49.9289473559796, 54.604030254423975, 53.958762417361655, 53.03339560920388, 57.026277390001425, 62.10810455035879, 66.10098633115634, 71.1828134915137, 75.17569527231124, 80.25752243266861, 84.25040421346615, 89.33223137382352, 93.32511315462106]

radius_recoil_68_theta_20_end = [4.0754238481177705, 4.193693485630508, 5.14209420056253, 6.114996249971468, 7.7376807326481645, 8.551663213602291, 11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604, 25.04088272634407, 28.853696411302344, 34.72538105333071, 40.21218694947545, 46.07344239520299, 50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047, 74.82378572939959, 89.4528387422834, 93.18228303096758, 92.51751129204555, 98.80228884380018, 111.17537347472128, 120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185, 177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696]

radius_68 = [radius_beam_68,radius_recoil_68_p_0_500_theta_0_10, radius_recoil_68_p_500_1500_theta_0_10,radius_recoil_68_theta_10_20,radius_recoil_68_theta_20_end]

def CallX(Hitz, Recoilx, Recoily, Recoilz, RPx, RPy, RPz):
    Point_xz = [Recoilx, Recoilz]
    #Almost never happens
    if RPx == 0:
        slope_xz = 99999
    else:
        slope_xz = RPz / RPx
    
    x_val = (float(Hitz - Point_xz[1]) / float(slope_xz)) + Point_xz[0]
    return x_val
    
def CallY(Hitz, Recoilx, Recoily, Recoilz, RPx, RPy, RPz):
    Point_yz = [Recoily, Recoilz]
    #Almost never happens
    if RPy == 0:
        slope_yz = 99999
    else:
        slope_yz = RPz / RPy
    
    y_val = (float(Hitz - Point_yz[1]) / float(slope_yz)) + Point_yz[0]
    return y_val

def mask(id):
    cell   = (id & 0xFFFF8000) >> 15
    module = (id & 0x7000) >> 12
    layer  = (id & 0xFF0) >> 4
    return cell, module, layer

#calculate angle of electrons wrt normal of front ECal face
def getTheta(p):
    return math.acos(p[2]/np.sqrt(p[0]**2+p[1]**2+p[2]**2))

def getMagnitude(p):
    return np.sqrt(p[0]**2+p[1]**2+p[2]**2)

# get scoring plane hits for electrons (both target and ECAL SP)
def getElectronSPHits(simParticles, spHits, targetSPHits):
    electronSPHits = []
    for particle in simParticles:
        """
        if particle.getPdgID() == 22 and particle.getParentCount() > 0 and particle.getParent(0).getPdgID()==11 and particle.getVertex()[2] < 1.2 and particle.getEnergy() > 2500:
            print 'found photon'
            particle.Print()
            for hit in spHits:
                sp = hit.getSimParticle()
                if sp == particle and hit.getMomentum()[2] > 0 and hit.getLayerID()==1:
                    hit.Print()
        """
        if particle.getPdgID() == 11 and particle.getParentCount() == 0:
            #print 'found electron'
            #particle.Print()
            pmax = 0
            ecalSPHit = None
            targetSPHit = None
            isphit =0
            #print 'Looping over HCAL SP hits'
            for hit in spHits:
                sp = hit.getSimParticle()
                #if sp.getPdgID() == 11 and sp.getParentCount() == 0:
                    #print isphit
                #    sp.Print()
                #    isphit += 1
                if sp == particle and hit.getMomentum()[2] > 0 and hit.getLayerID()==1:
                    pvec = hit.getMomentum()
                    p = math.sqrt(pvec[0]**2 + pvec[1]**2 + pvec[2]**2)
                    #hit.Print()
                    #print 'momentum:',p
                    if p > pmax:
                        pmax = p
                        ecalSPHit = hit
            #print 'Looping over Target SP hits'
            pmax = 0
            for hit in targetSPHits:
                sp = hit.getSimParticle()
                if sp == particle and hit.getMomentum()[2] > 0 and hit.getLayerID()==2:
                    #hit.Print()
                    pvec = hit.getMomentum()
                    p = math.sqrt(pvec[0]**2 + pvec[1]**2 + pvec[2]**2)
                    if p > pmax:
                        pmax = p
                        targetSPHit = hit
            if ecalSPHit != None:
                electronSPHits.append((ecalSPHit,targetSPHit,particle))
    return electronSPHits

# project particle trajectory based on initial momentum and position
def getElectronTrajectory(pvec, pos):
    positions = []
    for layerZ in layerZs:
        posX = pos[0] + pvec[0]/pvec[2]*(layerZ - pos[2])
        posY = pos[1] + pvec[1]/pvec[2]*(layerZ - pos[2])
        positions.append((posX,posY))
    return positions

# Event setup ... defines tree branches
class EcalVetoEvent(TreeModel):
    evtNum = IntCol()
    pnWeight = FloatCol()
    trigPass = BoolCol()
    nReadoutHits = IntCol()
    summedDet = FloatCol()
    summedTightIso = FloatCol()
    maxCellDep = FloatCol()
    showerRMS = FloatCol()
    xStd = FloatCol()
    yStd = FloatCol()
    avgLayerHit = FloatCol()
    stdLayerHit = FloatCol()
    deepestLayerHit = IntCol()
    discValue = FloatCol()
    hcalMaxPE = FloatCol()
    passHcalVeto = BoolCol()
    passTrackerVeto = BoolCol()
    recoilPx = FloatCol()
    recoilPy = FloatCol()
    recoilPz = FloatCol()
    recoilX = FloatCol()
    recoilY = FloatCol()
    fiducial = BoolCol()
    leadhadpid = IntCol()
    leadhadke = FloatCol()
    leadhadthetaz = FloatCol()
    nelectrons = IntCol()
    trigEnergy = FloatCol()
    ecalBackEnergy = FloatCol()
    eleP = FloatArrayCol(3)
    elePTarget = FloatArrayCol(3)
    elePosSP = FloatArrayCol(3)
    photonP = FloatArrayCol(3)
    photonPosSP = FloatArrayCol(3)
    ele68TotalEnergies = FloatArrayCol(34)
    photon68TotalEnergies = FloatArrayCol(34)
    overlap68TotalEnergies = FloatArrayCol(34)
    outside68TotalEnergies = FloatArrayCol(34)
    outside68TotalNHits = FloatArrayCol(34)
    outside68Xmeans = FloatArrayCol(34)
    outside68Ymeans = FloatArrayCol(34)
    outside68Xstds = FloatArrayCol(34)
    outside68Ystds = FloatArrayCol(34)
    ele68ContEnergy = FloatCol()
    ele68x2ContEnergy = FloatCol()
    ele68x3ContEnergy = FloatCol()
    ele68x4ContEnergy = FloatCol()
    ele68x5ContEnergy = FloatCol()
    photon68ContEnergy = FloatCol()
    photon68x2ContEnergy = FloatCol()
    photon68x3ContEnergy = FloatCol()
    photon68x4ContEnergy = FloatCol()
    photon68x5ContEnergy = FloatCol()
    overlap68ContEnergy = FloatCol()
    overlap68x2ContEnergy = FloatCol()
    overlap68x3ContEnergy = FloatCol()
    overlap68x4ContEnergy = FloatCol()
    overlap68x5ContEnergy = FloatCol()
    outside68ContEnergy = FloatCol()
    outside68x2ContEnergy = FloatCol()
    outside68x3ContEnergy = FloatCol()
    outside68x4ContEnergy = FloatCol()
    outside68x5ContEnergy = FloatCol()
    outside68ContNHits = FloatCol()
    outside68x2ContNHits = FloatCol()
    outside68x3ContNHits = FloatCol()
    outside68x4ContNHits = FloatCol()
    outside68x5ContNHits = FloatCol()
    outside68ContXmean = FloatCol()
    outside68x2ContXmean = FloatCol()
    outside68x3ContXmean = FloatCol()
    outside68x4ContXmean = FloatCol()
    outside68x5ContXmean = FloatCol()
    outside68ContYmean = FloatCol()
    outside68x2ContYmean = FloatCol()
    outside68x3ContYmean = FloatCol()
    outside68x4ContYmean = FloatCol()
    outside68x5ContYmean = FloatCol()
    outside68ContXstd = FloatCol()
    outside68x2ContXstd = FloatCol()
    outside68x3ContXstd = FloatCol()
    outside68x4ContXstd = FloatCol()
    outside68x5ContXstd = FloatCol()
    outside68ContYstd = FloatCol()
    outside68x2ContYstd = FloatCol()
    outside68x3ContYstd = FloatCol()
    outside68x4ContYstd = FloatCol()
    outside68x5ContYstd = FloatCol()
    outside68ContShowerRMS = FloatCol()
    outside68x2ContShowerRMS = FloatCol()
    outside68x3ContShowerRMS = FloatCol()
    outside68x4ContShowerRMS = FloatCol()
    outside68x5ContShowerRMS = FloatCol()


# Tree maker class
class bdtTreeMaker:
    def __init__(self,outdir,outfile,inputfiles,issignal):
        # Setup
        self.outdir = outdir
        self.outfile = outfile
        self.issignal = issignal

        # Create TChain reading from ntuples
        self.intree = r.TChain('LDMX_Events')

        for f in inputfiles:
            self.intree.Add(f)

        # Setup to read the collections we care about
        self.evHeader = r.ldmx.EventHeader()
        self.ecalVetoRes = r.TClonesArray('ldmx::EcalVetoResult')
        self.hcalVetoRes = r.TClonesArray('ldmx::HcalVetoResult')
        self.trackerVetoRes = r.TClonesArray('ldmx::TrackerVetoResult')
        self.hcalhits = r.TClonesArray('ldmx::HcalHit') #added by Jack
        self.trigRes = r.TClonesArray('ldmx::TriggerResult')
        self.hcalhits = r.TClonesArray('ldmx::HcalHit') #added by Jack
        self.ecalhits = r.TClonesArray('ldmx::EcalHit')
        self.simParticles = r.TClonesArray('ldmx::SimParticle')
        self.spHits = r.TClonesArray('ldmx::SimTrackerHit')
        self.targetSPHits = r.TClonesArray('ldmx::SimTrackerHit')

        self.intree.SetBranchAddress('EventHeader',r.AddressOf(self.evHeader))
        self.intree.SetBranchAddress('EcalVeto_recon',r.AddressOf(self.ecalVetoRes))
        self.intree.SetBranchAddress('HcalVeto_recon',r.AddressOf(self.hcalVetoRes))
        self.intree.SetBranchAddress('TrackerVeto_recon',r.AddressOf(self.trackerVetoRes))
        self.intree.SetBranchAddress('ecalDigis_recon',r.AddressOf(self.ecalhits)) 
        self.intree.SetBranchAddress('hcalDigis_recon',r.AddressOf(self.hcalhits)) 
        self.intree.SetBranchAddress('SimParticles_sim',r.AddressOf(self.simParticles))
        self.intree.SetBranchAddress('HcalScoringPlaneHits_sim',r.AddressOf(self.spHits))
        self.intree.SetBranchAddress('TargetScoringPlaneHits_sim',r.AddressOf(self.targetSPHits))

        #if not self.issignal:
            #self.intree.SetBranchAddress('Trigger_recon',r.AddressOf(self.trigRes))

        # Create output file and tree
        self.tfile = root_open(self.outfile,'recreate')
        self.tree = Tree('EcalVeto',model=EcalVetoEvent)

        # Initialize event count
        self.event_count = 0

    # Loop over all input events
    def run(self):
        while self.event_count < self.intree.GetEntries():
            self.intree.GetEntry(self.event_count)
            if self.event_count%1000 == 0:
                print 'Processing event ',self.event_count
            #if self.event_count>1000:
            #    return
            self.process()

    # Process an event
    def process(self):
        self.event_count += 1

        self.tree.evtNum = self.evHeader.getEventNumber()
        #self.tree.trigPass = 1 if self.issignal else self.trigRes[0].passed()
        self.tree.trigPass = 1 #self.trigRes[0].passed()

        # BDT input variables
        self.tree.nReadoutHits = self.ecalVetoRes[0].getNReadoutHits()
        self.tree.summedDet = self.ecalVetoRes[0].getSummedDet()
        self.tree.summedTightIso = self.ecalVetoRes[0].getSummedTightIso()
        self.tree.maxCellDep = self.ecalVetoRes[0].getMaxCellDep()
        self.tree.showerRMS = self.ecalVetoRes[0].getShowerRMS()
        self.tree.xStd = self.ecalVetoRes[0].getXStd()
        self.tree.yStd = self.ecalVetoRes[0].getYStd()
        self.tree.avgLayerHit = self.ecalVetoRes[0].getAvgLayerHit()
        self.tree.stdLayerHit = self.ecalVetoRes[0].getStdLayerHit()
        self.tree.deepestLayerHit = self.ecalVetoRes[0].getDeepestLayerHit()

        kemax = 0
        thetamax = 0
        pidmax = -1
        for particle in self.simParticles:
            if particle.getParentCount()>0 and particle.getParent(0).getPdgID()==22:
                p = particle.getMomentum()
                ke = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                if ke > kemax:
                    kemax = ke
                    thetamax = math.acos(p[2]/ke)
                    pidmax = particle.getPdgID()

        self.tree.leadhadpid = pidmax
        self.tree.leadhadke = kemax
        self.tree.leadhadthetaz = thetamax*180/math.pi

        # Stored value of bdt discriminator
        self.tree.discValue = self.ecalVetoRes[0].getDisc()
        #print self.event_count,self.evHeader.getEventNumber(),self.ecalVetoRes[0].getDisc(),self.trigRes[0].passed()

        # HCal MaxPE value, needed for HCAL veto
        self.tree.hcalMaxPE = self.hcalVetoRes[0].getMaxPEHit().getPE()
        self.tree.passHcalVeto = self.hcalVetoRes[0].passesVeto()
        self.tree.passTrackerVeto = self.trackerVetoRes[0].passesVeto()

        # Need to update to get this from first layer of tracker
        recoilPx = self.ecalVetoRes[0].getRecoilMomentum()[0]
        recoilPy = self.ecalVetoRes[0].getRecoilMomentum()[1]
        recoilPz = self.ecalVetoRes[0].getRecoilMomentum()[2]
        recoilX = self.ecalVetoRes[0].getRecoilX()
        recoilY = self.ecalVetoRes[0].getRecoilY()

        recoilfX = CallX(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)
        recoilfY = CallY(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)

        # Fiducial or not
        inside = False

        if not recoilX == -9999 and not recoilY == -9999 and not recoilPx == -9999 and not recoilPy == -9999 and not recoilPz == -9999:
            for x in cellMap:
                xdis = recoilfY - x[2]
                ydis = recoilfX - x[1]
                celldis = np.sqrt(xdis**2 + ydis**2)
                if celldis <= cell_radius:
                    inside = True
                    break

        self.tree.recoilPx = recoilPx
        self.tree.recoilPy = recoilPy
        self.tree.recoilPz = recoilPz
        self.tree.recoilX = recoilX
        self.tree.recoilY = recoilY
        self.tree.fiducial = inside

        # find electrons at scoring plane
        electronsAtSP = getElectronSPHits(self.simParticles, self.spHits, self.targetSPHits)
        #print electronsAtSP

        self.tree.nelectrons = len(electronsAtSP)

        electronLayerIntercepts = []
        photonLayerIntercepts = []
        recoilangle = -1 
        recoil_p = -1

        if len(electronsAtSP) > 0:
            eleinfo = electronsAtSP[0]
            ecalSPHit = eleinfo[0]
            targetSPHit = eleinfo[1]
            simParticle = eleinfo[2]
            pvec = ecalSPHit.getMomentum()
            pos = ecalSPHit.getPosition()
            recoilangle = getTheta(pvec)*180./math.pi 
            recoil_p = getMagnitude(pvec)
            pvec0 = [0,0,0]
            pos0 = [0,0,0]
            self.tree.eleP = [pvec[0],pvec[1],pvec[2]]
            self.tree.elePosSP = [pos[0],pos[1],pos[2]]
            pvec0 = targetSPHit.getMomentum() if targetSPHit != None else [0,0,0]
            pos0 = targetSPHit.getPosition() if targetSPHit != None else [0,0,0]
            photonP = [-pvec0[0],-pvec0[1],4000.0-pvec0[2]]
            self.tree.elePTarget = [pvec0[0],pvec0[1],pvec0[2]]
            self.tree.photonP = [photonP[0],photonP[1],photonP[2]]
            self.tree.photonPosSP = [pos0[0],pos0[1],pos0[2]]
            #print 'Photon momentum: ',photonP
            electronLayerIntercepts = getElectronTrajectory(pvec,pos)
            if pvec0 != [0,0,0]:
                photonLayerIntercepts = getElectronTrajectory(photonP,pos0)

        #print electronLayerIntercepts
        #print photonLayerIntercepts

        # compute new variables based on containment regions
        ele68Totals = np.zeros(34)
        ele68ContEnergy = 0
        ele68x2ContEnergy = 0
        ele68x3ContEnergy = 0
        ele68x4ContEnergy = 0
        ele68x5ContEnergy = 0
        photon68Totals = np.zeros(34)
        photon68ContEnergy = 0
        photon68x2ContEnergy = 0
        photon68x3ContEnergy = 0
        photon68x4ContEnergy = 0
        photon68x5ContEnergy = 0
        overlap68Totals = np.zeros(34)
        overlap68ContEnergy = 0
        overlap68x2ContEnergy = 0
        overlap68x3ContEnergy = 0
        overlap68x4ContEnergy = 0
        overlap68x5ContEnergy = 0
        outside68Totals = np.zeros(34)
        outside68ContEnergy = 0
        outside68x2ContEnergy = 0
        outside68x3ContEnergy = 0
        outside68x4ContEnergy = 0
        outside68x5ContEnergy = 0
        outside68NHits = np.zeros(34)
        outside68ContNHits = 0
        outside68x2ContNHits = 0
        outside68x3ContNHits = 0
        outside68x4ContNHits = 0
        outside68x5ContNHits = 0
        outside68Xmean = np.zeros(34)
        outside68ContXmean = 0
        outside68x2ContXmean = 0
        outside68x3ContXmean = 0
        outside68x4ContXmean = 0
        outside68x5ContXmean = 0
        outside68Ymean = np.zeros(34)
        outside68ContYmean = 0
        outside68x2ContYmean = 0
        outside68x3ContYmean = 0
        outside68x4ContYmean = 0
        outside68x5ContYmean = 0
        outside68Xstd = np.zeros(34)
        outside68ContXstd = 0
        outside68x2ContXstd = 0
        outside68x3ContXstd = 0
        outside68x4ContXstd = 0
        outside68x5ContXstd = 0
        outside68Ystd = np.zeros(34)
        outside68ContYstd = 0
        outside68x2ContYstd = 0
        outside68x3ContYstd = 0
        outside68x4ContYstd = 0
        outside68x5ContYstd = 0
        outside68HitPositions = []
        outside68x2HitPositions = []
        outside68x3HitPositions = []
        outside68x4HitPositions = []
        outside68x5HitPositions = []
        outside68WgtCentroidCoords = (0,0)
        outside68x2WgtCentroidCoords = (0,0)
        outside68x3WgtCentroidCoords = (0,0)
        outside68x4WgtCentroidCoords = (0,0)
        outside68x5WgtCentroidCoords = (0,0)
        trigEnergy = 0
        ecalBackEnergy = 0
        ir = -1
        if recoilangle==-1 or recoil_p==-1:
            ir = 1
        elif recoilangle<10 and recoil_p<500:
            ir = 1
        elif recoilangle<10 and recoil_p >= 500:
            ir = 2
        elif recoilangle<=20:
            ir = 3
        else:
            ir = 4
        ip = 1
        for hit in self.ecalhits:
            hitE = hit.getEnergy()
            if not hitE > 0:
                continue
            cell, module, layer = mask(hit.getID())
            if layer<20:
                trigEnergy += hitE
            else:
                ecalBackEnergy += hitE
            #print 'layer:',layer,hit.getLayer()
            mcid_val = 10*cell + module
            hitX = cellMap[mcid.index(mcid_val)][1]
            hitY = cellMap[mcid.index(mcid_val)][2]
            # get distances of hit from projected positions of electrons and photon
            distanceEle = math.sqrt((hitX-electronLayerIntercepts[layer][0])**2 + (hitY-electronLayerIntercepts[layer][1])**2) if len(electronLayerIntercepts) > 0 else -1
            distancePhoton = math.sqrt((hitX-photonLayerIntercepts[layer][0])**2 + (hitY-photonLayerIntercepts[layer][1])**2) if len(photonLayerIntercepts) > 0 else -1
            # add to containment totals depending on distance relative to containment radii
            # add to containment totals depending on distance relative to containment radii
            # i selects which radius of containment to use
            if distanceEle < radius_68[ir][layer] and distanceEle>0:
                ele68Totals[layer] += hitE
                ele68ContEnergy += hitE
            elif distanceEle >= radius_68[ir][layer] and distanceEle < 2*radius_68[ir][layer]:
                ele68x2ContEnergy += hitE
            elif distanceEle >= 2*radius_68[ir][layer] and distanceEle < 3*radius_68[ir][layer]:
                ele68x3ContEnergy += hitE
            elif distanceEle >= 3*radius_68[ir][layer] and distanceEle < 4*radius_68[ir][layer]:
                ele68x4ContEnergy += hitE
            elif distanceEle >= 4*radius_68[ir][layer] and distanceEle < 5*radius_68[ir][layer]:
                ele68x5ContEnergy += hitE
            if distancePhoton < radius_68[ip][layer] and distancePhoton>0:
                photon68Totals[layer] += hitE
                photon68ContEnergy += hitE
            elif distancePhoton >= radius_68[ip][layer] and distancePhoton  < 2*radius_68[ip][layer]:
                photon68x2ContEnergy += hitE
            elif distancePhoton >= 2*radius_68[ip][layer] and distancePhoton  < 3*radius_68[ip][layer]:
                photon68x3ContEnergy += hitE
            elif distancePhoton >= 3*radius_68[ip][layer] and distancePhoton  < 4*radius_68[ip][layer]:
                photon68x4ContEnergy += hitE
            elif distancePhoton >= 4*radius_68[ip][layer] and distancePhoton  < 5*radius_68[ip][layer]:
                photon68x5ContEnergy += hitE
            if distanceEle < radius_68[ir][layer] and distancePhoton < radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68Totals[layer] += hitE
                overlap68ContEnergy += hitE
            if distanceEle < 2*radius_68[ir][layer] and distancePhoton < 2*radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68x2ContEnergy += hitE
            if distanceEle < 3*radius_68[ir][layer] and distancePhoton < 3*radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68x3ContEnergy += hitE
            if distanceEle < 4*radius_68[ir][layer] and distancePhoton < 4*radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68x4ContEnergy += hitE
            if distanceEle < 5*radius_68[ir][layer] and distancePhoton < 5*radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68x5ContEnergy += hitE
            if distanceEle > radius_68[ir][layer] and distancePhoton > radius_68[ip][layer]:
                outside68Totals[layer] += hitE
                outside68NHits[layer] += 1
                outside68Xmean[layer] += hitX*hitE
                outside68Ymean[layer] += hitY*hitE
                outside68ContEnergy += hitE
                outside68ContNHits += 1
                outside68ContXmean += hitX*hitE
                outside68ContYmean += hitY*hitE
                outside68WgtCentroidCoords = (outside68WgtCentroidCoords[0] + hitX*hitE, outside68WgtCentroidCoords[1] + hitY*hitE)
                outside68HitPositions.append((hitX,hitY,layer,hitE))
            if distanceEle > 2*radius_68[ir][layer] and distancePhoton > 2*radius_68[ip][layer]:
                outside68x2ContEnergy += hitE
                outside68x2ContNHits += 1
                outside68x2ContXmean += hitX*hitE
                outside68x2ContYmean += hitY*hitE
                outside68x2WgtCentroidCoords = (outside68x2WgtCentroidCoords[0] + hitX*hitE, outside68x2WgtCentroidCoords[1] + hitY*hitE)
                outside68x2HitPositions.append((hitX,hitY,layer,hitE))
            if distanceEle > 3*radius_68[ir][layer] and distancePhoton > 3*radius_68[ip][layer]:
                outside68x3ContEnergy += hitE
                outside68x3ContNHits += 1
                outside68x3ContXmean += hitX*hitE
                outside68x3ContYmean += hitY*hitE
                outside68x3WgtCentroidCoords = (outside68x3WgtCentroidCoords[0] + hitX*hitE, outside68x3WgtCentroidCoords[1] + hitY*hitE)
                outside68x3HitPositions.append((hitX,hitY,layer,hitE))
            if distanceEle > 4*radius_68[ir][layer] and distancePhoton > 4*radius_68[ip][layer]:
                outside68x4ContEnergy += hitE
                outside68x4ContNHits += 1
                outside68x4ContXmean += hitX*hitE
                outside68x4ContYmean += hitY*hitE
                outside68x4WgtCentroidCoords = (outside68x4WgtCentroidCoords[0] + hitX*hitE, outside68x4WgtCentroidCoords[1] + hitY*hitE)
                outside68x4HitPositions.append((hitX,hitY,layer,hitE))
            if distanceEle > 5*radius_68[ir][layer] and distancePhoton > 5*radius_68[ip][layer]:
                outside68x5ContEnergy += hitE
                outside68x5ContNHits += 1
                outside68x5ContXmean += hitX*hitE
                outside68x5ContYmean += hitY*hitE
                outside68x5WgtCentroidCoords = (outside68x5WgtCentroidCoords[0] + hitX*hitE, outside68x5WgtCentroidCoords[1] + hitY*hitE)
                outside68x5HitPositions.append((hitX,hitY,layer,hitE))

        outside68ContShowerRMS = 0
        # mean and std deviations of hits outside containment regions
        for hit in outside68HitPositions:
            distanceCentroid = math.sqrt((hit[0] - outside68WgtCentroidCoords[0])**2 + (hit[1] - outside68WgtCentroidCoords[1])**2)
            outside68ContShowerRMS += distanceCentroid*hit[3]
            layer = hit[2]
            if outside68Totals[layer] > 0:
                outside68Xstd[layer] += ((hit[0] - (outside68Xmean[layer]/outside68Totals[layer]))**2)*hit[3]
                outside68Ystd[layer] += ((hit[1] - (outside68Ymean[layer]/outside68Totals[layer]))**2)*hit[3]
            if outside68ContEnergy > 0:
                outside68ContXstd += ((hit[0] - (outside68ContXmean/outside68ContEnergy))**2)*hit[3]
                outside68ContYstd += ((hit[1] - (outside68ContYmean/outside68ContEnergy))**2)*hit[3]

        outside68x2ContShowerRMS = 0
        # mean and std deviations of hits outside containment regions
        for hit in outside68x2HitPositions:
            distanceCentroid = math.sqrt((hit[0] - outside68x2WgtCentroidCoords[0])**2 + (hit[1] - outside68x2WgtCentroidCoords[1])**2)
            outside68x2ContShowerRMS += distanceCentroid*hit[3]
            if outside68x2ContEnergy > 0:
                outside68x2ContXstd += ((hit[0] - (outside68x2ContXmean/outside68x2ContEnergy))**2)*hit[3]
                outside68x2ContYstd += ((hit[1] - (outside68x2ContYmean/outside68x2ContEnergy))**2)*hit[3]

        outside68x3ContShowerRMS = 0
        # mean and std deviations of hits outside containment regions
        for hit in outside68x3HitPositions:
            distanceCentroid = math.sqrt((hit[0] - outside68x3WgtCentroidCoords[0])**2 + (hit[1] - outside68x3WgtCentroidCoords[1])**2)
            outside68x3ContShowerRMS += distanceCentroid*hit[3]
            if outside68x3ContEnergy > 0:
                outside68x3ContXstd += ((hit[0] - (outside68x3ContXmean/outside68x3ContEnergy))**2)*hit[3]
                outside68x3ContYstd += ((hit[1] - (outside68x3ContYmean/outside68x3ContEnergy))**2)*hit[3]

        outside68x4ContShowerRMS = 0
        # mean and std deviations of hits outside containment regions
        for hit in outside68x4HitPositions:
            distanceCentroid = math.sqrt((hit[0] - outside68x4WgtCentroidCoords[0])**2 + (hit[1] - outside68x4WgtCentroidCoords[1])**2)
            outside68x4ContShowerRMS += distanceCentroid*hit[3]
            if outside68x4ContEnergy > 0:
                outside68x4ContXstd += ((hit[0] - (outside68x4ContXmean/outside68x4ContEnergy))**2)*hit[3]
                outside68x4ContYstd += ((hit[1] - (outside68x4ContYmean/outside68x4ContEnergy))**2)*hit[3]

        outside68x5ContShowerRMS = 0
        # mean and std deviations of hits outside containment regions
        for hit in outside68x5HitPositions:
            distanceCentroid = math.sqrt((hit[0] - outside68x5WgtCentroidCoords[0])**2 + (hit[1] - outside68x5WgtCentroidCoords[1])**2)
            outside68x5ContShowerRMS += distanceCentroid*hit[3]
            if outside68x5ContEnergy > 0:
                outside68x5ContXstd += ((hit[0] - (outside68x5ContXmean/outside68x5ContEnergy))**2)*hit[3]
                outside68x5ContYstd += ((hit[1] - (outside68x5ContYmean/outside68x5ContEnergy))**2)*hit[3]

        if outside68ContEnergy > 0:
            outside68ContXmean /= outside68ContEnergy
            outside68ContYmean /= outside68ContEnergy
            outside68ContXstd = math.sqrt(outside68ContXstd/outside68ContEnergy)
            outside68ContYstd = math.sqrt(outside68ContYstd/outside68ContEnergy)

        if outside68x2ContEnergy > 0:
            outside68x2ContXmean /= outside68x2ContEnergy
            outside68x2ContYmean /= outside68x2ContEnergy
            outside68x2ContXstd = math.sqrt(outside68x2ContXstd/outside68x2ContEnergy)
            outside68x2ContYstd = math.sqrt(outside68x2ContYstd/outside68x2ContEnergy)

        if outside68x3ContEnergy > 0:
            outside68x3ContXmean /= outside68x3ContEnergy
            outside68x3ContYmean /= outside68x3ContEnergy
            outside68x3ContXstd = math.sqrt(outside68x3ContXstd/outside68x3ContEnergy)
            outside68x3ContYstd = math.sqrt(outside68x3ContYstd/outside68x3ContEnergy)

        if outside68x4ContEnergy > 0:
            outside68x4ContXmean /= outside68x4ContEnergy
            outside68x4ContYmean /= outside68x4ContEnergy
            outside68x4ContXstd = math.sqrt(outside68x4ContXstd/outside68x4ContEnergy)
            outside68x4ContYstd = math.sqrt(outside68x4ContYstd/outside68x4ContEnergy)

        if outside68x5ContEnergy > 0:
            outside68x5ContXmean /= outside68x5ContEnergy
            outside68x5ContYmean /= outside68x5ContEnergy
            outside68x5ContXstd = math.sqrt(outside68x5ContXstd/outside68x5ContEnergy)
            outside68x5ContYstd = math.sqrt(outside68x5ContYstd/outside68x5ContEnergy)


        for i in range(34):
            if outside68Totals[i] > 0:
                outside68Xmean[i] /= outside68Totals[i]
                outside68Ymean[i] /= outside68Totals[i]
                outside68Xstd[i] = math.sqrt(outside68Xstd[i]/outside68Totals[i])
                outside68Ystd[i] = math.sqrt(outside68Ystd[i]/outside68Totals[i])

        self.tree.trigEnergy = trigEnergy
        self.tree.ecalBackEnergy = ecalBackEnergy
        self.tree.ele68TotalEnergies = ele68Totals
        self.tree.photon68TotalEnergies = photon68Totals
        self.tree.overlap68TotalEnergies = overlap68Totals
        self.tree.outside68TotalEnergies = outside68Totals
        self.tree.outside68TotalNHits = outside68NHits
        self.tree.outside68Xmeans = outside68Xmean
        self.tree.outside68Ymeans = outside68Ymean
        self.tree.outside68Xstds = outside68Xstd
        self.tree.outside68Ystds = outside68Ystd
        self.tree.ele68ContEnergy = ele68ContEnergy
        self.tree.ele68x2ContEnergy = ele68x2ContEnergy
        self.tree.ele68x3ContEnergy = ele68x3ContEnergy
        self.tree.ele68x4ContEnergy = ele68x4ContEnergy
        self.tree.ele68x5ContEnergy = ele68x5ContEnergy
        self.tree.photon68ContEnergy = photon68ContEnergy
        self.tree.photon68x2ContEnergy = photon68x2ContEnergy
        self.tree.photon68x3ContEnergy = photon68x3ContEnergy
        self.tree.photon68x4ContEnergy = photon68x4ContEnergy
        self.tree.photon68x5ContEnergy = photon68x5ContEnergy
        self.tree.overlap68ContEnergy = overlap68ContEnergy
        self.tree.overlap68x2ContEnergy = overlap68x2ContEnergy
        self.tree.overlap68x3ContEnergy = overlap68x3ContEnergy
        self.tree.overlap68x4ContEnergy = overlap68x4ContEnergy
        self.tree.overlap68x5ContEnergy = overlap68x5ContEnergy
        self.tree.outside68ContEnergy = outside68ContEnergy
        self.tree.outside68x2ContEnergy = outside68x2ContEnergy
        self.tree.outside68x3ContEnergy = outside68x3ContEnergy
        self.tree.outside68x4ContEnergy = outside68x4ContEnergy
        self.tree.outside68x5ContEnergy = outside68x5ContEnergy
        self.tree.outside68ContNHits = outside68ContNHits
        self.tree.outside68x2ContNHits = outside68x2ContNHits
        self.tree.outside68x3ContNHits = outside68x3ContNHits
        self.tree.outside68x4ContNHits = outside68x4ContNHits
        self.tree.outside68x5ContNHits = outside68x5ContNHits
        self.tree.outside68ContXmean = outside68ContXmean
        self.tree.outside68x2ContXmean = outside68x2ContXmean
        self.tree.outside68x3ContXmean = outside68x3ContXmean
        self.tree.outside68x4ContXmean = outside68x4ContXmean
        self.tree.outside68x5ContXmean = outside68x5ContXmean
        self.tree.outside68ContYmean = outside68ContYmean
        self.tree.outside68x2ContYmean = outside68x2ContYmean
        self.tree.outside68x3ContYmean = outside68x3ContYmean
        self.tree.outside68x4ContYmean = outside68x4ContYmean
        self.tree.outside68x5ContYmean = outside68x5ContYmean
        self.tree.outside68ContXstd = outside68ContXstd
        self.tree.outside68x2ContXstd = outside68x2ContXstd
        self.tree.outside68x3ContXstd = outside68x3ContXstd
        self.tree.outside68x4ContXstd = outside68x4ContXstd
        self.tree.outside68x5ContXstd = outside68x5ContXstd
        self.tree.outside68ContYstd = outside68ContYstd
        self.tree.outside68x2ContYstd = outside68x2ContYstd
        self.tree.outside68x3ContYstd = outside68x3ContYstd
        self.tree.outside68x4ContYstd = outside68x4ContYstd
        self.tree.outside68x5ContYstd = outside68x5ContYstd
        self.tree.outside68ContShowerRMS = outside68ContShowerRMS
        self.tree.outside68x2ContShowerRMS = outside68x2ContShowerRMS
        self.tree.outside68x3ContShowerRMS = outside68x3ContShowerRMS
        self.tree.outside68x4ContShowerRMS = outside68x4ContShowerRMS
        self.tree.outside68x5ContShowerRMS = outside68x5ContShowerRMS

        # Fill the tree with values for this event
        self.tree.fill(reset=True)

    # Write and copy the output
    def end(self):
        self.tree.write()
        self.tfile.close()
        print 'cp %s %s' % (self.outfile,self.outdir)
        os.system('cp %s %s' % (self.outfile,self.outdir))



# Process command line arguments and run the tree maker
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make tree with BDT inputs and result')
    parser.add_argument('--swdir', dest='swdir',  default='/nfs/slac/g/ldmx/users/vdutta/ldmx-sw-1.7/ldmx-sw-install', help='ldmx-sw build directory')
    parser.add_argument('--signal', dest='issignal', action='store_true', help='Signal file [Default: False]')
    parser.add_argument('--interactive', dest='interactive', action='store_true', help='Run in interactive mode [Default: False]')
    parser.add_argument('-o','--outdir', dest='outdir', default='/nfs/slac/g/ldmx/users/vdutta/test', help='Name of output directory')
    parser.add_argument('-f','--outfile', dest='outfile', default='test.root', help='Name of output file')
    parser.add_argument('-i','--inputfiles', dest='inputfiles', nargs='*', default=['/nfs/slac/g/ldmx/data/mc/v9/4pt0_gev_e_ecal_pn_bdt_training/4pt0_gev_1e_ecal_pn_v5_20190507_00305a2c_tskim_recon.root'], help='List of input files')
    parser.add_argument('--filelist', dest='filelist', default = '', help='Text file with list of input files')

    args = parser.parse_args()

    r.gSystem.Load(args.swdir+'/lib/libEvent.so')

    
    inputfiles = []
    # If an input file list is provided, read from that
    if args.filelist != '':
        print 'Loading input files from',args.filelist
        with open(args.filelist,'r') as f:
            inputfiles = f.read().splitlines()
    else:
        inputfiles = args.inputfiles

    # Create output directory if it doesn't already exist
    print 'Creating output directory %s' % args.outdir
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
   
    # Create the scratch directory if it doesn't already exist
    scratch_dir = '%s/%s' % (os.getcwd(),os.environ['USER']) if args.interactive else '/scratch/%s' % os.environ['USER']
    print 'Using scratch path %s' % scratch_dir
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
  
    # Create a tmp directory that can be used to copy files into
    tmp_dir = '%s/%s' % (scratch_dir, 'tmp') if args.interactive else '%s/%s' % (scratch_dir, os.environ['LSB_JOBID'])
    print 'Creating tmp directory %s' % tmp_dir
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    os.chdir(tmp_dir)
   
    # Copy input files to the tmp directory
    print 'Copying input files into tmp directory'
    for f in inputfiles:
        os.system("cp %s ." % f )
    os.system("ls .")

    # Just get the file names without the full path
    localfiles = [f.split('/')[-1] for f in inputfiles]
  
    # Run the tree maker with these inputs
    treeMaker = bdtTreeMaker(args.outdir,args.outfile,localfiles,args.issignal)
    treeMaker.run()
    treeMaker.end()

    # Remove tmp directory
    print 'Removing tmp directory %s' % tmp_dir
    os.system('rm -rf %s' % tmp_dir)


