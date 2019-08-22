import math
import os
import argparse
import numpy as np
import ROOT as r
from rootpy.tree import Tree, TreeModel, TreeChain, FloatCol, IntCol, BoolCol, FloatArrayCol
from rootpy.io import root_open
#import rootpy.stl as stl
import math


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import csv

#WARNING WARNING WARNING:  Make sure this is actually being run on the sim input files!

adds = "sim_1000MeV_TMP"
savePath = "/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/LDMX-scripts/mipTracking/hits/"

# Will go in utils
cellMap = np.loadtxt('cellmodule.txt')
neighborList = np.loadtxt('/nfs/slac/g/ldmx/users/jbargem/neighborList.txt')
scoringPlaneZ = 220
ecalFaceZ = 223.8000030517578
cell_radius = 5
mcid = cellMap[:,0].tolist()

layerZs = [223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,
        266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,
        322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,
        375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,
        448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125]

zdict = {}
i = 0
for layerZ in layerZs:
    zdict.update({str(i) : layerZ})
    i += 1

layerWeights = [1.641, 3.526, 5.184, 6.841, 8.222, 8.775, 8.775, 8.775, 8.775, 8.775, 8.775, 8.775, 8.775, 8.775,
                8.775, 8.775, 8.775, 8.775, 8.775, 8.775, 8.775, 8.775, 12.642, 16.51, 16.51, 16.51, 16.51, 16.51, 16.51, 16.51, 16.51, 16.51, 16.51, 8.45]

weightdict = {}
i = 0
for weight in layerWeights:
    weightdict.update({str(i) : weight})
    i += 1

#radius = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
#4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
#9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
#17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
#30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
#55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
#119.06854141, 121.20048803, 127.5236134, 121.99024095]

radius_beam_68 = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]

#NOTE: RECOIL RADII HAVE NO LINEAR FITTING

radius_recoil_68_p_0_500_theta_0_10 = [4.133635240238758, 4.16812935640922, 4.4972890656145825, 4.874728621973265, 6.303886567828347, 7.03492411074243, 9.872717278770837, 11.974467146545866, 16.182039708653516, 18.520795512393878, 22.127158790924135, 24.08883427742839, 27.253557947119685, 28.773990194296754, 31.784424217923515, 33.08768884322611, 36.41675249882992, 37.741265082666004, 40.38597333415027, 42.10409079975839, 45.87406948906708, 46.27355705728598, 50.50927634209807, 51.13456941833934, 59.69044797003487, 58.53201386395721, 64.45411874165674, 69.14043466244003, 69.97562060995436, 95.5817513624476, 94.28874779330216, 118.1644810499474, 110.71356659554978, 156.05280837484543] 

radius_recoil_68_p_500_1500_theta_0_10 = [4.131769922612919, 4.14869550057709, 4.319549768431474, 4.402833154368089, 4.9395921122541475, 5.13192179115052, 6.454500157016699, 6.856947548635726, 8.652996754952376, 9.353494265202425, 11.631545494441484, 12.699443811273142, 15.43711152731991, 16.592818892889063, 19.88568649152623, 21.236409612363694, 24.433635236601134, 25.815018265917285, 29.539008362992053, 30.85521495489097, 34.63152769513493, 36.2698715072556, 40.798840419619495, 43.16669504462533, 47.7031557932884, 49.49979587657616, 54.28711701931053, 62.74928320843017, 64.91409823325857, 76.83949824874166, 80.31050281973198, 81.33239157335822, 91.72029450237251, 123.79621613269566] 

radius_recoil_68_theta_10_20 = [4.111304836250707, 4.149186172123573, 4.567238286530641, 5.31021268925882, 7.849683986578496, 9.970156496004812, 19.901973916081673, 29.96146972572831, 36.89590940133657, 40.41257794966759, 44.79662676710757, 47.347017542831345, 51.10413068333052, 53.33933211430637, 57.130192854436636, 58.28813559234106, 62.29715540362617, 64.6280231249923, 69.14235094987426, 70.41112183666935, 72.93326874773042, 76.43729989793896, 79.91585435631359, 81.9914496215926, 85.24481224667284, 86.28554614577115, 96.35140953371192, 99.57491222274099, 110.26506421339637, 133.85591445677275, 103.84487736936528, 113.33073246937222, 155.99455466012552, 184.00482766215032] 

radius_recoil_68_theta_20_end = [4.230182007240027, 4.414600123912876, 6.816404489046071, 9.937082469378527, 25.437104262936863, 71.90603036916038, 97.34880555008928, 112.16148848233995, 126.07245241030888, 136.06365161336078, 147.19244204848312, 155.60736966051817, 164.10362729713083, 170.99566124022613, 180.01725184344872, 185.0017381385014, 194.2724571497957, 203.41253245397303, 208.1007978177679, 218.30705568792524, 217.55392061113216, 221.33651851995552, 243.76569355219482, 232.72991284754758, 256.75724252717566, 249.55100102170172, 294.0271067439577, 264.2176237845432, 227.11674741267723, 262.3252867753147, 322.69504790428397, 365.6299292399247, 263.2823090542055, 248.91973635569462] 

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
    return p[2]/np.sqrt(p[0]**2+p[1]**2+p[2]**2)

def getMagnitude(p):
    return np.sqrt(p[0]**2+p[1]**2+p[2]**2)

# get scoring plane hits for electrons (both target and ECAL SP)
def getElectronSPHits(simParticles, spHits, targetSPHits):
    electronSPHits = []    
    for particle in simParticles:
        if particle.getPdgID() == 22 and particle.getParentCount() > 0 and particle.getParent(0).getPdgID()==11 and particle.getVertex()[2] < 1.2 and particle.getEnergy() > 2500:
            #print 'found photon'
            particle.Print()
            for hit in spHits:
                sp = hit.getSimParticle()
                if sp == particle and hit.getMomentum()[2] > 0 and hit.getLayerID()==1:
                    hit.Print()
       
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
def getElectronTrajectory(eventcount,pvec,pos,grouptag):
    positions = []
    layercount = 0
    for layerZ in layerZs:
        posX = pos[0] + pvec[0]/pvec[2]*(layerZ - pos[2])
        posY = pos[1] + pvec[1]/pvec[2]*(layerZ - pos[2])
        positions.append((posX,posY))
        hitInfo = [eventcount,posX,posY,layerZ,0,0,grouptag,999999]
	if layercount == 0 or layercount == 33:
	    with open(savePath + "hits_" + adds + ".txt", 'a') as hits:
	        writer = csv.writer(hits)
	        #print("in trajectory writing loop")    
		for entery in hitInfo:
		    hits.write(str(entery) + " ")
	        hits.write('\n')
	layercount += 1
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
    photon68ContEnergy = FloatCol()
    photon68x2ContEnergy = FloatCol()
    photon68x3ContEnergy = FloatCol()
    overlap68ContEnergy = FloatCol()
    overlap68x2ContEnergy = FloatCol()
    overlap68x3ContEnergy = FloatCol()
    outside68ContEnergy = FloatCol()
    outside68x2ContEnergy = FloatCol()
    outside68x3ContEnergy = FloatCol()
    outside68ContNHits = FloatCol()
    outside68x2ContNHits = FloatCol()
    outside68x3ContNHits = FloatCol()
    outside68ContXmean = FloatCol()
    outside68x2ContXmean = FloatCol()
    outside68x3ContXmean = FloatCol()
    outside68ContYmean = FloatCol()
    outside68x2ContYmean = FloatCol()
    outside68x3ContYmean = FloatCol()
    outside68ContXstd = FloatCol()
    outside68x2ContXstd = FloatCol()
    outside68x3ContXstd = FloatCol()
    outside68ContYstd = FloatCol()
    outside68x2ContYstd = FloatCol()
    outside68x3ContYstd = FloatCol()
    outside68ContShowerRMS = FloatCol()
    outside68x2ContShowerRMS = FloatCol()
    outside68x3ContShowerRMS = FloatCol()

    numtracks1 = IntCol()
    numtracks2 = IntCol()
    numtracks3 = IntCol()
    impactparameter1 = FloatCol()
    impactparameter2 = FloatCol()
    impactparameter3 = FloatCol()


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
        self.trigRes = r.TClonesArray('ldmx::TriggerResult')
	self.hcalhits = r.TClonesArray('ldmx::HcalHit') #added by Jack
	self.ecalhits = r.TClonesArray('ldmx::SimCalorimeterHit')
        self.simParticles = r.TClonesArray('ldmx::SimParticle')
        self.spHits = r.TClonesArray('ldmx::SimTrackerHit')
        self.targetSPHits = r.TClonesArray('ldmx::SimTrackerHit')

        self.intree.SetBranchAddress('EventHeader',r.AddressOf(self.evHeader))
        self.intree.SetBranchAddress('EcalVeto_recon',r.AddressOf(self.ecalVetoRes))
        self.intree.SetBranchAddress('EcalSimHits_sim',r.AddressOf(self.ecalhits)) 
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
    # Change number of events
    def run(self):

        #Create all files if they don't exist.  Overwrite them if they do.
        open(savePath + "hits_" + adds + ".txt",'w').close()
        open(savePath + "momenta_" + adds + ".txt", 'w').close()
        open(savePath + "particleinfo_" + adds + ".txt", 'w').close()

        while self.event_count < self.intree.GetEntries():
            self.intree.GetEntry(self.event_count)
            #print("NEW EVENT")
            if self.event_count>10000:
                return
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
            recoilangle = getTheta(pvec) 
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
            electronLayerIntercepts = getElectronTrajectory(self.event_count,pvec,pos,4)
            if pvec0 != [0,0,0]:
                photonLayerIntercepts = getElectronTrajectory(self.event_count,photonP,pos0,5)

            with open(savePath + "momenta_" + adds + ".txt",'a') as hits:
                writer = csv.writer(hits)
                hits.write(str(self.event_count) + " " + str(recoil_p) + " " + str(getMagnitude(photonP)))
                hits.write('\n')

        #print electronLayerIntercepts
        #print photonLayerIntercepts

        # compute new variables based on containment regions
        
        ele68Totals = np.zeros(34)
        ele68ContEnergy = 0
        ele68x2ContEnergy = 0
        ele68x3ContEnergy = 0
        photon68Totals = np.zeros(34)
        photon68ContEnergy = 0
        photon68x2ContEnergy = 0
        photon68x3ContEnergy = 0
        overlap68Totals = np.zeros(34)
        overlap68ContEnergy = 0
        overlap68x2ContEnergy = 0
        overlap68x3ContEnergy = 0
        outside68Totals = np.zeros(34)
        outside68ContEnergy = 0
        outside68x2ContEnergy = 0
        outside68x3ContEnergy = 0
        outside68NHits = np.zeros(34)
        outside68ContNHits = 0
        outside68x2ContNHits = 0
        outside68x3ContNHits = 0
        outside68Xmean = np.zeros(34)
        outside68ContXmean = 0
        outside68x2ContXmean = 0
        outside68x3ContXmean = 0
        outside68Ymean = np.zeros(34)
        outside68ContYmean = 0
        outside68x2ContYmean = 0
        outside68x3ContYmean = 0
        outside68Xstd = np.zeros(34)
        outside68ContXstd = 0
        outside68x2ContXstd = 0
        outside68x3ContXstd = 0
        outside68Ystd = np.zeros(34)
        outside68ContYstd = 0
        outside68x2ContYstd = 0
        outside68x3ContYstd = 0
        
        outside68HitPositions = []
        outside68x2HitPositions = []
        outside68x3HitPositions = []
       
        outside68WgtCentroidCoords = (0,0)
        outside68x2WgtCentroidCoords = (0,0)
        outside68x3WgtCentroidCoords = (0,0)
        trigEnergy = 0
        ecalBackEnergy = 0
        ir = -1
        if recoilangle==-1 or recoil_p==-1:
            ir = 0
        elif recoilangle<10 and recoil_p<500:
            ir = 1
        elif recoilangle<10 and recoil_p >= 500:
            ir = 2
        elif recoilangle<=20:
            ir = 3
        else:
            ir = 4
        ip = 0

        hitcounter = 0  #Keep track of which hit (within the event) is being considered
        particleList = []  #Store list of unique particle objects
        pdgList = []  #PDG for each particle
        hitLists = []  #List of hits corresp to each particle

        for hit in self.ecalhits:
            cell, module, layer = mask(hit.getID())
	    hitE = ((hit.getEdep()/0.130)*weightdict[str(layer)]+hit.getEdep())
            if not hitE > 0:
                continue
            layerZ = zdict[str(layer)]
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
            """
            if distanceEle < radius_68[ir][layer] and distanceEle>0:
                ele68Totals[layer] += hitE
                ele68ContEnergy += hitE
            elif distanceEle >= radius_68[ir][layer] and distanceEle < 2*radius_68[ir][layer]:
                ele68x2ContEnergy += hitE
            elif distanceEle >= 2*radius_68[ir][layer] and distanceEle < 3*radius_68[ir][layer]:
                ele68x3ContEnergy += hitE
            if distancePhoton < radius_68[ip][layer] and distancePhoton>0:
                photon68Totals[layer] += hitE
                photon68ContEnergy += hitE
            elif distancePhoton >= radius_68[ip][layer] and distancePhoton  < 2*radius_68[ip][layer]:
                photon68x2ContEnergy += hitE
            elif distancePhoton >= 2*radius_68[ip][layer] and distancePhoton  < 3*radius_68[ip][layer]:
                photon68x3ContEnergy += hitE
            if distanceEle < radius_68[ir][layer] and distancePhoton < radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68Totals[layer] += hitE
                overlap68ContEnergy += hitE
            if distanceEle < 2*radius_68[ir][layer] and distancePhoton < 2*radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68x2ContEnergy += hitE
            if distanceEle < 3*radius_68[ir][layer] and distancePhoton < 3*radius_68[ip][layer] and distanceEle>0 and distancePhoton>0:
                overlap68x3ContEnergy += hitE
            """
	    #outside EP radii
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
                outside68HitPositions.append((hitX,hitY,layer,hitE,mcid_val))
		hitInfo = [self.event_count,hitX,hitY,layerZ,hitE,mcid_val,0,hitcounter]
		with open(savePath + "hits_" + adds + ".txt",'a') as hits:
		    writer = csv.writer(hits)
		    #print("in hit writing loop")    
		    for entery in hitInfo:
			hits.write(str(entery) + " ")
		    hits.write('\n')
	    #in E radius only
	    if distanceEle < radius_68[ir][layer] and distancePhoton > radius_68[ip][layer]:
                outside68Totals[layer] += hitE
                outside68NHits[layer] += 1
                outside68Xmean[layer] += hitX*hitE
                outside68Ymean[layer] += hitY*hitE
                outside68ContEnergy += hitE
                outside68ContNHits += 1
                outside68ContXmean += hitX*hitE
                outside68ContYmean += hitY*hitE
                outside68WgtCentroidCoords = (outside68WgtCentroidCoords[0] + hitX*hitE, outside68WgtCentroidCoords[1] + hitY*hitE)
                outside68HitPositions.append((hitX,hitY,layer,hitE,mcid_val))
		hitInfo = [self.event_count,hitX,hitY,layerZ,hitE,mcid_val,1,hitcounter]
		with open(savePath + "hits_" + adds + ".txt",'a') as hits:
		    writer = csv.writer(hits)
		    for entery in hitInfo:    
			hits.write(str(entery) + " ")
		    hits.write('\n')
	    #in P radius only
	    if distanceEle > radius_68[ir][layer] and distancePhoton < radius_68[ip][layer]:
                outside68Totals[layer] += hitE
                outside68NHits[layer] += 1
                outside68Xmean[layer] += hitX*hitE
                outside68Ymean[layer] += hitY*hitE
                outside68ContEnergy += hitE
                outside68ContNHits += 1
                outside68ContXmean += hitX*hitE
                outside68ContYmean += hitY*hitE
                outside68WgtCentroidCoords = (outside68WgtCentroidCoords[0] + hitX*hitE, outside68WgtCentroidCoords[1] + hitY*hitE)
                outside68HitPositions.append((hitX,hitY,layer,hitE,mcid_val))
		hitInfo = [self.event_count,hitX,hitY,layerZ,hitE,mcid_val,2,hitcounter]
		with open(savePath + "hits_" + adds + ".txt", 'a') as hits:
		    writer = csv.writer(hits)
		    for entery in hitInfo:    
			hits.write(str(entery) + " ")
		    hits.write('\n')
	    #in EP radii
	    if distanceEle < radius_68[ir][layer] and distancePhoton < radius_68[ip][layer]:
                outside68Totals[layer] += hitE
                outside68NHits[layer] += 1
                outside68Xmean[layer] += hitX*hitE
                outside68Ymean[layer] += hitY*hitE
                outside68ContEnergy += hitE
                outside68ContNHits += 1
                outside68ContXmean += hitX*hitE
                outside68ContYmean += hitY*hitE
                outside68WgtCentroidCoords = (outside68WgtCentroidCoords[0] + hitX*hitE, outside68WgtCentroidCoords[1] + hitY*hitE)
                outside68HitPositions.append((hitX,hitY,layer,hitE,mcid_val))
		hitInfo = [self.event_count,hitX,hitY,layerZ,hitE,mcid_val,3,hitcounter]
		with open(savePath + "hits_" + adds + ".txt",'a') as hits:
		    writer = csv.writer(hits)
		    for entery in hitInfo:    
			hits.write(str(entery) + " ")
		    hits.write('\n')



            #INSERT NEW PARENT CODE HERE!!

            #for each particle in contribs:
            #    check to see whether it's already in particleList
            #    if not:  add to particleList, add pdg, add [] to hitList
            #    add hit to corresponding hitList
            #NOTE:  We only need charged particles, so ignore anything that's neutral.  Should resolve issue with unphysical tracks/one neutron showing up as a shower.

            for ic in range(0, hit.getNumberOfContribs()):
                cntrb = hit.getContrib(ic)
                particle = cntrb.particle
                newParticle = True
                if particle.getCharge() == 0:  continue  #Ignore neutral particles
                for part in particleList:
                    if part == particle:  newParticle = False
                #if particle.getPdgID() == 22:  print("PHOTON")
                if newParticle:
                    #print("  NEW PARTICLE")
                    particle_ = particle
                    while particle_.getParentCount() > 0:
                        #NOTE:  For signal, this info is questionably reliable.  Included mainly for formatting purposes.
                        #if particle_.getPdgID() == 22 and (particle_.getParent(0)).getPdgID()==11 and particle_.getParent(0).getParentCount() == 0:  break
                        particle_ = particle_.getParent(0)
                    #print("    final pdgID = "+str(particle_.getPdgID()))
                    originPDGID = particle_.getPdgID()

                    particleList.append(particle)
                    pdgList.append(particle.getPdgID())
                    hitLists.append([originPDGID])
                hitLists[particleList.index(particle)].append(hitcounter)

            hitcounter += 1

            """originList = []
            partList = []
            for ic in range(0, hit.getNumberOfContribs()):
                particle = (hit.getContrib(ic)).particle
                pdgID = particle.getPdgID()
                partList.append(pdgID)
                if particle.getParentCount() == 0:
                    parentPDGID = pdgID
                else:
                    parentPDGID = particle.getParent(0).getPdgID()
                while particle.getParentCount() > 0:
                    if particle.getPdgID() == 22 and abs((particle.getParent(0)).getPdgID())==11 and particle.getParent(0).getParentCount() == 0:  break
                    particle = particle.getParent(0)
                originPDGID = particle.getPdgID()
                originList.append(originPDGID)
            with open(savePath + "parts_" + adds + ".txt",'a') as parts:
                writer = csv.writer(parts)
                for entery in partList:
                    parts.write(str(entery) + " ")
                parts.write('\n')
            with open(savePath + "parents_" + adds + ".txt",'a') as parents:
                writer = csv.writer(parents)
                for entery in originList:
                    parents.write(str(entery) + " ")
                parents.write('\n')
            """


	    """
	    if distanceEle > 2*radius_68[ir][layer] and distancePhoton > 2*radius_68[ip][layer]:
                outside68x2ContEnergy += hitE
                outside68x2ContNHits += 1
                outside68x2ContXmean += hitX*hitE
                outside68x2ContYmean += hitY*hitE
                outside68x2WgtCentroidCoords = (outside68x2WgtCentroidCoords[0] + hitX*hitE, outside68x2WgtCentroidCoords[1] + hitY*hitE)
                outside68x2HitPositions.append((hitX,hitY,layer,hitE,mcid_val))
            if distanceEle > 3*radius_68[ir][layer] and distancePhoton > 3*radius_68[ip][layer]:
                outside68x3ContEnergy += hitE
                outside68x3ContNHits += 1
                outside68x3ContXmean += hitX*hitE
                outside68x3ContYmean += hitY*hitE
                outside68x3WgtCentroidCoords = (outside68x3WgtCentroidCoords[0] + hitX*hitE, outside68x3WgtCentroidCoords[1] + hitY*hitE)
                outside68x3HitPositions.append((hitX,hitY,layer,hitE,mcid_val))
		hitInfo = [self.event_count,hitX,hitY,layer,hitE,mcid_val)]
		with open("hits.txt") as hits:
		    csv_reader = csv.reader(csv_file, delimiter=' ')
		    hits.write(str(hitInfo))
		    hits.write(" ")
		    hits.write('\n')
		    if self.event_count%1000 == 0: 
		        print(str(hitInfo))
	    """	
        #Finished loading hit/particle/parent info into arrays, so print it to txt file:
        with open(savePath + "particleinfo_" + adds + ".txt",'a') as prts:
            writer = csv.writer(prts)
            for part in range(len(particleList)):
                prts.write(str(self.event_count) + " ")
                prts.write(str(pdgList[part]) + " ")
                for hitNum in hitLists[part]:
                    prts.write(str(hitNum) + " ")
                prts.write('\n')





        outside68ContShowerRMS = 0
        # mean and std deviations of hits outside containment regions
        """
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


        for i in range(34):
            if outside68Totals[i] > 0:
                outside68Xmean[i] /= outside68Totals[i]
                outside68Ymean[i] /= outside68Totals[i]
                outside68Xstd[i] = math.sqrt(outside68Xstd[i]/outside68Totals[i])
                outside68Ystd[i] = math.sqrt(outside68Ystd[i]/outside68Totals[i])
	"""

        #tracking
        #track1     loose def
        #track2     isolated end
        #track3     all isolated
	"""
        possibleTrack = True
        tracklist1 = []
        tracklist2 = []
        tracklist3 = []
        if len(outside68HitPositions)<1:
          possibleTrack = False
        while possibleTrack==True:
          possibleTrack=False
          outsideHitCopy = outside68HitPositions
          currentmcid = outsideHitCopy[0][4]
          currentlayer = outside68HitPositions[0][2]
          tracknumber = 0


          for hit3 in outsideHitCopy:
           if isolatedend(hit3[4],hit3[2],outside68HitPositions):

            track2 = []
            track2.append(hit3[2]*10000+hit3[4])
            newtrack2 = checkforhits(hit3[4],hit3[2],outside68HitPositions,track2,False)
            if len(newtrack2)>=5:
              tracklist2.append(newtrack2)

            track3 = []
            track3.append(hit3[2]*10000+hit3[4])
            newtrack3 = checkforhits(hit3[4],hit3[2],outside68HitPositions,track3,True)
            if len(newtrack3)>=5:
              tracklist3.append(newtrack3)

           track1 = []
           track1.append(hit3[2]*10000+hit3[4])
           newtrack1 = checkforhits(hit3[4],hit3[2],outside68HitPositions,track1,False)
           if len(newtrack1)>=5:
             tracklist1.append(newtrack1)
              #print newtrack
        #print tracklist1
        #print tracklist3,"," 
	
	"""
	"""

        mindist1 = 10000
        if len(tracklist1)>0: 
          for track in tracklist1:
            trackpos = gettrackpositions(track)
            for layer in range(34):
              dist = math.sqrt((photonLayerIntercepts[layer][0]-trackpos[layer][0])**2+(photonLayerIntercepts[layer][1]-trackpos[layer][1])**2)/radius_beam_68[layer]
              if dist < mindist1:
                mindist1 = dist
          self.tree.impactparameter1 = mindist1 
        
        mindist2 = 10000
        if len(tracklist2)>0: 
          for track in tracklist2:
            trackpos = gettrackpositions(track)
            for layer in range(34):
              dist = math.sqrt((photonLayerIntercepts[layer][0]-trackpos[layer][0])**2+(photonLayerIntercepts[layer][1]-trackpos[layer][1])**2)/radius_beam_68[layer]
              if dist < mindist2:
                mindist2 = dist
          self.tree.impactparameter2 = mindist2
 
        mindist3 = 10000
        if len(tracklist3)>0: 
          for track in tracklist3:
            trackpos = gettrackpositions(track)
            for layer in range(34):
              dist = math.sqrt((photonLayerIntercepts[layer][0]-trackpos[layer][0])**2+(photonLayerIntercepts[layer][1]-trackpos[layer][1])**2)/radius_beam_68[layer]
              if dist < mindist3:
                mindist3 = dist
          self.tree.impactparameter3 = mindist3 

        #%matplotlib inline
        #plt.hist(tracklist, normed=True, bins=30)

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
        self.tree.photon68ContEnergy = photon68ContEnergy
        self.tree.photon68x2ContEnergy = photon68x2ContEnergy
        self.tree.photon68x3ContEnergy = photon68x3ContEnergy
        self.tree.overlap68ContEnergy = overlap68ContEnergy
        self.tree.overlap68x2ContEnergy = overlap68x2ContEnergy
        self.tree.overlap68x3ContEnergy = overlap68x3ContEnergy
        self.tree.outside68ContEnergy = outside68ContEnergy
        self.tree.outside68x2ContEnergy = outside68x2ContEnergy
        self.tree.outside68x3ContEnergy = outside68x3ContEnergy
        self.tree.outside68ContNHits = outside68ContNHits
        self.tree.outside68x2ContNHits = outside68x2ContNHits
        self.tree.outside68x3ContNHits = outside68x3ContNHits
        self.tree.outside68ContXmean = outside68ContXmean
        self.tree.outside68x2ContXmean = outside68x2ContXmean
        self.tree.outside68x3ContXmean = outside68x3ContXmean
        self.tree.outside68ContYmean = outside68ContYmean
        self.tree.outside68x2ContYmean = outside68x2ContYmean
        self.tree.outside68x3ContYmean = outside68x3ContYmean
        self.tree.outside68ContXstd = outside68ContXstd
        self.tree.outside68x2ContXstd = outside68x2ContXstd
        self.tree.outside68x3ContXstd = outside68x3ContXstd
        self.tree.outside68ContYstd = outside68ContYstd
        self.tree.outside68x2ContYstd = outside68x2ContYstd
        self.tree.outside68x3ContYstd = outside68x3ContYstd
        self.tree.outside68ContShowerRMS = outside68ContShowerRMS
        self.tree.outside68x2ContShowerRMS = outside68x2ContShowerRMS
        self.tree.outside68x3ContShowerRMS = outside68x3ContShowerRMS

        self.tree.numtracks1 = len(tracklist1)
        self.tree.numtracks2 = len(tracklist2)
        self.tree.numtracks3 = len(tracklist3)
        #self.tree.impactparameter1 = mindist1 
        #self.tree.impactparameter2 = mindist2 
        #self.tree.impactparameter3 = mindist3 
	"""

	
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
    parser.add_argument('--swdir', dest='swdir',  default='/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/ldmx-sw-install', help='ldmx-sw build directory')
    #parser.add_argument('--swdir', dest='swdir',  default='/nfs/slac/g/ldmx/users/vdutta/ldmx-sw/install', help='ldmx-sw build directory')
    parser.add_argument('--signal', dest='issignal', action='store_true', help='Signal file [Default: False]')
    parser.add_argument('--interactive', dest='interactive', action='store_true', help='Run in interactive mode [Default: False]')
    parser.add_argument('-o','--outdir', dest='outdir', default='/nfs/slac/g/ldmx/users/jbargem/ldmx-sw/ldmx-sw-install/test', help='Name of output directory')
    parser.add_argument('-f','--outfile', dest='outfile', default='test.root', help='Name of output file')
    parser.add_argument('-i','--inputfiles', dest='inputfiles', nargs='*', default=['/nfs/slac/g/ldmx/data/mc/v9-magnet/ucsb_test/4pt0_gev_1e_ecal_pn_v9_magnet_test_0824c660_108198_tskim_recon.root'], help='List of input files')
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

