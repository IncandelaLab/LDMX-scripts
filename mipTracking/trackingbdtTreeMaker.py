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

radius = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]


def checkforhits(mcid,layer,hitlist,track):
  for hit in hitlist:
    hitlayer = hit[2]
    hitmcid = hit[4]
    if hitlayer == layer+1:
      for id in neighborList[mcid]:
        if hitmcid == id:
          track.append(10000*hitlayer+hitmcid)
          return checkforhits(hitmcid,hitlayer,hitlist,track)
  return track

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
            #isphit =0
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
    recoilPx = FloatCol()
    recoilPy = FloatCol()
    recoilPz = FloatCol()
    recoilX = FloatCol()
    recoilY = FloatCol()
    fiducial = BoolCol()
    hcalenergy = FloatCol()    
    leadhadpid = IntCol()
    leadhadke = FloatCol()
    leadhadthetaz = FloatCol()
    nelectrons = IntCol()
    ele0P = FloatArrayCol(3)
    ele1P = FloatArrayCol(3)
    ele0PosSP = FloatArrayCol(3)
    ele1PosSP = FloatArrayCol(3)
    trigEnergy = FloatCol()
    ele0TotalEnergies = FloatArrayCol(34)
    ele1TotalEnergies = FloatArrayCol(34)
    photonTotalEnergies = FloatArrayCol(34)
    overlapTotalEnergies = FloatArrayCol(34)
    overlapPlusPhotonTotalEnergies = FloatArrayCol(34)
    outsideTotalEnergies = FloatArrayCol(34)
    outsideTotalNHits = FloatArrayCol(34)
    outsideMinusPhotonTotalEnergies = FloatArrayCol(34)
    outsideMinusPhotonTotalNHits = FloatArrayCol(34)
    outsideXmeans = FloatArrayCol(34)
    outsideYmeans = FloatArrayCol(34)
    outsideXstds = FloatArrayCol(34)
    outsideYstds = FloatArrayCol(34)

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
        self.ecalhits = r.TClonesArray('ldmx::EcalHit')
        self.simParticles = r.TClonesArray('ldmx::SimParticle')
        self.spHits = r.TClonesArray('ldmx::SimTrackerHit')
        self.targetSPHits = r.TClonesArray('ldmx::SimTrackerHit')

        self.intree.SetBranchAddress('EventHeader',r.AddressOf(self.evHeader))
        self.intree.SetBranchAddress('EcalVeto_recon',r.AddressOf(self.ecalVetoRes))
        self.intree.SetBranchAddress('ecalDigis_recon',r.AddressOf(self.ecalhits)) 
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
        while self.event_count < 10:
        #while self.event_count < self.intree.GetEntries():
            self.intree.GetEntry(self.event_count)
            if self.event_count%1000 == 0:
                print 'Processing event ',self.event_count
            #if self.event_count>10:
            #    return
            self.process()

    # Process an event
    def process(self):
        self.event_count += 1

        self.tree.evtNum = self.evHeader.getEventNumber()
        #self.tree.trigPass = 1 if self.issignal else self.trigRes[0].passed()

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

        # find KE and angle of hardest hadron
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

        # find electrons at scoring plane
        electronsAtSP = getElectronSPHits(self.simParticles, self.spHits, self.targetSPHits)

        self.tree.nelectrons = len(electronsAtSP)

        # get positions at which electron and photon trajectories interesect with each layer
        # NB: ordering of beam and recoil electrons below not guaranteed to be correct! May need to additionally require recoil e energy < 1.5 GeV
        electronLayerIntercepts = []
        photonLayerIntercepts = []
        iele = 0
        for electron in electronsAtSP:
            ecalSPHit = electron[0]
            targetSPHit = electron[1]
            simParticle = electron[2]
            pvec = ecalSPHit.getMomentum()
            pos = ecalSPHit.getPosition()
            pvec0 = [0,0,0]
            pos0 = [0,0,0]
            if self.issignal:
                # beam electron
                if iele == 0:
                    self.tree.ele1P = [pvec[0],pvec[1],pvec[2]]
                    self.tree.ele1PosSP = [pos[0],pos[1],pos[2]]
                # recoil electron
                elif iele == 1:
                    self.tree.ele0P = [pvec[0],pvec[1],pvec[2]]
                    self.tree.ele0PosSP = [pos[0],pos[1],pos[2]]
                    pvec0 = targetSPHit.getMomentum() if targetSPHit != None else [0,0,0]
                    pos0 = targetSPHit.getPosition() if targetSPHit != None else [0,0,0]
                    photonP = [-pvec0[0],-pvec0[1],4000.0-pvec0[2]]
                    #print 'Photon momentum: ',photonP
            else:
                # recoil electron
                if iele == 0:
                    self.tree.ele0P = [pvec[0],pvec[1],pvec[2]]
                    self.tree.ele0PosSP = [pos[0],pos[1],pos[2]]
                    pvec0 = targetSPHit.getMomentum() if targetSPHit != None else [0,0,0]
                    pos0 = targetSPHit.getPosition() if targetSPHit != None else [0,0,0]
                    photonP = [-pvec0[0],-pvec0[1],4000.0-pvec0[2]]
                    #print 'Photon momentum: ',photonP
                # beam electron
                elif iele == 1:
                    self.tree.ele1P = [pvec[0],pvec[1],pvec[2]]
                    self.tree.ele1PosSP = [pos[0],pos[1],pos[2]]
            #print pvec[0],pvec[1],pvec[2],pos[0],pos[1],pos[2]
            positions = getElectronTrajectory(pvec,pos)
            electronLayerIntercepts.append(positions)
            if photonP != [0,0,0]:
                photonLayerIntercepts = getElectronTrajectory(photonP,pos0)
            iele += 1
        #print electronLayerIntercepts
        #print photonLayerIntercepts

        # compute new variables based on containment regions
        ele0Totals = np.zeros(34)
        ele1Totals = np.zeros(34)
        photonTotals = np.zeros(34)
        overlapTotals = np.zeros(34)
        overlapPlusPhotonTotals = np.zeros(34)
        outsideTotals = np.zeros(34)
        outsideMinusPhotonTotals = np.zeros(34)
        outsideNHits = np.zeros(34)
        outsideMinusPhotonNHits = np.zeros(34)
        outsideXmean = np.zeros(34)
        outsideYmean = np.zeros(34)
        outsideXstd = np.zeros(34)
        outsideYstd = np.zeros(34)
        outsideHitPositions = []
        trigEnergy = 0
        for hit in self.ecalhits:
            hitE = hit.getEnergy()
            if not hitE > 0:
                continue
            cell, module, layer = mask(hit.getID())
            if layer<20:
                trigEnergy += hitE
            #print 'layer:',layer,hit.getLayer()
            mcid_val = 10*cell + module
            hitX = cellMap[mcid.index(mcid_val)][1]
            hitY = cellMap[mcid.index(mcid_val)][2]
            # get distances of hit from projected positions of electrons and photon
            if self.issignal:
                distanceEle0 = math.sqrt((hitX-electronLayerIntercepts[1][layer][0])**2 + (hitY-electronLayerIntercepts[1][layer][1])**2) if len(electronLayerIntercepts) > 1 else -1
                distanceEle1 = math.sqrt((hitX-electronLayerIntercepts[0][layer][0])**2 + (hitY-electronLayerIntercepts[0][layer][1])**2) if len(electronLayerIntercepts) > 0 else -1
            else:
                distanceEle0 = math.sqrt((hitX-electronLayerIntercepts[0][layer][0])**2 + (hitY-electronLayerIntercepts[0][layer][1])**2) if len(electronLayerIntercepts) > 0 else -1
                distanceEle1 = math.sqrt((hitX-electronLayerIntercepts[1][layer][0])**2 + (hitY-electronLayerIntercepts[1][layer][1])**2) if len(electronLayerIntercepts) > 1 else -1
            distancePhoton = math.sqrt((hitX-photonLayerIntercepts[layer][0])**2 + (hitY-photonLayerIntercepts[layer][1])**2) if len(photonLayerIntercepts) > 0 else -1
            # add to containment totals depending on distance relative to containment radii
            if distanceEle0 < radius[layer] and distanceEle0>0:
                ele0Totals[layer] += hitE
            if distanceEle1 < radius[layer] and distanceEle1>0:
                ele1Totals[layer] += hitE
            if distancePhoton < radius[layer] and distancePhoton>0:
                photonTotals[layer] += hitE
            if distanceEle0 < radius[layer] and distanceEle1 < radius[layer] and distanceEle0>0 and distanceEle1>0:
                overlapTotals[layer] += hitE
                if distancePhoton < radius[layer] and distancePhoton>0:
                    overlapPlusPhotonTotals += hitE
            if distanceEle0 > radius[layer] and distanceEle1 > radius[layer]:
                outsideTotals[layer] += hitE
                outsideNHits[layer] += 1
                outsideXmean[layer] += hitX*hitE
                outsideYmean[layer] += hitY*hitE
                outsideHitPositions.append((hitX,hitY,layer,hitE,mcid_val))
                if distancePhoton > radius[layer]:
                    outsideMinusPhotonTotals[layer] += hitE
                    outsideMinusPhotonNHits[layer] += 1

        # mean and std deviations of hits outside containment regions
        for hit in outsideHitPositions:
            layer = hit[2]
            if outsideTotals[layer] > 0:
                outsideXstd[layer] += ((hit[0] - (outsideXmean[layer]/outsideTotals[layer]))**2)*hit[3]
                outsideYstd[layer] += ((hit[1] - (outsideYmean[layer]/outsideTotals[layer]))**2)*hit[3]

        #tracking
        possibleTrack = True
        tracklist = []
        if len(outsideHitPositions)<1:
          possibleTrack = False
        while possibleTrack==True:
          possibleTrack=False
          outsideHitCopy = outsideHitPositions
          currentmcid = outsideHitCopy[0][4]
          currentlayer = outsideHitPositions[0][2]
          tracknumber = 0


          for hit3 in outsideHitCopy:
            track = []
            track.append(hit3[2]*10000+hit3[4])
            newtrack = checkforhits(hit3[4],hit3[2],outsideHitPositions,track)
            if len(newtrack)>=2:
              tracklist.append(newtrack)             
              #print newtrack
        print len(tracklist)     
 



        for i in range(34):
            if outsideTotals[i] > 0:
                outsideXmean[i] /= outsideTotals[i]
                outsideYmean[i] /= outsideTotals[i]
                outsideXstd[i] = math.sqrt(outsideXstd[i]/outsideTotals[i])
                outsideYstd[i] = math.sqrt(outsideYstd[i]/outsideTotals[i])

        self.tree.trigEnergy = trigEnergy
        self.tree.ele0TotalEnergies = ele0Totals
        self.tree.ele1TotalEnergies = ele1Totals
        self.tree.photonTotalEnergies = photonTotals
        self.tree.overlapTotalEnergies = overlapTotals
        self.tree.overlapPlusPhotonTotalEnergies = overlapPlusPhotonTotals
        self.tree.outsideTotalEnergies = outsideTotals
        self.tree.outsideTotalNHits = outsideNHits
        self.tree.outsideMinusPhotonTotalEnergies = outsideMinusPhotonTotals
        self.tree.outsideMinusPhotonTotalNHits = outsideMinusPhotonNHits
        self.tree.outsideXmeans = outsideXmean
        self.tree.outsideYmeans = outsideYmean
        self.tree.outsideXstds = outsideXstd
        self.tree.outsideYstds = outsideYstd

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
    parser.add_argument('--swdir', dest='swdir',  default='/nfs/slac/g/ldmx/users/vdutta/ldmx-sw/install', help='ldmx-sw build directory')
    parser.add_argument('--signal', dest='issignal', action='store_true', help='Signal file [Default: False]')
    parser.add_argument('--interactive', dest='interactive', action='store_true', help='Run in interactive mode [Default: False]')
    parser.add_argument('-o','--outdir', dest='outdir', default='/nfs/slac/g/ldmx/users/vdutta/ldmx-sw/scripts/test', help='Name of output directory')
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


