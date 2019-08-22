import numpy as np
#import statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import colorsys

#Implementation of MIP tracking code
#inradii version:  Look inside the electron radius of containment for cases where the electron
#  is extremely close to/almost parallel to the photon.
#Also has removed isolatedEnd().

visualize = True  #Plot each individual event?

min_track_len = 5  #Minimum length required for valid tracks
min_straight_len = 4  #Minimum length required for valid straight tracks
filePath = '/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/LDMX-scripts/mipTracking/tracking_dev/'
showHists = True  #Create histos after full analysis
saveHists = False  #Save histos instead of plotting them, mutually incompatible with show

bkg_file = 'hits_sim_bkg_10k.txt'
bkg_parent_file = 'particleinfo_sim_bkg_10k.txt'
bkg_momentum_file = 'momenta_sim_bkg_10k.txt'
bkgFiles = [bkg_file, bkg_parent_file, bkg_momentum_file]
bkgOn = True  #Run the algorithm on background files

sig_file = 'hits_sim_1000MeV_1k.txt'
sig_parent_file = 'particleinfo_sim_1000MeV_1k.txt'
sig_momentum_file = 'momenta_sim_1000MeV_1k.txt'
sigFiles = [sig_file, sig_parent_file, sig_momentum_file]
sigOn = True  #Run the algorithm on signal files

unv_file = 'hits_sim_unvet_1k.txt'
unv_parent_file = 'particleinfo_sim_unvet_1k.txt'
unv_momentum_file = 'momenta_sim_unvet_1k.txt'
unvFiles = [unv_file, unv_parent_file, unv_momentum_file]
unvOn = False  #Do not run the algorithm on unvetoed events

filesArr = [bkgFiles, sigFiles, unvFiles]
runList = [bkgOn,sigOn,unvOn]

adds = "isoEnd_tracks"
qualifier = "without isoEnd"

#INITIALIZING LAYER GEOMETRY INFO:

layerZs_full = [223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,
        266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,
        322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,
        375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,
        448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125]

layerZs = [round(i) for i in layerZs_full]  #Round to nearest int

layer_intervals=[]
for l in range(len(layerZs_full)-1):
    layer_intervals.append(round(layerZs[l+1])-round(layerZs[l]))
layer_double_intervals=[]
for l in range(32): #32 of the 34 layer have two layers ahead on them
    layer_double_intervals.append(round(layerZs[l+2])-round(layerZs[l]))
#print(layer_double_intervals)

#FUNCTIONS USED:

def neighbors(h1,h2,mult):
    #sqrt(76)mm ~ 8mm = neighbors are within one cell.  mult=2 -> within 2 cells, etc.
    #Note:  This only compares x and y coordinates.  Two hits in different layers w/ identical xy coords will always be neighbors.
    return np.sqrt((h1[1]-h2[1])**2 + (h1[2]-h2[2])**2) < 8.7*mult

#Check whether h2 is a layer in front of h1 and a neighbor of h1; if so, return true
#remembervar is in case nINL needs to "remember" previous results, e.g. whether a hop has been used in the track
def neighInNextLayer(h1,h2,remembervar):
    #If h2 is a neighbor(,,2) and 2 layers away, accept it, but return mustBeLong
    """
    printing=False
    if h2[0]==1301 or h2[0]==1300:  printing=True
    if printing:
        if neighbors(h1,h2,1):  print("First neighbors")
        if neighbors(h1,h2,2):  print("Second neighbors")
        if neighbors(h1,h2,3):  print("Third neighbors")
    """
    rememberOut = []
    hopUsed=False
    inNearestLayer=False
    inSameLayer=False
    reject = False
    if h1[3]>228:  #If not in the front two layers...
        hopUsed = h1[3]-layer_double_intervals[layerZs.index(h1[3])-2]==h2[3] and neighbors(h1,h2,2)
    if h1[3]>225:  #If not in the front layer...
        inNearestLayer = h1[3]-layer_intervals[layerZs.index(h1[3])-1]==h2[3] and neighbors(h1,h2,2)
    #if printing:  print("Truth vals are "+str(inNearestLayer)+", "+str(hopUsed)+", "+str(doublehop)+", "+str(triplehop))
    #inSameLayer = h1[3]==h2[3] and neighbors(h1,h2,1)

    #Remember slope of previous jump
    posdist = np.sqrt((h1[1]-h2[1])**2 + (h1[2]-h2[2])**2 + (h1[3]-h2[3])**2)
    current_slope = [(h1[i] - h2[i])/posdist for i in range(1,4)]
    #If the current slope differs too much from the old slope, reject it.  Currently considering a 45 degree cone.
    dot = -2
    if remembervar != [] and (inNearestLayer or hopUsed): # or inSameLayer):  # or doublehop or triplehop):
        dot = np.dot(current_slope, remembervar)
	if dot < .75:
            reject = True
    if (inNearestLayer or hopUsed) and not reject:  #NOTE temporarily taking out inSameLayer for first version of ambig
        rememberOut = current_slope
        return True, rememberOut, dot
    else:  return False, rememberOut, dot

#Returns true iff isolatedEnd has been found
def isolatedEnd(hit,hitlist):
    for h in hitlist:
        if h==hit:  continue
        #far_neighbor = False
        #if hit[3]<492:  #If not at the back of the ecal:
        #    if hit[3]-layer_intervals[layerZs.index(hit[3])+1]==h[3] and neighbors(hit,h,1):  far_neighbor=True
        if hit[3]<502:
            if ((h[3]==hit[3] or hit[3]-layer_intervals[layerZs.index(hit[3])]==h[3]) \
                    and neighbors(hit,h,1)):   # or far_neighbor:
                return False
        #else:  return True, since it's at the back of the ecal 
    return True

#Convert a list of 5=n-tuples (i,x,y,z...) to a np array
def convertListToArr(lst):
    if len(lst)==0:  return []
    xdim=len(lst)
    ydim=len(lst[0])
    data=np.zeros((xdim,ydim))
    for i in range(xdim):
        for j in range(ydim):
            data[i,j]=lst[i][j]
    return data

def createHistoArr(trkArr):
    trkLens=[]
    for trklist in trkArr:
        for trk in trklist:
            trkLens.append(len(trk))
    return trkLens

def distance(h1, h2):
    return np.sqrt((h1[0]-h2[0])**2+(h1[1]-h2[1])**2+(h1[2]-h2[2])**2)

def hitDistance(h1, h2):
    return np.sqrt((h1[1]-h2[1])**2+(h1[2]-h2[2])**2+(h1[3]-h2[3])**2)

#Min distance from pt h1 to line specified by points p1 and p2--see point-line distance 3-dimensional, Wolfram
def distPtToLine(h1,p1,p2):
    #Should work if there's no np addition/subtraction errors...
    return np.linalg.norm(np.cross((h1-p1),(h1-p2))) / np.linalg.norm(p1-p2)   #.norm is just the magnitude

#Returns the min distance between two lines, each line defined by two pts
def distTwoLines(h1,h2,p1,p2):
    e1=h1-h2  #Vec along the direction of the line
    e2=p1-p2
    crs=np.cross(e1,e2)
    return np.dot(crs,h1-p1) / np.linalg.norm(crs)

#Fill hitarr (hits used in alg) and regionMap (all hit data, by region) with data from inputArr
def fillDataLists(hitarr,regionMap,inputArr,i_start,i_end,lookin):
    #Sort array of positions by z position to help speed up tracking alg (by DECREASING z)
    if lookin:
	for j in range(i_start,i_end-1):
            #Add hit to corresponding list, based on the index number (j,6) provided
            regionMap[int(inputArr[j,6])].append((j,inputArr[j,1],inputArr[j,2],inputArr[j,3],inputArr[j,4],inputArr[j,7]))
            #Add hits that the tracking alg looks at to a separate array, hitarr
            if inputArr[j,6]==0 or inputArr[j,6]==2 or inputArr[j,6]==3:
                hitarr.append((j,inputArr[j,1],inputArr[j,2],round(inputArr[j,3]),inputArr[j,7]))
    else:
	for j in range(i_start,i_end-1):
            #Add hit to corresponding list, based on the index number (j,6) provided
            regionMap[int(inputArr[j,6])].append((j,inputArr[j,1],inputArr[j,2],inputArr[j,3],inputArr[j,4],inputArr[j,7]))
            #Add hits that the tracking alg looks at to a separate array, hitarr
            if inputArr[j,6]==0 or inputArr[j,6]==2:
                hitarr.append((j,inputArr[j,1],inputArr[j,2],round(inputArr[j,3]),inputArr[j,7]))

    hitarr.sort(key=lambda tup: tup[3], reverse=True)  #Sort by 3rd elem in arr in reverse order

def consider(h, hits):
    box = 30
    cback = h[3] - 30
    cwalls = (h[1]-box/2, h[1]+box/2, h[2]-box/2, h[2]+box/2)
    considerlist=[]
    for hit in hits:
	if cback<=hit[3]<=h[3] and cwalls[0]<=hit[1]<=cwalls[1] and cwalls[2]<=hit[2]<=cwalls[3]:
            #print("Considering "+str(hit))
	    considerlist.append(hit)
    considerlist.remove(h)
    return considerlist

def parse(parent_file):
    parentsList = []
    with open(parent_file, 'rb') as f:
	fList = list(csv.reader(f))
	for row in fList:
	    b = row[0].split(' ')
	    #print(row[0])
	    b.remove('')
	    for i in range(len(b)): b[i] = int(b[i])
	    parentsList.append(b)
	#print(parentsList)
    return parentsList

def findTracks(hitlist, mtl):  #Hitlist=all hits in one event, mtl=min track length
    #TRACKING STARTS HERE

    normtracklist=[]
    hitscopy=hitlist

    for hit in hitlist:  #Go through all hits, starting at the back of the ecal
        if True:  #isolatedEnd(hit, hitlist):
	    track = []
            remember_var = []  #This is used in case neighInNextLayer needs info about past hits (e.g. how many hops used, slope of past jump)
            track.append(hit)
	    lastHit = hit   #Most recent hit added to track
	    nextLayerHits = [] #next being the one after where the last was addedd
	    nnextLayerHits = [] #hits in the layer that would require a "hop"  NOTE might want to merge current and Next hits to give more emphasis to dot than layer dist
	    #lastLayerHits = []	layer where we just added the last hit -- will come back to last
	    #nextLayersHits = []

	    trackComplete = False
	    while not trackComplete:
                nextLayerHits = []
                #nnextLayerHits = []
                #nextLayersHits = []
		for h in hitscopy:
                    if h[3]>380:  printing = True
                    if h==lastHit:  continue
		    #if h[3] == lastHit[3] - layer_intervals[layerZs.index(hit[3]) - 1]:  
		    #    #NOTE s: REMEBER TO NOT LOOK BEYOND THE FIRST LAYER // PICK HIT W/ HIGHEST dot //some redund in checking zs here check that out later
		    neighFound, remember_tmp, dot  = neighInNextLayer(lastHit, h, remember_var) 
		    if neighFound:
                        #    if printing:  print("Appending:  "+str(h))
                        
			nextLayerHits.append((dot, remember_tmp, h))
		#for h in hitscopy:
                #    if h==lastHit:  continue
		#    if h[3] == lastHit[3] - layer_double_intervals[layerZs.index(hit[3]) - 2]:   
		#	#NOTE: definatley room for optimization here apart from the obvious do them at the same time if keeping them apart doesn't turn out to help at all
		#	neighFound, remember_tmp, dot = neighInNextLayer(lastHit, h, remember_var)
		#	if neighFound:
                #            if printing:  print("Appending 2x:  "+str(h))
		#	    nextLayerHits.append((dot, remember_tmp, h))
		#nextLayersHits = nextLayerHits + nnextLayerHits
                #print("  nextlayershits = "+str(nextLayersHits))
		if nextLayerHits == []:
		    trackComplete = True
		    continue
		#print(nextLayersHits)
                #was tup[0]...now sorting by distance
		nextLayerHits.sort(key=lambda tup: np.sqrt((tup[2][1]-lastHit[1])**2 + (tup[2][2]-lastHit[2])**2 +(tup[2][3]-lastHit[3])**2), reverse=False)
		lastHit = nextLayerHits[0][2]
                remember_tmp = nextLayerHits[0][1]
                #print("nlh = "+str(nextLayerHits[0][1]))
                #Update remember_var with exponentially weighted moving average:
                if remember_var == []:
                    remember_var = remember_tmp
                else:
                    remember_var = [.7*remember_var[j] + .3*remember_tmp[j] for j in range(3)]
		track.append(lastHit)
		#if printing:  print("Current track: "+str(track))
		#lastHit = selecthit
            #print("TRACK: "+str(track))
	    if len(track) >=  mtl:
		for h in track:
		    hitlist.remove(h)
	        normtracklist.append(track)
    #print(normtracklist)
    return normtracklist



def findStraightTracks(hitlist, mst):  #Hitlist=all hits in one event, mst = min straight track len
    #In another pass over all events, check for exclusively straight tracks and guarantee that they're accepted.
    #TODO:  Ignore/loosen isolatedEnd restrictions (i.e. only check single preceeding hit)
    #TODO:  Try with and without single hops.

    strtracklist = []
    hitscopy=hitlist
    hopsOn = False

    for hit in hitlist:  #Go through all hits, starting at the back of the ecal
       if True:  #isolatedEnd(hit, hitlist):
            #print("Found isolated end: "+str(hit))
            #Create new track, then sift through all other hits to find others in same track
            track=[]
            track.append(hit)
            currenthit=hit  #Stores "trailing" hit in track being constructed

            possibleNeigh=False
            #jumpCounter = 0  #EXPERIMENTAL
            for h in hitscopy:
                if h[3]==currenthit[3]:
                    possibleNeigh=True  #Optimization
                    continue
                if not possibleNeigh:  continue
                if currenthit[3] - h[3] > 25:  #Optimization
                    possibleNeigh=False
                    continue
                neighFound=False
                otherFound = False
                #EXPERIMENTAL:  Now allowing a NN search every 3-4 hits to allow track merging...
                #rearNeighFound = (currenthit[3]-layer_intervals[layerZs.index(currenthit[3])-1]==h[3]  \
                #        or currenthit[3]-layer_double_intervals[layerZs.index(currenthit[3])-2]==h[3]) \
                #        and h[1]==currenthit[1] and h[2]==currenthit[2]
                #if jumpCounter==0 and not rearNeighFound:
                #    #check for NINL, no hops
                #    otherFound = currenthit[3]-layer_intervals[layerZs.index(currenthit[3])-1]==h[3] and neighbors(h,currenthit,1)
                #    if otherFound:  jumpCounter=4
                #neighFound = rearNeighFound or otherFound
                #OLD:
                neighFound = (currenthit[3]-layer_intervals[layerZs.index(currenthit[3])-1]==h[3]  \
                        or currenthit[3]-layer_double_intervals[layerZs.index(currenthit[3])-2]==h[3]) \
                        and h[1]==currenthit[1] and h[2]==currenthit[2]
                if neighFound:  #neighInNextLayer(currenthit,h,remember_var):
                #    jumpCounter -= 1
                    track.append(h)
                    currenthit=h
                #print("Current = "+str(track))
            if len(track) >= mst:  #This is independently kept at 4, maybe 3
                for ht in track:
                    hitlist.remove(ht)
                strtracklist.append(track)
                #print("Finished creating track with length "+str(len(track)))
                #print(track)
            #print("Tracklist: "+str(strtracklist))

    return strtracklist


def colorfn(i):  #essentially triangular.  0<=i<=1
    if 0<=i<1:  return (1,i,0)
    elif 1<=i<2:  return (2-i,1,0)
    elif 2<=i<3:  return (0,1,i-2)
    elif 3<=i<4:  return (0,4-i,1)
    elif 4<=i<5:  return (i-4,0,1)
    elif 5<=i<6:  return (1,0,6-i)
    else:  return (0,0,0)


def colorList(N):  #Return a list of N triplets spaced out along the RGB spectrum
    #5.6*255 max
    lst = []
    for i in range(N):
        lst.append(colorfn(float(i)/float(N)*6.0))
    return lst


def plotEvent(dataMap,track_list,othertrack_list,otherPartList,originList,p_electron,p_photon):
    #Prepare arrays for plotting
    pltDict = [convertListToArr(dataMap[i]) for i in range(6)]

    tracks=[]
    otracks=[]
    for trk in track_list:
        #print("Converting to list")
        tracks.append(convertListToArr(trk))
    for otrk in othertrack_list:
        otracks.append(convertListToArr(otrk))

    #Plot all hits, color-coded by region
    fig=plt.figure(i)
    ax=Axes3D(fig)
    """
    if pltDict[0]!=[]:
        ax.scatter(pltDict[0][:,3], pltDict[0][:,1], pltDict[0][:,2], c='r')#, label="Outside both radii")  #outside 68 cont
    if pltDict[1]!=[]:
        ax.scatter(pltDict[1][:,3], pltDict[1][:,1], pltDict[1][:,2], c='gray')#, label="Inside e- radius")
    if pltDict[2]!=[]:
        ax.scatter(pltDict[2][:,3], pltDict[2][:,1], pltDict[2][:,2], c='g')#, label="Inside photon radius")
    if pltDict[3]!=[]:
        ax.scatter(pltDict[3][:,3], pltDict[3][:,1], pltDict[3][:,2], c='m')#, label="Inside both radii")
    """
    if pltDict[4]!=[]:
        ax.plot3D(pltDict[4][:,3], pltDict[4][:,1], pltDict[4][:,2], c='y', label="Projected electron, p="+str(round(p_electron))+" MeV/c")
    if pltDict[5]!=[]:
        ax.plot3D(pltDict[5][:,3], pltDict[5][:,1], pltDict[5][:,2], c='c', label="Projected photon, p="+str(round(p_photon))+" MeV/c")
    
    #Plot the tracks
    legendAdded=False
    #markertypes = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'P', '*', '+', 'x', 'X', 'D', 'd']

    for pltarr in tracks:
        if not legendAdded:
            ax.plot(pltarr[:,3],pltarr[:,1],pltarr[:,2], color='k', label="Found tracks")
            legendAdded=True
        else:  ax.plot(pltarr[:,3],pltarr[:,1],pltarr[:,2], color='k')
    legendAdded=False
    trackCount = 0
    colors = colorList(len(otracks))
    
    for opltarr in otracks:
        if not legendAdded:
            #NOTE:  Was ax.plot, ':'!!
            #color_ = np.random.rand(3,)
            ax.scatter(opltarr[:,3],opltarr[:,1],opltarr[:,2], color=colors[trackCount], \
                s = [el/2.0+5 for el in opltarr[:,4]], \
                label = "pdgid="+str(otherPartList[trackCount])+", origin="+str(originList[trackCount]))
            legendAdded=True
        else:
            color_ = np.random.rand(3,)
            ax.scatter(opltarr[:,3],opltarr[:,1],opltarr[:,2], color=colors[trackCount], \
                s = [el/2.0+5 for el in opltarr[:,4]], \
                label="pdgid="+str(otherPartList[trackCount])+", origin="+str(originList[trackCount]))
        trackCount += 1
    
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    plt.legend(loc=4)
    #ax.set_xlim3d(200,350)
    plt.show()


#Main function.  Calls fillDataLists, findTracks, and plotEvent.
def analyze(inArr, min_len, parent_file, momentum_file):  #inArr=array of all hits for all events; min_len=min track length
    i_end = 0
    i_start = 0
    straightTracks = []
    normTracks = []
    regTracks = []
    allTracks = []
    missingEPNum = 0
    missingPNum = 0
    missingEventNum = 0
    p_electron = 0
    p_photon = 0
    parentsList = parse(parent_file)
    momentaList = np.loadtxt(momentum_file)
    i_start_p = 0
    i_end_p = 0
    #print("TESTING parent parsing:")  #This is probably okay
    #print(parentsList[0])
    events =  int(inArr[len(inArr)-1,0]) #- 850
    mom_row = 0  #Store current row in momenta file

    for i in range(1, events+1):  #i=current event being examined
        while not i_end == len(inArr) and i == inArr[i_end,0]:        #print('i_end='+str(i_end))
            i_end += 1
        if i_end == len(inArr):  i_end += 1
        if i%100 == 0:
            print("Processing event "+str(i))
        if i_start == i_end: 
            missingEventNum += 1
	    i_start = i_end  
            continue

        #Get momenta from list
        mom_row = 0
        while momentaList[mom_row,0] != i:
            mom_row += 1
            if mom_row == len(momentaList):
                #print("Warning:  event "+str(i)+" has no corresponding momenta")
                break
        if mom_row == len(momentaList):
            p_electron = 0
            p_photon = 0
        else:
            p_electron = momentaList[mom_row,1]
            p_photon = momentaList[mom_row,2]
        #Check for presence of e- traj.  If none, discard event.
        if inArr[i_start,6] == 4 and inArr[i_start+1,6] == 4 and not inArr[i_start+2,6] == 5 and not inArr[i_start+3,6] == 5:
            missingPNum += 1
            i_start = i_end
            continue
	if not (inArr[i_start,6] == 4 and inArr[i_start+1,6] == 4 and inArr[i_start+2,6]==5 and inArr[i_start+3,6]==5):
	    missingEPNum += 1
	    i_start = i_end
            while not i_end_p == len(parentsList) and i == parentsList[i_end_p][0]:        #print('i_end='+str(i_end))
                i_end_p += 1
            if i_end_p == len(parentsList):  i_end_p += 1
	    continue
        #print(np.array([inArr[i_start+1,1],inArr[i_start+1,2],inArr[i_start+1,3]]))
        #print(np.array([inArr[i_start,1],inArr[i_start,2],inArr[i_start,3]]))
	e_traj = np.array([inArr[i_start+1,1],inArr[i_start+1,2],inArr[i_start+1,3]]) - np.array([inArr[i_start,1],inArr[i_start,2],inArr[i_start,3]])
	p_traj = np.array([inArr[i_start+3,1],inArr[i_start+3,2],inArr[i_start+3,3]]) - np.array([inArr[i_start+2,1],inArr[i_start+2,2],inArr[i_start+2,3]])
	#print(e_traj)
        e_traj_norm = [e_traj[j]/np.sqrt(e_traj[0]**2+e_traj[1]**2+e_traj[2]**2) for j in range(0,3)]
        p_traj_norm = [p_traj[j]/np.sqrt(p_traj[0]**2+p_traj[1]**2+p_traj[2]**2) for j in range(0,3)]
        #print("Dot result is "+str(np.dot(e_traj_norm,p_traj_norm)))
        #ADDED:  Also check whether the traj starting points are nearby
        neartraj = np.sqrt((inArr[i_start,1]-inArr[i_start+2,1])**2 + (inArr[i_start,2]-inArr[i_start+2,2])**2 + (inArr[i_start,3]-inArr[i_start+2,3])**2) < 8.8*2
        colinear = abs(np.dot(e_traj_norm,p_traj_norm)) > .97  #14 deg
	lookin = neartraj and colinear

	#Now, the range i_start to i_end-1 of fileArr[] contains all points for event i.
        #To make things easier, create list of event coordinates:
        #print("Processing event "+str(i))
	hits     = []
        outside  = []  #Arrays containing hits in corresponding regions
        insideE  = []
        insideP  = []
        insideEP = []
        etraj    = []
        ptraj    = []
        categoryMap = {0:outside, 1:insideE, 2:insideP, 3:insideEP, 4:etraj, 5:ptraj}
	
	fillDataLists(hits, categoryMap, inArr, i_start, i_end,lookin)
	#if skip:
	   # print('skipping in analyze')
	    #missingTrackNum += 1
	    #continue

        #print("Finding tracks, first iteration")
        straighttracklist = findStraightTracks(hits, min_straight_len)
        straightTracks.append(straighttracklist)
        normtracklist = findTracks(hits, min_len)  #Perform full tracking algorithm
        normTracks.append(normtracklist)
        #print("Finding tracks, second iteration")
        #print("straight:")
        #print(straighttracklist)
        #print("not:")
        #print(normtracklist)
	#regtracklist = findregTracks(hits, min_len, etraj, ptraj)
        #regtracklist = []
	#regTracks.append(regtracklist)      
        allTracks.append(straighttracklist + normtracklist)   # + regtracklist)


        #Prep for parent plotting info:
        while not i_end_p == len(parentsList) and i == parentsList[i_end_p][0]:        #print('i_end='+str(i_end))
            i_end_p += 1
        if i_end_p == len(parentsList):  i_end_p += 1
        
        acttracklist = []
        actpartlist = []
        actoriglist = []
        hitlist_ = outside+insideP+insideE+insideEP  #include ALL hits for real tracks...
        #print("len hitlist = "+str(len(hitlist_)))
        #print("istart="+str(i_start_p))
        
        while i_start_p != i_end_p:
            #print(i_start_p)
            #Read in real track info, turn into list of points, add to acttracklist
            currentTrack = []
            #Read through each row of parentsList
            originID = parentsList[i_start_p][2]  #pdgID of parent--either photon (22, common) or electron (11, rare)
            row = parentsList[i_start_p][3:]  #Grab list of hit IDs
            #print("row = "+str(row))
            #Grab corresponding hit coordinates
            for h in hitlist_:
                for hnum in row:
                    #print('h5='+str(h[5]))
                    #print('hnum='+str(hnum))
                    if h[5] == hnum:
                        currentTrack.append((h[0],h[1],h[2],h[3],h[4]))
            currentTrack.sort(key=lambda tup: tup[3], reverse=True)
            #print("CurrentTrack = "+str(currentTrack))
            if currentTrack != [] and len(currentTrack)>0:
                acttracklist.append(currentTrack)
                actpartlist.append(parentsList[i_start_p][1])
                actoriglist.append(originID)
            i_start_p += 1
        
        #print("**Actual tracks:**")
        #print(acttracklist)
        #print("Done.  Plotting...")
        if visualize:# and len(straighttracklist+normtracklist)==0:
            print("Plotting event "+str(i)+"...")
            plotEvent(categoryMap, normtracklist+straighttracklist, acttracklist, actpartlist, actoriglist, p_electron, p_photon)
	

        i_start = i_end
    eventsUsed = (events - missingEventNum - missingEPNum - missingPNum)
    print('Number of events missing: ' + str(missingEventNum))
    print('Number of events missing photon trajectory: ' + str(missingPNum))
    print('Number of events missing e- and photon trajectory: ' + str(missingEPNum))
    print('Fraction of  events used: ' + str(eventsUsed)+'/'+str(events))
    return allTracks


#BEGIN ACTUAL PROGRAM

"""min_track_len = 5
min_straight_len = 4
filePath = '/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/LDMX-scripts/mipTracking/tracking_dev/'
bkg_file = 'hits_sim_bkg_1k.txt'  #NOTE:  Used to be 0, 1...now using new files.
sig_file = 'hits_sim_1000MeV_1k.txt' #AND RECON  #'hits_Sig0_1000sig.txt'
bkg_parent_file = 'particleinfo_sim_bkg_1k.txt'  #NOTE:  Not the "parents" file!
sig_parent_file = 'particleinfo_sim_1000MeV_1k.txt'
bkg_momentum_file = 'momenta_sim_bkg_1k.txt'
sig_momentum_file = 'momenta_sim_1000MeV_1k.txt'
#unv_momentum_file = 'momenta_sim_unvet_1k.txt'
print("Loading files")
unv_parent_file = 'particle_tmp.txt'  #'particleinfo_sim_unvet_2.txt'
unv_file = 'unvet_temp.txt'  #'hits_sim_unvet_2.txt'
"""

#bkgFiles, sigFiles, unvFiles=[hits,parent,mom]

bkgATracks = []
sigATracks = []
unvATracks = []
foundTracks = [bkgATracks, sigATracks, unvATracks]
for q in range(len(runList)):
    if runList[q]:
        print("Loading "+str(filesArr[q][0]))
        fileArr = np.loadtxt(filePath+filesArr[q][0])
        print("Analyzing...")
        foundTracks[q].extend(analyze(fileArr, min_track_len, filePath+filesArr[q][1], filePath+filesArr[q][2]))
print('Finished counting tracks.')

#Histogram Plotting
x_max=12  #Please don't make this an non-integer and ruin everything

if bkgOn and showHists:
    bkgNTracks = []
    for trks in bkgATracks:
        bkgNTracks.append(len(trks))
    plt.figure(1)
    plt.hist(bkgNTracks, bins=x_max, range=(0,x_max))
    plt.title("bkg, "+qualifier+", min len="+str(min_track_len))
    plt.xlabel("Tracks per event")
    plt.ylabel("Number of events")
    if saveHists: plt.savefig("tracks_figs/bkg_"+adds+"_"+str(min_track_len)+".png")
    else:  plt.show()

if sigOn and showHists:
    sigNTracks = []
    for trks in sigATracks:
        sigNTracks.append(len(trks))
    plt.figure(2)
    plt.hist(sigNTracks, bins=x_max, range=(0,x_max))
    plt.title("sig, "+qualifier+", min len="+str(min_track_len))
    plt.xlabel("Tracks per event")
    plt.ylabel("Number of events")
    if saveHists: plt.savefig("tracks_figs/sig_"+adds+"_"+str(min_track_len)+".png")
    else:  plt.show()

if unvOn and showHists:
    unvNTracks = []
    for trks in unvATracks:
        unvNTracks.append(len(trks))
    print('Number of tracks by event: '+str(unvNTracks))
    plt.figure(3)
    plt.hist(unvNTracks, bins=x_max, range=(0,x_max))
    plt.title("un-vetoed, "+qualifier+", min len="+str(min_track_len))
    plt.xlabel("Tracks per event")
    plt.ylabel("Number of events")
    if saveHists: plt.savefig("tracks_figs/unvet_"+adds+"_"+str(min_track_len)+".png")
    else:  plt.show()


"""
bkgArr = np.loadtxt(filePath+bkg_file)
sigArr = np.loadtxt(filePath+sig_file)
print("Files loaded.  Analyzing background...")
bkgATracks = analyze(bkgArr, min_track_len, bkg_parent_file, bkg_momentum_file)
print("Analyzing signal...")
sigATracks = analyze(sigArr, min_track_len, sig_parent_file, sig_momentum_file)

#inpArr = np.loadtxt(filePath+unv_file)
#tmpTracks = analyze(inpArr, min_track_len, unv_parent_file, unv_momentum_file)

print("Finished counting tracks.")

#Plotting


sigNTracks = []
bkgNTracks = []
for trks in sigATracks:
    sigNTracks.append(len(trks))
for trks in bkgATracks:
    bkgNTracks.append(len(trks))

x_max=10  #Please don't make this an non-integer and ruin everything
adds = "str_plus_plotting"
qualifier = "with straight tracks"

plt.figure(3)
plt.hist(bkgNTracks, bins=x_max, range=(0,x_max))
plt.title("bkg, "+qualifier+", min len="+str(min_track_len))
plt.xlabel("Tracks per event")
plt.ylabel("Number of events")
plt.show()
#plt.savefig("linreg_figs/bkg_"+adds+"_"+str(min_track_len)+".png")

plt.figure(4)
plt.hist(sigNTracks, bins=x_max, range=(0,x_max))
plt.title("sig, "+qualifier+", min len="+str(min_track_len))
plt.xlabel("Tracks per event")
plt.ylabel("Number of events")
plt.show()
#plt.savefig("linreg_figs/sig_"+adds+"_"+str(min_track_len)+".png")
"""




