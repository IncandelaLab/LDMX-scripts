import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import colorsys

print('################################################################ New Run ########################################################################')

#Files to run on:
min_track_len = 4
min_straight_len = 4
filePath = '/nfs/slac/g/ldmx/users/jmlazaro/ldmx-sw/scripts/mipTracking/outHitTxts/'
showHists = False 
saveHists = False

bkg_file = 'hits_sim_bkg_2.txt'  
bkg_parent_file = 'particleinfo_sim_bgk_2.txt'  
bkg_momentum_file = 'momenta_sim_bkg_2.txt'
bkgFiles = [bkg_file, bkg_parent_file, bkg_momentum_file]
bkgOn = False

sig_file = 'hits_sim_1000MeV_2.txt' 
sig_parent_file = 'particleinfo_sim_1000MeV_2.txt'
sig_momentum_file = 'momenta_sim_1000MeV_2.txt'
sigFiles = [filePath , sig_parent_file, sig_momentum_file]
sigOn = False

unv_file = 'hits_sim_unvet_2.txt'
unv_parent_file = 'particleinfo_sim_unvet_2.txt'
unv_momentum_file = 'momenta_sim_unvet_2.txt'
unvFiles = [unv_file, unv_parent_file, unv_momentum_file]
unvOn = True

filesArr = [bkgFiles, sigFiles, unvFiles]
runList = [bkgOn,sigOn,unvOn]

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

#FUNCTIONS USED:

def neighbors(h1,h2,mult):
    #sqrt(76)mm ~ 8mm = neighbors are within one cell.  mult=2 -> within 2 cells, etc.
    #Note:  This only compares x and y coordinates.  Two hits in different layers w/ identical xy coords will always be neighbors.
    return np.sqrt((h1[1]-h2[1])**2 + (h1[2]-h2[2])**2) < 8.7*mult

#Check whether h2 is a layer in front of h1 and a neighbor of h1; if so, return true
#remembervar is in case nINL needs to "remember" previous results, e.g. whether a hop has been used in the track
def neighInNextLayer(h1,h2,remembervar):
    #checks if h2 is in the layer(s) in front of h1 and a neighbor of h1
    rememberOut = []
    hopUsed=False
    inNearestLayer=False
    inSameLayer=False
    reject = False
    if h1[3]>228:  #If not in the front two layers...
        hopUsed = h1[3]-layer_double_intervals[layerZs.index(h1[3])-2]==h2[3] and neighbors(h1,h2,2)
    if h1[3]>225:  #If not in the front layer...
        inNearestLayer = h1[3]-layer_intervals[layerZs.index(h1[3])-1]==h2[3] and neighbors(h1,h2,2)

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
	if hit[3]<502:
            if ((h[3]==hit[3] or hit[3]-layer_intervals[layerZs.index(hit[3])]==h[3]) and neighbors(hit,h,1)): #or far_neighbor:
                return False
        #else:  return True, since it's at the back of the ecal
    return True

#Convert a list of 5-tuples (i,x,y,z,E) to a np array
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

#Min distance from pt h1 to line specified by points p1 and p2--see point-line distance 3-dimensional, Wolfram
def distPtToLine(h1,p1,p2):
    return np.linalg.norm(np.cross((h1-p1),(h1-p2))) / np.linalg.norm(p1-p2)

#Returns the min distance between two lines, each line defined by two pts
def distTwoLines(h1,h2,p1,p2):
    e1=h1-h2  #Vec along the direction of the line
    e2=p1-p2
    crs=np.cross(e1,e2)
    return np.dot(crs,h1-p1) / np.linalg.norm(crs)

#Fill hitarr (hits used in alg) and regionMap (all hit data, by region) with data from inputArr
def fillDataLists(hitarr,regionMap,inputArr,i_start,i_end,lookin):
    if lookin:
        for j in range(i_start,i_end-1):
            #Add hit to corresponding list, based on the index number (j,6) provided
	    #format of added hits: (0 = eventnum, x, y, z, 4 = E, 5 = hitID)
            regionMap[int(inputArr[j,6])].append((inputArr[j,0],inputArr[j,1],inputArr[j,2],inputArr[j,3],inputArr[j,4],inputArr[j,7]))
            #Add hits that the tracking alg looks at to a separate array, hitarr
            if inputArr[j,6]==0 or inputArr[j,6]==2 or inputArr[j,6]==3:
                hitarr.append((inputArr[j,0],inputArr[j,1],inputArr[j,2],round(inputArr[j,3]),inputArr[j,4],inputArr[j,7]))
    else:
        for j in range(i_start,i_end-1):
            #Add hit to corresponding list, based on the index number (j,6) provided
            regionMap[int(inputArr[j,6])].append((inputArr[j,0],inputArr[j,1],inputArr[j,2],inputArr[j,3],inputArr[j,4],inputArr[j,7]))
            #Add hits that the tracking alg looks at to a separate array, hitarr
            if inputArr[j,6]==0 or inputArr[j,6]==2:
                hitarr.append((inputArr[j,0],inputArr[j,1],inputArr[j,2],round(inputArr[j,3]),inputArr[j,4],inputArr[j,7]))

    hitarr.sort(key=lambda tup: tup[3], reverse=True)  #Sort by 3rd elem in arr in reverse order

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

def findStraightTracks(hitlist, mst):  #Hitlist=all hits in one event, mst = min straight track len
    strtracklist = []
    hitscopy=hitlist
    hopsOn = False

    for hit in hitlist: 
       if isolatedEnd(hit, hitlist):
            track=[]
            track.append(hit)
            currenthit=hit  #Stores "trailing" hit in track being constructed

            possibleNeigh=False
            jumpCounter = 0  #EXPERIMENTAL
            for h in hitscopy:
                if h[3]==currenthit[3]:
                    possibleNeigh=True 
                    continue
                if not possibleNeigh:  continue
                if currenthit[3] - h[3] > 25:
                    possibleNeigh=False
                    continue
                neighFound=False
                otherFound = False
                #EXPERIMENTAL:  Now allowing a NN search every 3-4 hits to allow track merging...
                rearNeighFound = (currenthit[3]-layer_intervals[layerZs.index(currenthit[3])-1]==h[3]  \
                        or currenthit[3]-layer_double_intervals[layerZs.index(currenthit[3])-2]==h[3]) \
                        and h[1]==currenthit[1] and h[2]==currenthit[2]
                if jumpCounter==0 and not rearNeighFound:
                    #check for NINL, no hops
                    otherFound = currenthit[3]-layer_intervals[layerZs.index(currenthit[3])-1]==h[3] and neighbors(h,currenthit,1)
                    if otherFound:  jumpCounter=4
                neighFound = rearNeighFound or otherFound
                if neighFound:  #neighInNextLayer(currenthit,h,remember_var):
                    jumpCounter -= 1
                    track.append(h)
                    currenthit=h
            if len(track) >= mst:  #This is independently kept at 4, maybe 3
                for ht in track:
                    hitlist.remove(ht)
                strtracklist.append(track)
    return strtracklist

def findNormTracks(hitlist, mtl):
    normtracklist=[]
    hitscopy=hitlist

    for hit in hitlist:  
        if isolatedEnd(hit, hitlist):
            track = []
            remember_var = []   
            track.append(hit)
            lastHit = hit   #Most recent hit added to track

            trackComplete = False
            while not trackComplete:
                nextLayerHits = []
                for h in hitscopy:
                    if h==lastHit:  continue
                    neighFound, remember_tmp, dot  = neighInNextLayer(lastHit, h, remember_var)
                    if neighFound:
                        nextLayerHits.append((dot, remember_tmp, h))
                if nextLayerHits == []:
                    trackComplete = True
                    continue
                #nextLayerHits.sort(key=lambda tup: np.sqrt((tup[2][1]-lastHit[1])**2 + (tup[2][2]-lastHit[2])**2 +(tup[2][3]-lastHit[3])**2), reverse=False)
                lastHit = nextLayerHits[0][2]
                remember_tmp = nextLayerHits[0][1]
                #Update remember_var with exponentially weighted moving average:
                if remember_var == []:
                    remember_var = remember_tmp
                else:
                    remember_var = [.6*remember_var[j] + .4*remember_tmp[j] for j in range(3)]
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

"""
def regCheck(trackList)
    for track in trackList:
	if r_squared(track) < min_r_squared: trackList.remove(track)
"""

def measurePurity(funcTracks, realTracks):
    print('in meaurePurity')
#    for row in funcTracks:
#	print(row)
    found_s = 0
    found_e = 0
    real_s = 0
    real_e = 0
    #for ev in range(1, len()funcTracks[]):
    purities = []
    #print(realTracks[14])
    #print(realTracks[11])
    while found_e < len(funcTracks):
	#print('found_e: ' + str(found_e))
	#print(funcTracks[found_e])
	while funcTracks[found_e][0][0] == funcTracks[found_s][0][0]:
	    found_e += 1
	    if found_e == len(funcTracks): continue
	    #print('found_s: ' + str(found_s) + ', found_e: ' + str(found_e))
	    #print('realTracks[found_s]: ' + str(realTracks[found_s]))
	    #print('realTracks: ' + str(realTracks))
	print('found_s: ' + str(found_s) + ', found_e: ' + str(found_e))
	while realTracks[real_e][0][0] == funcTracks[found_s][0][0]:
	    real_e += 1
	#real_e += 1
	    #print('real_s: ' + str(real_s) + ', real_e: ' + str(real_e))
	while found_s < found_e:
	    print('found_s: ' + str(found_s))
	    foundIDs = []
	    posibpurities = []
	    #print('hitsInFoundTrack: ' + str(funcTracks[found_s]))
	    for hit in funcTracks[found_s]:
		foundIDs.append(hit[5])
	    #print('foundIDs: ' + str(foundIDs))
	    print('real_s: ' + str(real_s) + ', real_e: ' + str(real_e))
	    # #################################################################################################i
	    real_s_copy = real_s
	    while real_s < real_e:
		matches = 0
		#print('matches: ' + str(matches))
		#print('realTrack: ' + str(realTracks[real_s]))
		for h in realTracks[real_s]:
		    #print('realhit: ' + str(h))
		    if h[5] in foundIDs:
			matches += 1
		    #print('matches: ' + str(matches))
		posibpurities.append(matches / len(funcTracks[found_s]))
	        real_s += 1
	    if found_s < found_e: real_s = real_s_copy
	    print('posibpurities: ' + str(posibpurities))
	    purities.append(max(posibpurities))
	    print('max(posibpurities): ' + str(max(posibpurities)))
	    found_s += 1
	real_s = real_e
	print('found_s: ' + str(found_s) + ', found_e: ' + str(found_e))
    print('finished measurePurity')
    return [np.mean(purities), np.std(purities), len(purities)]

def totalPurity(funcTracksList,realTracks):
    print('in totalPurity')
    #print(funcTracksList)
    functionPurities = []
    for tracks in funcTracksList:
	if tracks == funcTracksList[0]: #NOTE remove this line later 
	    if tracks != []: functionPurities.append(measurePurity(tracks, realTracks))
    totalPurity = sum([functionPurities[w][2]*functionPurities[w][0] for w in range(len(functionPurities))])/sum(functionPurities[:][2])
    totalPuritySD = np.sqrt(sum([functionPurities[A][2]*(np.array(functionPurities[A][1]))**2 for A in range(len(functionPurities))])/sum(functionPurities[:][2]))
    return [functionPutrities[:2], totalPurity, totalPuritySD]

def plotEvent(dataMap,track_list,othertrack_list,otherPartList,p_electron,p_photon,eventNumb):
    #Prepare arrays for plotting
    pltDict = [convertListToArr(dataMap[k]) for k in range(6)]

    tracks=[]
    otracks=[]
    for trk in track_list:
        #print("Converting to list")
        tracks.append(convertListToArr(trk))
    for otrk in othertrack_list:
        otracks.append(convertListToArr(otrk))

    #Plot all hits, color-coded by region
    fig=plt.figure(eventNumb)
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
    if pltDict[4]!=[]: #NOTE: try removing, shouldnt be dealing with events w/o both trajs anyway
        ax.plot3D(pltDict[4][:,3], pltDict[4][:,1], pltDict[4][:,2], c='y', label="Projected electron, p="+str(p_electron)+" MeV")
    if pltDict[5]!=[]:
        ax.plot3D(pltDict[5][:,3], pltDict[5][:,1], pltDict[5][:,2], c='c', label="Projected photon trajectory, p="+str(p_photon)+" MeV")

    #Plot the tracks
    legendAdded=False
    #markertypes = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'P', '*', '+', 'x', 'X', 'D', 'd']

    N = len(otracks)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    for pltarr in tracks:
        if not legendAdded:
            ax.plot(pltarr[:,3],pltarr[:,1],pltarr[:,2], color='k', label="Found tracks")
            legendAdded=True
        else:  ax.plot(pltarr[:,3],pltarr[:,1],pltarr[:,2], color='k')
    legendAdded=False
    trackCount = 0
    for opltarr in otracks:
        if not legendAdded:
            #NOTE:  Was ax.plot, ':'!!
            color_ = np.random.rand(3,)
            ax.scatter(opltarr[:,3],opltarr[:,1],opltarr[:,2], color=color_, label="pdgid="+str(otherPartList[trackCount]))
            legendAdded=True
        else:
            color_ = np.random.rand(3,)
            ax.scatter(opltarr[:,3],opltarr[:,1],opltarr[:,2], color=color_, label="pdgid="+str(otherPartList[trackCount]))
        trackCount += 1
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("X (mm)")
    ax.set_zlabel("Y(mm)")
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
    funcTracksList = [[],[]]
    realTracks = []
    missingEPNum = 0
    missingPNum = 0
    missingEventNum = 0
    p_electron = 0
    p_photon = 0
    parentsList = parse(parent_file)
    momentaList = np.loadtxt(momentum_file)
    i_start_p = 0
    i_end_p = 0
    events =  int(inArr[len(inArr)-1,0]) #- 850
    mom_row = 0  #Store current row in momenta file

    for i in range(1, events+1):  #i=current event being examined
        #print('#################################################### New Event: ' + str(i) + ' ####################################################')
	while not i_end == len(inArr) and i == inArr[i_end,0]:        #print('i_end='+str(i_end))
            i_end += 1
        if i_end == len(inArr):  i_end += 1
        if i%100 == 0:
            print("Processing event "+str(i))
        if i_start == i_end:
            print(i_start)
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
       
        #print("Finding tracks, first iteration")
        straighttracklist = findStraightTracks(hits, min_straight_len)  #Perform full tracking algorithm
        straightTracks.append(straighttracklist)
	normtracklist = findNormTracks(hits,min_len)
	normTracks.append(normtracklist)
        #regtracklist = findregTracks(hits, min_len, etraj, ptraj)
        #regtracklist = []
        #regTracks.append(regtracklist)
        allTracks.append(straighttracklist + normtracklist)   # + regtracklist)
	funcTracksList[0].extend(straighttracklist)
	funcTracksList[1].extend(normtracklist)
	#if i == 1 or i == 2: print(funcTracksList)

        #Prep for parent plotting info:
        while not i_end_p == len(parentsList) and i == parentsList[i_end_p][0]:        #print('i_end='+str(i_end))
            i_end_p += 1

        actpartlist = []
        acttracklist = []
        hitlistFull = outside+insideP+insideE+insideEP  #include ALL hits for real tracks...
        #print("len hitlist = "+str(len(hitlistFull)))
        #print("istart="+str(i_start_p))
        while i_start_p != i_end_p:
            #print(i_start_p)
            #Read in real track info, turn into list of points, add to acttracklist
            currentTrack = []
            #Read through each row of parentsList
	    #print(i_start_p, i_end_p)
            hitIDs = parentsList[i_start_p][2:]  #Grab list of hit IDs
            #print("row = "+str(row))
            #Grab corresponding hit coordinates
            for h in hitlistFull:
                if h[5] in hitIDs:
                    currentTrack.append(h)
            currentTrack.sort(key=lambda tup: tup[3], reverse=True)
            #print("CurrentTrack = "+str(currentTrack))
            if currentTrack != [] and len(currentTrack)>0: #not like but i guess if a track has 2/3  or even 1/3 it should be recorded
                acttracklist.append(currentTrack)
                actpartlist.append(parentsList[i_start_p][1])
            i_start_p += 1
	#print('len(acttracklist): ' + str(len(acttracklist)))
	realTracks.extend(acttracklist)
	    #print('len(acttracklist): ' + str(len(acttracklist)))
	#print('True track list: ' + str(acttracklist))
	#print('Particle list: ' + str(actpartlist))

        #print("Done.  Plotting...")
        if False:  #len(straighttracklist+normtracklist)>0:
            print("Plotting event "+str(i)+"...")
            plotEvent(categoryMap, normtracklist+straighttracklist, acttracklist, actpartlist, p_electron, p_photon,i)

        i_start = i_end

   
    eventsUsed = (events - missingEventNum - missingEPNum - missingPNum)
    print('Number of events missing: ' + str(missingEventNum))
    print('Number of events missing photon trajectory: ' + str(missingPNum))
    print('Number of events missing e- and photon trajectory: ' + str(missingEPNum))
    print('Fraction of  events used: ' + str(eventsUsed)+'/'+str(events))
    print(totalPurity(funcTracksList,realTracks))
    return allTracks


#BEGIN ACTUAL PROGRAM
bkgATracks = []
sigATracks = []
unvATracks = []
foundTracks = [bkgATracks, sigATracks, unvATracks]
for q in range(len(runList)):
    if runList[q]:
	fileArr = np.loadtxt(filePath+filesArr[q][0])
	foundTracks[q].extend(analyze(fileArr, min_track_len, filePath+filesArr[q][1], filePath+filesArr[q][2]))
print('Finished counting tracks.')

#Histogram Plotting
x_max=5  #Please don't make this an non-integer and ruin everything
adds = "straight_tracks"
qualifier = "with straight tracks"

if bkgOn and showHists:
    bkgNTracks = []
    for trks in bkgATracks:
	bkgNTracks.append(len(trks))
    plt.figure(3)
    plt.hist(bkgNTracks, bins=x_max, range=(0,x_max))
    plt.title("bkg, "+qualifier+", min len="+str(min_track_len))
    plt.xlabel("Tracks per event")
    plt.ylabel("Number of events")
    plt.show()
    if saveHists: plt.savefig("linreg_figs/bkg_"+adds+"_"+str(min_track_len)+".png")

if sigOn and showHists:
    sigNTracks = []
    for trks in sigATracks:
	sigNTracks.append(len(trks))
    plt.figure(4)
    plt.hist(sigNTracks, bins=x_max, range=(0,x_max))
    plt.title("sig, "+qualifier+", min len="+str(min_track_len))
    plt.xlabel("Tracks per event")
    plt.ylabel("Number of events")
    plt.show()
    if saveHists: plt.savefig("linreg_figs/sig_"+adds+"_"+str(min_track_len)+".png")

if unvOn and showHists:
    unvNTracks = []
    for trks in unvATracks:
	unvNTracks.append(len(trks))
    print('Number of tracks by event: '+str(unvNTracks))
    plt.figure(5)
    plt.hist(unvNTracks, bins=x_max, range=(0,x_max))
    plt.title("un-vetoed, "+qualifier+", min len="+str(min_track_len))
    plt.xlabel("Tracks per event")
    plt.ylabel("Number of events")
    plt.show()
    if saveHists: plt.savefig("linreg_figs/unvet_"+adds+"_"+str(min_track_len)+".png")
  

