import numpy as np
#import statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Implementation of MIP tracking code
#No-ambiguity version:  meant to resolve previous ambig in which hit to choose next...
#...and maybe ambiguities where multiple equidistant hits are candidates.
#This requires restructuring of the tracking algorithm.
#(Linreg unmodified.)

#INITIALIZING LAYER GEOMETRY INFO:

layerZs_full = [223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,
        266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,
        322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,
        375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,
        448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125]

layerZs = [round(i) for i in layerZs_full]  #Round to nearest int

layer_intervals=[]
for l in range(len(layerZs_full)-1):
    layer_intervals.append(round(layerZs[l+1]-layerZs[l]))


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
    #doublehop = False
    #triplehop = False
    reject = False
    #if h1[3]>236:
    #    triplehop = h1[3]-layer_intervals[layerZs.index(h1[3])-4]==h2[3] and neighbors(h1,h2,4)
    #if h1[3]>232:
    #    doublehop = h1[3]-layer_intervals[layerZs.index(h1[3])-3]==h2[3] and neighbors(h1,h2,3)
    #print("        hopUsed: "+str(h1[3]-layer_intervals[layerZs.index(h1[3])-2])+" and "+str(h2[3])+", "+str(neighbors(h1,h2,2)))
    #print("        hopUsed: "+str(h1[3]-layer_intervals[layerZs.index(h1[3])-1])+" and "+str(h2[3])+", "+str(neighbors(h1,h2,2)))
    if h1[3]>228:  #If not in the front two layers...
        hopUsed = h1[3]-layer_intervals[layerZs.index(h1[3])-2]==h2[3] and neighbors(h1,h2,2)
    if h1[3]>225:  #If not in the front layer...
        inNearestLayer = h1[3]-layer_intervals[layerZs.index(h1[3])-1]==h2[3] and neighbors(h1,h2,2)
    #print("Truth vals are "+str(inNearestLayer)+", "+str(hopUsed))
    inSameLayer = h1[3]==h2[3] and neighbors(h1,h2,1)

    #Remember slope of previous jump
    posdist = np.sqrt((h1[1]-h2[1])**2 + (h1[2]-h2[2])**2 + (h1[3]-h2[3])**2)
    current_slope = [(h1[i] - h2[i])/posdist for i in range(1,4)]
    #If the current slope differs too much from the old slope, reject it.  Currently considering a 45 degree cone.
    if remembervar != [] and (inNearestLayer or hopUsed or inSameLayer):  # or doublehop or triplehop):
        if np.dot(current_slope, remembervar) < .5:
            reject = True
    if (inNearestLayer or hopUsed or inSameLayer) and not reject:
        rememberOut = current_slope
        return True, rememberOut
    else:  return False, rememberOut

#Returns true iff isolatedEnd has been found
def isolatedEnd(hit,hitlist):
    for h in hitlist:
        if h==hit:  continue
        far_neighbor = False
        #NOTE:  Experimentally commenting these lines out.  Sorted consideration/removal should stop 2x counting.
        #if hit[3]<492:  #If not at the back of the ecal:
        #    if hit[3]-layer_intervals[layerZs.index(hit[3])+1]==h[3] and neighbors(hit,h,1):  far_neighbor=True
        if hit[3]<502:
            if ((h[3]==hit[3] or hit[3]-layer_intervals[layerZs.index(hit[3])]==h[3]) and neighbors(hit,h,1)) or far_neighbor:
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
def fillDataLists(hitarr,regionMap,inputArr,i_start,i_end):
    #if i_start<100:
        #print('\n'+'new event:'+' \n'+ str(inputArr[i_start,(0,6)]) +'\n'+ str(inputArr[i_start+1,(0,6)]) +'\n'+ str(inputArr[i_start+2,(0,6)]) +'\n'+ str(inputArr[i_start+3,(0,6)]))
    #if not (inputArr[i_start,6]==4 and inputArr[i_start+1,6]==4 and inputArr[i_start+2,6]==5 and inputArr[i_start+3,6]==5):
	#print('\n'+'new event:'+' \n'+ str(inputArr[i_start,(0,6)]) +'\n'+ str(inputArr[i_start+1,(0,6)]) +'\n'+ str(inputArr[i_start+2,(0,6)]) +'\n'+ str(inputArr[i_start+3,(0,6)]))
	#print('skipping in fillData')
	#skip = True	
	#print(skip)	
    for j in range(i_start,i_end-1):
        #Add hit to corresponding list, based on the index number (j,6) provided
        regionMap[int(inputArr[j,6])].append((j,inputArr[j,1],inputArr[j,2],inputArr[j,3],inputArr[j,4]))
        #Add hits that the tracking alg looks at to a separate array, hitarr
        if inputArr[j,6]==0 or inputArr[j,6]==2:
            hitarr.append((j,inputArr[j,1],inputArr[j,2],round(inputArr[j,3])))
    #Sort array of positions by z position to help speed up tracking alg (by DECREASING z)
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

def findTracks(hitlist, mtl):  #Hitlist=all hits in one event, mtl=min track length
    #TRACKING STARTS HERE

    normtracklist=[]
    hitscopy=hitlist

    """currentHit = []  #Initializing for (maybe) performance--can be deleted without issue
    track = []
    remember_var = []
    currentHit = []
    cuerrentIndex = 0
    lastHit = []
    lastIndex = 0
    afterList = []
    sameList = []
    foundHit = []
    foundIndex = 0
    foundVar = []
    minDistance = 100
    minFound = False"""

    hitIndex = 0  #Stores location of hit in hitlist.  Useful b/c want to find hits near it in hitlist...
    for hit in hitlist:  #Go through all hits, starting at the back of the ecal
        print("Considering hit "+str(hit))
        if isolatedEnd(hit, hitlist):
            print("Found isolated end: "+str(hit))
            track = []
            remember_var = []  #This is used in case neighInNextLayer needs info about past hits (e.g. how many hops used, slope of past jump)
            track.append(hit)

            currentHit = hit  #Stores "trailing" hit in track being constructed
            currentIndex = hitIndex  #Stores index of current hit being considered
            lastHit = hit  #Last hit added to track
            lastIndex = hitIndex  #Stores index of last hit added to track
            afterList = []  #List of cand hits beyond current layer
            sameList = []  #List of cand hits in current layer

            trackIncomplete = True
            while trackIncomplete:
                #print("*Searching...")
                #Given lastHit:  iterate back through list, changing currentHit in the process
                while currentIndex != 0 and hitlist[currentIndex-1][3] <= lastHit[3]:  #While not a layer behind:
                    currentIndex -= 1
                    currentHit = hitlist[currentIndex]
                    #print("    lasthit[] = "+str(lastHit[3]))
                    #print("    hitlist... = "+str(hitlist[currentIndex-1][3]))
                  #currentIndex += 1
                #print("Part 2")
                #print("   currInd = " + str(currentIndex))
                #print("   currHit = " + str(currentHit))
                
                #Until considered hit is 2 layers after the candidate (i.e. not part of track):
                while layerZs[layerZs.index(lastHit[3])-2] <= currentHit[3]:   #NOTE:  Was <=, -2
                    #print("   layers = "+str(layerZs[layerZs.index(lastHit[3])-1]))
                    #print("   curr hit layer = "+str(currentHit[3]))
                    if currentHit == lastHit:
                        currentIndex += 1
                        if currentIndex == len(hitlist):
                            break
                        currentHit = hitlist[currentIndex]
                        continue
                    #print("Checking neighbor w/ "+str(currentHit))
                    neighFound, remember_current = neighInNextLayer(lastHit, currentHit, remember_var)
                    if neighFound:
                        #print("   Neighbor found!")
                        if currentHit[3] == lastHit[3]:  #In same layer
                            sameList.append((currentHit,currentIndex,remember_current))
                        else:  #In two following layers
                            afterList.append((currentHit,currentIndex,remember_current))
                    currentIndex += 1
                    if currentIndex == len(hitlist):
                        break
                    currentHit = hitlist[currentIndex]
                    #print("     Next loop:  currentHit = "+str(currentHit))
                #print("Part 3")
                foundHit = []
                foundIndex = 0
                foundVar = []
                minDist = 100
                minFound = False
                #print("   afterList: "+str(afterList))
                #Add the candidate that's physically closest to lastHit, prioritizing hits in other layers
                for candidate in afterList:  #Check hits in other layers first
                    if hitDistance(candidate[0], lastHit) < minDist:
                        #print("Candidate found: "+str(candidate[0]))
                        minFound = True
                        foundHit = candidate[0]
                        foundIndex = candidate[1]
                        foundVar = candidate[2]
                if not minFound:  #Then search for same-layer neighbors if necessary
                    for candidate in sameList:
                        if hitDistance(candidate[0], lastHit) < minDist:
                            #print("Same-layer candidate found: "+str(candidate[0]))
                            minFound = True
                            foundHit = candidate[0]
                            foundIndex = candidate[1]
                            foundVar = candidate[2]
                if not minFound:
                    trackIncomplete = False
                    continue
                track.append(foundHit)
                lastHit = foundHit
                lastIndex = foundIndex
                remember_var = foundVar
            #Track is complete
            if len(track) >= mtl:
                normtracklist.append(track)
                for ht in track:
                    hitlist.remove(ht)

        hitIndex += 1
    #print("                               ***full tracks:***  "+str(normtracklist))
    return normtracklist



    """        possibleNeigh=False
            for h in hitscopy:  #NOTE:  Because hitscopy is sorted, and because hit h' can't be a parent of hit h if h'[z]>h[z],
                #...searching over all hits in hitscopy is unnecessary.
	        if h[3]==currenthit[3]:
                    possibleNeigh=True  #Optimization:  None of the points examined before currenthit in hitlist can be neighbors; they're all behind it.
                    continue
                if not possibleNeigh:  continue
                if currenthit[3] - h[3] > 25:  #Optimization:  Don't check hits more than 2 layers away
                    possibleNeigh=False
                    continue

                #if 0<h[1]<80 and 125<h[2]<250 and 375<h[3]:
                #    print("Comparing hit "+str(h)+" with "+str(currenthit))
                neighFound, remember_var = neighInNextLayer(currenthit,h,remember_var)
                if neighFound:  #neighInNextLayer(currenthit,h,remember_var):
		#    if 0<h[1]<80 and 125<h[2]<250 and 375<h[3]:  print("Neighbor in next layer found, adding "+str(h)+" to track.")
		    track.append(h)
		    currenthit=h
            if len(track) >= mtl:  #Long enough to count, so add it:
                for ht in track:
                    hitlist.remove(ht)
                normtracklist.append(track)
                #print("Finished creating track with length "+str(len(track)))
                #print(track)
    return normtracklist"""


def findregTracks(hitlist, mtl, etraj, ptraj):
    regtracklist=[]
    trackdiam = 5.0
    min_r = .95

    if etraj==[]:  #This happens rarely--ASK!
        #print("0 etraj points found")
        return []
    #print etraj
    e_traj = convertListToArr(etraj)[:,1:4]
    p_traj = convertListToArr(ptraj)[:,1:4]
    for hit in hitlist:  #>O(n^3) at the moment, so not quite ideal...we'll see whether this can be optimized.
	if isolatedEnd(hit, hitlist):
	    track=[]
	    #track.append(hit)
            hitlist2=hitlist
	    considerlist = consider(hit, hitlist)
	    considerlist2 = considerlist
            maxdata = {'r': 0, "mean": None, "point": None, "h1": None, "h2": None, "slope": None}
	    for h1 in considerlist:  #Look at all possible combos of h1, h2, and hit
                #print("h1: "+str(h1))
		for h2 in considerlist2:
		    if h2 == h1: continue
                    #print("h2: "+str(h2))
		    regdata = [hit, h1, h2]
		    regArr = convertListToArr(regdata)[:,1:4]  #NOTE:  Want only the coords here, not the hit IDs and energies!
                    #print("Considering trio "+str(regArr))
		    regmean = regArr.mean(axis=0)
		    u, d, v = np.linalg.svd(regArr - regmean)
		    slopevec = v[0]  #This contains the slope of the best-fit line; regmean is the point that specifies it
                    pt = slopevec + regmean  #Another point on the best-fit line, arbitrary
                    #Check whether line intersects w/ electron or photon trajectory
		    closest = min(distTwoLines(e_traj[0,:], e_traj[1,:], regmean, pt), distTwoLines(p_traj[0,:], p_traj[1,:], regmean, pt))
                    if closest > 15:  continue  #If closest dst of approach is >15 mm, ignore the track
                    vrnc = distance(regmean,regArr[0])**2+distance(regmean,regArr[1])**2+distance(regmean,regArr[2])**2  #Rough N*variance equivalent
                    sumerr = distPtToLine(regArr[0],regmean,pt)**2 + distPtToLine(regArr[1],regmean,pt)**2 + distPtToLine(regArr[2],regmean,pt)**2
                    r_corr = 1 - sumerr/vrnc
		    if r_corr>maxdata["r"] and r_corr > min_r:
                        #print("Updating slope: "+str(slopevec))
                        #print("mean: "+str(regmean))
			maxdata.update({"r":r_corr, "mean":regmean, "point":pt, "h1": h1, "h2":h2, "slope": slopevec})
	    track.extend([hit,maxdata["h1"],maxdata["h2"]])

            if maxdata["r"] == 0:  continue
            #find vec such that dot(V...hit...)<0 for pts outside of region.
            #Qvec is the point on the linreg line s.t. closest approach to etraj (ptraj).
            #Q = regmean + slopevec*tq   rp = r0p + up*tp
            #pclose = p1 + (p1-p2)*tp    rn = r0n + un*tn
            #r = regmean - p1
            #tq =  [ -r*slopevec + (r*(p1-p2))(slopevec*(p1-p2)) ] / [1 - (slopevec*(p1-p2))**2 ]
            regmean = maxdata["mean"]
            slopevec = maxdata["slope"]
            pt = maxdata["point"]
            #print("regmean: "+str(regmean))
            #print("e_traj0: "+str(e_traj))
            #print("e_traj: "+str(e_traj[0,:]))
            rvec = regmean - e_traj[0,:]  #etraj[0]=p1, etc
            pvec = e_traj[0,:] - e_traj[1,:]
            tq = ( -np.dot(rvec, slopevec) + np.dot(rvec,pvec)*np.dot(slopevec,pvec) ) / \
                ( 1 - (np.dot(slopevec,pvec))**2 )
            Qvec = regmean + slopevec*tq
            #print("regmean: "+str(regmean))
            #print("Qvec computed: "+str(Qvec))
            #print("e_traj: "+str(e_traj))
            #print("endv: "+str(np.array(hit)[1:4]))
            endVec = np.array(hit)[1:4] - Qvec
            prod = np.dot(slopevec, endVec)
            if prod > 0:  slopevec = slopevec*-1

            for h in track:  #Remove pts now so they won't get re-added in the next loop
                #print("Removing "+str(h))
                hitlist.remove(h)

            for h in hitlist2:
                hvec = np.array(h)[1:4]
                withinRegion = (np.dot(hvec - Qvec, slopevec) < 0)
		if distPtToLine(hvec,regmean,pt) <= trackdiam and withinRegion:
                    track.append(h)
                    hitlist.remove(h)
            if track != []:
                track.sort(key=lambda tup: tup[3], reverse=True)
                regtracklist.append(track)
                #print("Track: "+str(track))
    return regtracklist

def plotEvent(dataMap,track_list,regtrack_list):
    #Prepare arrays for plotting
    pltDict = [convertListToArr(dataMap[i]) for i in range(6)]

    tracks=[]
    rtracks=[]
    for trk in track_list:
        #print("Converting to list")
        tracks.append(convertListToArr(trk))
    for rtrk in regtrack_list:
        rtracks.append(convertListToArr(rtrk))

    #Plot all hits, color-coded by region
    fig=plt.figure(i)
    ax=Axes3D(fig)
    if pltDict[0]!=[]:
        ax.scatter(pltDict[0][:,3], pltDict[0][:,1], pltDict[0][:,2], c='r', label="Outside both radii")  #outside 68 cont
    if pltDict[1]!=[]:
        ax.scatter(pltDict[1][:,3], pltDict[1][:,1], pltDict[1][:,2], c='gray', label="Inside e- radius")
    if pltDict[2]!=[]:
        ax.scatter(pltDict[2][:,3], pltDict[2][:,1], pltDict[2][:,2], c='g', label="Inside photon radius")
    if pltDict[3]!=[]:
        ax.scatter(pltDict[3][:,3], pltDict[3][:,1], pltDict[3][:,2], c='m', label="Inside both radii")
    if pltDict[4]!=[]:
        ax.plot3D(pltDict[4][:,3], pltDict[4][:,1], pltDict[4][:,2], c='y', label="Projected electron trajectory")
    if pltDict[5]!=[]:
        ax.plot3D(pltDict[5][:,3], pltDict[5][:,1], pltDict[5][:,2], c='c', label="Projected photon trajectory")

    #Plot the tracks
    legendAdded=False
    for pltarr in tracks:
        if not legendAdded:
            ax.plot(pltarr[:,3],pltarr[:,1],pltarr[:,2], color='k', label="Ordinary tracks")
            legendAdded=True
        else:  ax.plot(pltarr[:,3],pltarr[:,1],pltarr[:,2], color='k')
    legendAdded=False
    for rpltarr in rtracks:
        if not legendAdded:
            ax.plot(rpltarr[:,3],rpltarr[:,1],rpltarr[:,2], 'b', label="Linreg tracks")
            legendAdded=True
        else:  ax.plot(rpltarr[:,3],rpltarr[:,1],rpltarr[:,2], 'b')
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    plt.legend(loc=4)
    #ax.set_xlim3d(200,350)
    plt.show()
 


#Main function.  Calls fillDataLists, findTracks, and plotEvent.
def analyze(inArr, min_len):  #inArr=array of all hits for all events; min_len=min track length
    i_end = 0
    i_start = 0
    normTracks = []
    regTracks = []
    allTracks = []
    missingEPNum = 0
    missingPNum = 0
    missingEventNum = 0
    events =  int(inArr[len(inArr)-1,0]) #- 850
    for i in range(1, events):  #i=current event being examined
        while not i_end == len(inArr) and i == inArr[i_end,0]:        #print('i_end='+str(i_end))
            i_end += 1
        if i_end == len(inArr):  i_end += 1
        if i%100 == 0:
            print("Processing event "+str(i))
        if i_start == i_end: 
            missingEventNum += 1
	    i_start = i_end  
            continue
        #Check for presence of e- traj.  If none, discard event.
        if inArr[i_start,6] == 4 and inArr[i_start+1,6] == 4 and not inArr[i_start+2,6] == 5 and not inArr[i_start+2,6] == 5:
            missingPNum += 1
        #    i_start = i_end
        #    continue
	if not (inArr[i_start,6] == 4 and inArr[i_start+1,6] == 4 and inArr[i_start+2,6]==5 and inArr[i_start+3,6]==5):
	    missingEPNum += 1
	    i_start = i_end
	    continue
	

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

        #skip = False
	fillDataLists(hits, categoryMap, inArr, i_start, i_end)
	#if skip:
	   # print('skipping in analyze')
	    #missingTrackNum += 1
	    #continue
        #print ("                                                      *****CONSIDERING NEW EVENT*****")
        #print("Finding tracks, first iteration")
        normtracklist = findTracks(hits, min_len)  #Perform full tracking algorithm
        normTracks.append(normtracklist)
        #print("Finding tracks, second iteration")

	#regtracklist = findregTracks(hits, min_len, etraj, ptraj)
        regtracklist = []
	regTracks.append(regtracklist)      
        allTracks.append(normtracklist+regtracklist)

        #print("Done.  Plotting...")
        #print(len(normtracklist+regtracklist))
	
        if True: #len(normtracklist+regtracklist) == 0:  #if more than 0 usable hits detected:
            print("Plotting event "+str(i)+"...")
            plotEvent(categoryMap, normtracklist, regtracklist)
	

        i_start = i_end
    eventsUsed = (events - missingEventNum - missingEPNum - missingPNum)
    print('Number of events missing: ' + str(missingEventNum))
    print('Number of events missing photon trajectory: ' + str(missingPNum))
    print('Number of events missing e- and photon trajectory: ' + str(missingEPNum))
    print('Fraction of  events used: ' + str(eventsUsed)+'/'+str(events))
    return allTracks


#BEGIN ACTUAL PROGRAM

min_track_len=3
filePath = '/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/LDMX-scripts/mipTracking/tracking_dev/'
bkg_file='hits_sim_bkg_1.txt'
sig_file='hits_sim_1000MeV_0.txt' #AND RECON  #'hits_Sig0_1000sig.txt'
bkgArr = np.loadtxt(filePath+bkg_file)
#sigArr = np.loadtxt(filePath+sig_file)

bkgATracks = analyze(bkgArr, min_track_len)
#sigATracks = analyze(sigArr, min_track_len)

print("Finished counting tracks.")

#Plotting

#sigNTracks = createHistoArr(sigATracks)
#bkgNTracks = createHistoArr(bkgATracks)
"""
#bkgTrkLens = createHistoArr(bkgATracks)
sigTrkLens = createHistoArr(sigATracks)

plt.figure(1)
plt.hist(bkgTrkLens, bins=20)
plt.xlabel("Track length")
plt.ylabel("Number of tracks")
plt.title("Histo of track lengths, all bkg events")
plt.show()

plt.figure(2)
plt.hist(sigTrkLens, bins=20)
plt.xlabel("Track length")
plt.ylabel("Number of tracks")
plt.title("Histo of track lengths, all sig events")
plt.show()
"""

sigNTracks = []
bkgNTracks = []
#for trks in sigATracks:
    #sigNTracks.append(len(trks))
for trks in bkgATracks:
    bkgNTracks.append(len(trks))

x_max=10  #Please don't make this an non-integer and ruin everything
adds = "vertical_slope"
qualifier = "vertical slopes allowed"


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

