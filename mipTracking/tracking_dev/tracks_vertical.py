import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Implementation of MIP tracking code
#Vertical edition:  Also consider tracks w/ multiple hits in one layer (high-theta)

def neighbors(h1,h2,mult):
    #sqrt(76)mm ~ 8mm = neighbors are within one cell.  This can be tweaked.  POSSIBLE:  Layer-dependent neighbor function??
    #Note:  This only compares x and y coordinates.  Two hits in different layers w/ identical xy coords will always be neighbors.
    return np.sqrt((h1[1]-h2[1])**2 + (h1[2]-h2[2])**2) < 8.7*mult  #*2

def neighInNextLayer(h1,h2):
    #check whether h2 is a layer in front of h1 and a neighbor of h1; if so, return true
    return h1[3]-1==h2[3] and neighbors(h1,h2,1)

def isolatedEnd(hit,hitlist):
    #Original:  look through all hits h in hitlist.  If h is in same layer as hit (or 1 layer back) AND neighbor, not isolated.
    #NOTE:  Assumes that the FIRST hit in hitlist is the candidate End.  This is because 1) hitlist is sorted by decreasing z, and because everything that's already part of a track gets removed from the list.
    #Since hits are sorted by decreasing layer, this should be reasonably fast.
    #print("Checking for isolated end")
    for h in hitlist:
        #if(neighbors(hit,h)):  print("Hit and h are neighbors; checking if nearby...")
        if h==hit:  continue
        if (h[3]==hit[3] or h[3]==hit[3]+1) and neighbors(hit,h,1):
            #print("Found nearby hit:  "+str(h))
            return False
    #print("Event is isolated")
    return True

def neighAbove(h1,h2):
    #Check whether h2 is an upper/lower neighbor of h2, for upper/lower tracks.
    #For now, checks all same-layer neighbors, plus next-layer neighbors EXCEPT the center.
    if h1==h2:  return False
    return (h1[3]==h2[3] and neighbors(h1,h2)) or (h1[3]==h2[3]-1 and neighbors(h1,h2) and (h1[0]!=h2[0] and h1[1]!=h2[1]))

def isolatedAbove(hit,hitlist):
    #Check and see if pt is a valid start of a high-theta track
    #For now, return false if there exists a pt that would call hit a neighborAbove.
    for h in hitlist:
        if h==hit:  continue
        #**WARNING:** See *PROBLEM* in my notes
        if (h1[3]==h2[3] and neighbors(h1,h2,1)) or (h1[3]==h2[3]-1 and neighbors(h1,h2,2) and (h1[0]!=h2[0] and h1[1]!=h2[1])):
            return False
    return True

def findCentroid(hitlist):
    sumx=0
    sumy=0
    sumz=0
    for hit in hitlist:
        sumx+=hit[1]
        sumy+=hit[2]
        sumz+=hit[3]
    return (sumx/len(hitlist),sumy/len(hitlist),sumz/len(hitlist))

def dist(p1,p2):
    #Adding a factor of 20, since layer # != actual distance in mm
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(20*p1[2]-20*p2[2])**2)



def detectHits(inArr, showEvents):
    i_end=0
    i_start=0
    allTracks=[]
    numTracks=[]
    for i in range(1,int(inArr[len(inArr)-1,0])):  #i=current event being examined
        while not i_end==len(inArr) and i==inArr[i_end,0]:        #print('i_end='+str(i_end))
            #print(fileArr[i_end,:])
            i_end+=1
        if i%100==0:
            print("Processing event "+str(i))
        if i_start==i_end:  #no events found
            #print("No hits found for event "+str(i))
            continue
        #Now, the range i_start to i_end-1 of fileArr[] contains all points for event i.
        #To make things easier, create list of event coordinates:
        #print("Processing event "+str(i))
	hits=[]
	if i_end==len(inArr):  i_end+=1
	for j in range(i_start,i_end-1):
            hits.append((j,inArr[j,1],inArr[j,2],inArr[j,3]))  #NOTE:  Intentionally hanging onto hit ID j
	#Sort array of positions by z position to help speed up tracking alg (by DECREASING z)
	hits.sort(key=lambda tup: tup[3], reverse=True)  #Sort by 3rd elem in arr in reverse order
	#print(hits)
	
	#TRACKING STARTS HERE
	
	#Original algorithm:
	#Start at back of Ecal, iterate through all hits.  If hit has neighbor in prev layer, add hit to track.
	#  Neighbor=hit within 76 mm
	#  NOTE:  I don't want to iterate through all hits in the event to find these...but could if necessary.
	
	#Alg v2:  seed event has no adjacent hits in same layer/layer behind (end must be isolated)
	#Alg v3:  Same, plus any hits with immediate neighbors are ignored.
	
	#Ideally, should save the track hit info too, for debugging/checking/etc.
	#Original code stores array of hit IDs for this.  Probably easiest to store the col the hit is in.
	
	#ORIGINAL ALGORITHM:  O(n^2), I think
	tracklist=[]
	hitscopy=hits
	#print(hitscopy)
	for hit in hits:  #Go through all hits, starting at the back of the ecal (note: order doesn't matter anymore)
	    if isolatedEnd(hit, hits):
		#Create new track, then sift through all other hits to find others in same track
		track=[]
		track.append(hit)
		currenthit=hit  #Stores "trailing" hit in track being constructed
		for h in hitscopy:  #NOTE:  Because hitscopy is sorted, and because hit h' can't be a parent of hit h track if h' is deeper in the ecal,
		    #...searching over all hits in hitscopy before h to find a new hit to add to h's track is unnecessary.
		    if h==currenthit: continue
		    if neighInNextLayer(currenthit,h):
			track.append(h)
			currenthit=h
                tracklist.append(track)

        #Now go over all pts again in a second pass, looking for high-theta tracks
        #First, sort all tracks by distance from centroid.  This breaks "ties".
        tracklist_vert=[]
        centroid=findCentroid(hits)
        hits.sort(key=lambda tup: dist(tup,centroid), reverse=True)
        hitscopy=hits
        for hit in hits:
            if isolatedAbove(hit, hits):
                track=[]
                track.append(hit)
                currenthit=hit
                for h in hitscopy:
                    if h==currenthit: continue
                    if neighAbove(currenthit,h):
                        track.append(h)
                        currenthit=h
                tracklist_vert.append(track)

        trackss=0
        tracksv=0
        for track in tracklist:
            if len(track)>=min_track_len:  trackss+=1  #Only tracks with a length>=min_track_len count
        for track_v in tracklist_vert:
            if len(track_v)>=min_track_len:  tracksv+=1
        numTracks.append(trackss)
        allTracks.append(tracklist)

        """
        #TEMP--plotting
        if trackss==0:   #If no tracks, then plot all points
            fig=plt.figure(i)
            ax=Axes3D(fig)
            trails=[]
            for trk in tracklist:
                pltarr=np.zeros((len(trk),3))
                hitnum=0
                for hit in trk:
                    pltarr[hitnum,0]=hit[1]
                    pltarr[hitnum,1]=hit[2]
                    pltarr[hitnum,2]=hit[3]
                    hitnum+=1
                trails.append(pltarr)
            ax.scatter(inArr[i_start:i_end-1,3],inArr[i_start:i_end-1,1],inArr[i_start:i_end-1,2], c = 'r', marker='o')
            for pltarr in trails:
                #if len(pltarr)<3: continue  #If the track is 3 hits or longer...
                ax.plot(pltarr[:,2],pltarr[:,0],pltarr[:,1], color='b')
            #ax.plot([1,20],[0,0],[0,0],color='r')
            ax.set_xlabel("z (layer #)")
            ax.set_ylabel("x (mm)")
            ax.set_zlabel("y (mm)")
            plt.show()
        """

        i_start=i_end
    return numTracks, allTracks



min_track_len=4
sig_file='hits_Sig0_1000sig.txt'
bkg_file='hits_Bkg0_bkg.txt'
#adds = 'Boop0'
filePath = "/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hitPlotting/"
#fileName = "hits_" + str(adds) + "_1000sig.txt"
#bkgArr = np.loadtxt('test_track.txt')
bkgArr = np.loadtxt(bkg_file)
#test_track.py has one 5-particle track, two non-track events.
sigArr = np.loadtxt(sig_file)

#Go through fileArr, find indices i and j where the next desired event is
#plot over those indices

#Looping over all events:
nEventsBkg=len(bkgArr)
nEventsSig=len(sigArr)


bkgNTracks, bkgATracks = detectHits(bkgArr,False)
sigNTracks, sigATracks = detectHits(sigArr,True)

print("Finished counting tracks.")


#Plotting

def createHistoArr(trkArr):
    trkLens=[]
    for trklist in trkArr:
        for trk in trklist:
            trkLens.append(len(trk))
    return trkLens

bkgTrkLens = createHistoArr(bkgATracks)
sigTrkLens = createHistoArr(sigATracks)

"""
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

plt.figure(3)
plt.hist(bkgNTracks, bins=10, range=(0,10))
plt.title("Plain, bkg, min len="+str(min_track_len))
plt.xlabel("Tracks per event")
plt.ylabel("Number of events")
plt.show()

plt.figure(4)
plt.hist(sigNTracks, bins=10, range=(0,10))
plt.title("Plain, sig, min len="+str(min_track_len))
plt.xlabel("Tracks per event")
plt.ylabel("Number of events")
plt.show()


"""
#3d plotting:  Find bkg events with no tracks (len=3, 2?), then plot all corresponding events.
eventnum=1  #NOTE:  Eventnum may not be the event id number, since events w no hits aren't included in the bkgNTracks array
for ne in bkgNTracks:
    print(ne)
    if ne==0:
        print("Event "+str(eventnum)+" has no tracks")
        #Loop through the background txt array until the corresp event is found, then grab all hits from it.
        lineno=0
        while bkgArr[lineno,0] != eventnum:  lineno+=1
        start_index=lineno
        while bkgArr[lineno,0] == eventnum:  lineno+=1
        fig=plt.figure(eventnum)
        ax=Axes3D(fig)
        ax.scatter(bkgArr[start_index:lineno,3],bkgArr[start_index:lineno,1],bkgArr[start_index:lineno,2], c = 'r', marker='o')
        ax.set_xlabel("z (layer #)")
        ax.set_ylabel("x (mm)")
        ax.set_zlabel("y (mm)")
        plt.show()

    eventnum+=1
"""

