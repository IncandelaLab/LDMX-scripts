import math
import numpy as np
from mods import physTools


# NOTE: Don't forget to order hits by reverse zpos before using the nXTracks funcs
# This is assumed to make use of some algorithm short cuts

##########################
# First layer with hit within one cell of the photon trajectory AND
# Number of hits within one cell of the photon trajectory
##########################
def nearPhotonInfo(trackingHitList, g_trajectory, returnLayer=True, returnNumber=True):

    layer = 33
    n = 0
    for hit in trackingHitList:

        # Near the photn trajectory
        if physTools.dist( physTools.pos(hit)[:2],
                g_trajectory[ physTools.ecal_layer( hit ) ] ) < physTools.cellWidth:
            n += 1

            # Earliest layer
            if physTools.ecal_layer( hit ) < layer:
                layer = physTools.ecal_layer( hit )

    # Prepare and return desired output
    out = []
    if returnLayer: out.append(layer)
    if returnNumber: out.append(n)

    return out

##########################
# Straight tracks
##########################

# Based on previous python; All of v9 analysis done with this
def findStraightTracks(hitlist, etraj_ends, ptraj_ends,\
                        mst = 2, returnN=True, returnHitList = False, returnTracks = False):
    
    # sort hitlist by decreasing distance from the ECal face
    hitlist.sort(key=lambda h: h.getZPos(), reverse=True)
    
    strtracklist = []   # Initialize output
    hitscopy = hitlist  # Need this because hitlist gets eddited
        
    for hit in hitlist:  #Go through all hits, starting at the back of the ecal
        track = [hit]
        currenthit = hit  #Stores "trailing" hit in track being constructed
        possibleNeigh = False
        for h in hitscopy:
            if h.getZPos() == currenthit.getZPos():
                possibleNeigh = True  #Optimization
                continue
            if not possibleNeigh:  continue
            # if currenthit.getZPos() - h.getZPos() > 36:  #Optimization
            # Optimization, ECal geometry independent
            if physTools.layerofHitZ(currenthit.getZPos())\
                                        - physTools.layerofHitZ(h.getZPos()) > 3:  
                possibleNeigh = False
                continue
            neighFound = (
                    (physTools.layerofHitZ(h.getZPos()) ==\
                            physTools.layerofHitZ(currenthit.getZPos()) - 1 or\
                     physTools.layerofHitZ(h.getZPos()) ==\
                            physTools.layerofHitZ(currenthit.getZPos()) -2) and\
                     h.getXPos() == currenthit.getXPos() and\
                     h.getYPos() == currenthit.getYPos() )
            if neighFound:
                track.append(h)
                currenthit = h
        # Too short
        if len(track) < mst: continue

        # If it's exactly the min, it has to be very close to ptraj
        if len(track) == mst: 
            for hitt in track:
                if physTools.distPtToLine( physTools.pos(hitt),
                        ptraj_ends[0], ptraj_ends[1] ) > physTools.cellWidth - 0.5:
                    break
                continue

        # Check that the track approaches the photon's and not the electron's
        trk_s = np.array( (track[ 0].getXPos(), track[ 0].getYPos(), track[ 0].getZPos() ) )
        trk_e = np.array( (track[-1].getXPos(), track[-1].getYPos(), track[-1].getZPos() ) )
        closest_e = physTools.distTwoLines( etraj_ends[0], etraj_ends[1], trk_s, trk_e )
        closest_p = physTools.distTwoLines( ptraj_ends[0], ptraj_ends[1], trk_s, trk_e )
        if closest_p > physTools.cellWidth and closest_e < 2*physTools.cellWidth:
            continue

        # Remove hits in current track from further consideration
        for h in track:
            hitlist.remove(h)

        # Append track to track list
        strtracklist.append(track)

    # Merge nearby straight tracks
    # Criteria: consider tail of track. 
    #           Merge if head of next track is 1/2 layers behind, within 1 cell of xy position.
    for base_track in strtracklist:
        tail_hit = base_track[-1]
        for checking_track in strtracklist:
            head_hit = checking_track[0]
            # if 1-2 layers behind, and xy within one cell...
            if abs(physTools.layerofHitZ(tail_hit.getZPos()) - \
                            physTools.layerofHitZ(head_hit.getZPos())) < 3 and \
                physTools.dist( [tail_hit.getXPos(), tail_hit.getYPos()], \
                                [head_hit.getXPos(), head_hit.getYPos()] ) < physTools.cellWidth:
                # ...then append the second track to the first one and delete it
                for hit in checking_track:
                    base_track.append(hit)
                strtracklist.remove(checking_track)
            
    # if len(strtracklist) != 0:
    #     print("strtracklist afetr merging: ",\
    #         np.array([np.array([(hit.getXPos(), hit.getYPos(), hit.getZPos()) for hit in track])\
    #                                                                           for track in strtracklist]))
    #     print("N = ", len(strtracklist))
    
    # Prepare and return desired output
    out = []
    if returnN: out.append( len(strtracklist) )
    if returnHitList: out.append( hitlist )
    if returnTracks: out.append( strtracklist )

    return out


# Based on C++ Analyzer
def nStraightTracks_c(trackingHitList, e_traj_ends, g_traj_ends):

    nTracks = 0

    # Seed a track with each hit
    iHit = 0
    while iHit < len(trackingHitList):
        track = 34*[999]
        track[0] = iHit
        currentHit = iHit
        trackLen = 1        

        # Search for hits in next two layers
        jHit = 0
        while jHit < len(trackingHitList):

            if trackingHitList[jHit].layer == trackingHitList[currentHit].layer or\
                    trackingHitList[jHit].layer > trackingHitList[currentHit].layer + 2:
                jHit += 1 # Don't keep checking this hit over and over again
                continue # Continue if not in the right range

            # If it's also directly behind the current hit, add it to the current track
            if trackingHitList[jHit].pos[:1] == trackingHitList[currentHit].pos[:1]:
                track[trackLen] = jHit
                currentHit = jHit # Update end of track
                trackLen += 1

            jHit += 1 # Move j along

        # Confirm if track is valid
        if trackLen >= 2: # Set min track length
            
            # Make sure the track is near the photon trajectory and away from the electron
            closest_e = physTools.distTwoLines( trackingHitList[ track[0         ] ].pos,
                                                trackingHitList[ track[trackLen-1] ].pos,
                                                e_traj_ends[0], e_traj_ends[1]
                                              )
            closest_g = physTools.distTwoLines( trackingHitList[ track[0         ] ].pos,
                                                trackingHitList[ track[trackLen-1] ].pos,
                                                g_traj_ends[0], g_traj_ends[1]
                                              )
            if closest_g > physTools.cellWidth and closest_e < 2*physTools.cellWidth:
                iHit += 1; continue
            if trackLen < 4 and closest_e > closest_g:
                iHit += 1; continue

            # If valid track is found, remove hits in track from hitList
            for kHit in range(trackLen):
                trackingHitList.pop( track[kHit] - kHit) 

            # nStraightTracks++
            nTracks += 1

            # Decrease iHit because the *current" seed will have been removed
            iHit -= 1

        iHit += 1 # Move iHit along

        # Possibley merge tracks later

    # return the trackingHitlist as is so Linreg doesn't look through 'removed' points
    return nTracks, trackingHitList

##########################
# Linreg tracks
##########################

# Not in v9 feature list so postponing this
def nLinregTracks(trackingHitList, e_traj_ends, g_traj_ends):

    nTracks = 0

    # Seed a track with each hit
    iHit = 0

    return 0
