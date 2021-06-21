import math
import numpy as np
# All space values in mm unless otherwise noted

# Gdml values
#Ecal
ecal_front_z = 240.5
sp_thickness = 0.001
clearance = 0.001
ECal_dz = 449.2
ecal_envelope_z = ECal_dz + 1
sp_ecal_front_z = ecal_front_z + (ecal_envelope_z - ECal_dz)/2 - sp_thickness/2 + clearance

sin60 = np.sin(np.radians(60))
module_radius = 85. # Dist form center to midpoint of side
module_side = module_radius/sin60
module_gap = 1.5 # Space between sides of side-by-side mods
cell_radius = 5 # Real circle
cellWidth = 8.7

# DetDescr
ecal_LAYER_MASK = 0x3F  # space for up to 64 layers
ecal_LAYER_SHIFT = 17
ecal_MODULE_MASK = 0x1F  # space for up to 32 modules/layer
ecal_MODULE_SHIFT = 12
ecal_CELL_MASK = 0xFFF  # space for 4096 cells/module (!)
ecal_CELL_SHIFT = 0

# Hcal
# Layer 1 has no absorber, layers 2 and 3 have absorber of different thickness
air_thick = 2
scint_thick = 20
hcal_back_dx = 3100
hcal_back_dy = 3100   
back_numLayers1 = 0        
back_numLayers2 = 100        
back_numLayers3 = 0        
back_abso2_thick = 25
back_abso3_thick = 50
back_layer1_thick = scint_thick + air_thick
back_layer2_thick = back_abso2_thick + scint_thick + 2.0*air_thick
back_layer3_thick = back_abso3_thick + scint_thick + 2.0*air_thick
hcal_back_dz1 = back_numLayers1*back_layer1_thick
hcal_back_dz2 = back_numLayers2*back_layer2_thick
hcal_back_dz3 = back_numLayers3*back_layer3_thick
hcal_back_dz = hcal_back_dz1 + hcal_back_dz2 + hcal_back_dz3

# Side HCal Layer component
sideTB_layers = 28
sideLR_layers = 26
side_abso_thick = 20
hcal_side_dz = 600

# Macro
back_start_z = 860.5 # ecal_front_z + hcal_side_dz + 20
hcal_dz = hcal_back_dz + hcal_side_dz

# DetDescr
# HcalSection BACK = 0, TOP = 1, BOTTOM = 2, LEFT = 4, RIGHT = 3
hcal_SECTION_MASK = 0x7 # space for up to 7 sections
hcal_SECTION_SHIFT = 18
hcal_LAYER_MASK = 0xFF  # space for up to 255 layers
hcal_LAYER_SHIFT = 10
hcal_STRIP_MASK = 0xFF  # space for 255 strips/layer
hcal_STRIP_SHIFT = 0

ecal_layerZs = ecal_front_z + np.array([7.850,   13.300,  26.400,  33.500,  47.950,
                                        56.550,  72.250,  81.350,  97.050,  106.150,
                                        121.850, 130.950, 146.650, 155.750, 171.450,
                                        180.550, 196.250, 205.350, 221.050, 230.150,
                                        245.850, 254.950, 270.650, 279.750, 298.950,
                                        311.550, 330.750, 343.350, 362.550, 375.150,
                                        394.350, 406.950, 426.150, 438.750        ])

ecal_zs_round = [round(z) for z in ecal_layerZs]
ecal_rz2layer = {}
for rz, i in zip(ecal_zs_round,range(1,35)):
    ecal_rz2layer[rz] = i

# For v12 reconstruction
mipSiEnergy = 0.130 # MeV
secondOrderEnergyCorrection = 4000./4010.
layerWeights = [
            1.675, 2.724, 4.398, 6.039, 7.696, 9.077, 9.630, 9.630, 9.630, 9.630, 9.630,
            9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630,
            9.630, 13.497, 17.364, 17.364, 17.364, 17.364, 17.364, 17.364, 17.364, 17.364,
            17.364, 8.990
            ]

# Arrays holding 68% containment radius per layer for different bins in momentum/angle
radius68_thetalt10_plt500 = [4.045666158618167, 4.086393662224346, 4.359141107602775, 4.666549994726691, 5.8569181911416015, 6.559716356124256, 8.686967529043072, 10.063482736354674, 13.053528344041274, 14.883496407943747, 18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406, 28.329169392203216, 29.468032123356345, 34.03271241527079, 35.03747443690781, 38.50748727211848, 39.41576583301171, 42.63622296033334, 45.41123601592071, 48.618139095742876, 48.11801717451056, 53.220539860213655, 58.87753380915155, 66.31550881539764, 72.94685877928593, 85.95506228335348, 89.20607201266672, 93.34370253818409, 96.59471226749734, 100.7323427930147, 103.98335252232795]
radius68_thetalt10_pgt500 = [4.081926458777424, 4.099431732299409, 4.262428482867968, 4.362017581473145, 4.831341579961153, 4.998346041276382, 6.2633736512415705, 6.588371889265881, 8.359969947444522, 9.015085558044309, 11.262722588206483, 12.250305471269183, 15.00547660437276, 16.187264014640103, 19.573764900578503, 20.68072032434797, 24.13797140783321, 25.62942209291236, 29.027596514735617, 30.215039667389316, 33.929540248019585, 36.12911729771914, 39.184563500620946, 42.02062468386282, 46.972125628650204, 47.78214816041894, 55.88428562462974, 59.15520134927332, 63.31816666637158, 66.58908239101515, 70.75204770811342, 74.022963432757, 78.18592874985525, 81.45684447449884]
radius68_theta10to20 = [4.0251896715647115, 4.071661598616328, 4.357690094817289, 4.760224640141712, 6.002480766325418, 6.667318981016246, 8.652513285172342, 9.72379373302137, 12.479492693251478, 14.058548828317289, 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302, 31.768946746958296, 35.79247330197688, 37.27810357669942, 41.657281051476545, 42.628141392692626, 47.94208483539388, 49.9289473559796, 54.604030254423975, 53.958762417361655, 53.03339560920388, 57.026277390001425, 62.10810455035879, 66.10098633115634, 71.1828134915137, 75.17569527231124, 80.25752243266861, 84.25040421346615, 89.33223137382352, 93.32511315462106]
radius68_thetagt20 = [4.0754238481177705, 4.193693485630508, 5.14209420056253, 6.114996249971468, 7.7376807326481645, 8.551663213602291, 11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604, 25.04088272634407, 28.853696411302344, 34.72538105333071, 40.21218694947545, 46.07344239520299, 50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047, 74.82378572939959, 89.4528387422834, 93.18228303096758, 92.51751129204555, 98.80228884380018, 111.17537347472128, 120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185, 177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696]

# Constants needed for the longitudinal segmentation variables
segLayers = [0, 6, 17, 34]
nRegions = 5
nSegments = len(segLayers) - 1

# Simple class for storing hit data
class HitData:
    def __init__(self,pos=None,layer=None):
        self.pos = pos
        self.layer = layer

###########################
# Miscellaneous functions
###########################

# Reconstructed energy from sim energy
def recE(siEnergy, layer):
    return ((siEnergy/mipSiEnergy)*layerWeights[layer-1]+siEnergy)*secondOrderEnergyCorrection

# 2D Rotation
def rotate(point,ang): # move to math eventually
    ang = np.radians(ang)
    rotM = np.array([[np.cos(ang),-np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    return list(np.dot(rotM,point))

# Get layer number from hitZ (Replaced by ecal_layer)
def layerofHitZ(hitZ, index = 0):
    num = ecal_rz2layer[ round(hitZ) ]
    if index == 1: return num
    elif index == 0: return num - 1
    else: print('index should be 0 or 1')

# Get Z in ecal_layerZs from hitZ (Replaced by layerZofHit)
def layerZofHitZ(hitZ):
    return ecal_layerZs[ layerofHitZ(hitZ) ]

# Get layerZ from hit
def layerZofHit(hit):
    return ecal_layerZs[ecal_layer(hit)]

# Project poimt to z_final
def projection(pos_init, mom_init, z_final): # infty >.<
    x_final = pos_init[0] + mom_init[0]/mom_init[2]*(z_final - pos_init[2])
    y_final = pos_init[1] + mom_init[1]/mom_init[2]*(z_final - pos_init[2])
    return (x_final, y_final)

# List of projected (x,y)s at each ECal layer
def layerIntercepts(pos,mom,layerZs=ecal_layerZs):
    return [projection(pos,mom,z) for z in layerZs]

# Magnitude of whatever
def mag(iterable):
    return math.sqrt(sum([x**2 for x in iterable]))

# Return normalized np array
def unit(arrayy):
    return np.array(arrayy)/mag(arrayy)

# Dot iterables
def dot(i1, i2):
    return sum( [i1[i]*i2[i] for i in range( len(i1) )] )

# Distance detween points
def dist(p1, p2):
    return math.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))

# Distance between a point and the nearest point on a line defined by endpoints
def distPtToLine(h1,p1,p2):
    return np.linalg.norm(np.cross((np.array(h1)-np.array(p1)),
        (np.array(h1)-np.array(p2)))) / np.linalg.norm(np.array(p1)-np.array(p2))

# Minimum distance between lines, each line defined by two points
def distTwoLines(h1,h2,p1,p2):
    e1  = unit( h1 - h2 )
    e2  = unit( p1 - p2 )
    crs = np.cross(e1,e2) # Vec perp to both lines
    if mag(crs) != 0:
        return abs( np.dot( crs,h1-p1) )
    else: # Lines are parallel; need different method
        return mag( np.cross(e1,h1-p1) )

# Angle between vectors (with z by default)
def angle(vec, units, vec2=[0,0,1]):
    if units=='degrees': return math.acos( dot( unit(vec), unit(vec2) ) )*180.0/math.pi
    elif units=='radians': return math.acos( dot( unit(vec), unit(vec2) ) )
    else: print('\nSpecify valid angle unit ("degrees" or "randians")')

# Get np.ndarray of hit position
def pos(hit):
    return np.array( ( hit.getXPos(), hit.getYPos(), hit.getZPos() ) )

###########################
# Get hitID-related info
###########################

# Get layerID from ecal hit
def ecal_layer(hit):
    return (hit.getID() >> ecal_LAYER_SHIFT) & ecal_LAYER_MASK

# Get moduleID from ecal hit
def ecal_module(hit):
    return (hit.getID() >> ecal_MODULE_SHIFT) & ecal_MODULE_MASK

# Get cellID from ecal hit
def ecal_cell(hit):
    return (hit.getID() >> ecal_CELL_SHIFT) & ecal_CELL_MASK

# Get sectionID from hcal hit
def hcal_section(hit):
    return (hit.getID() >> hcal_SECTION_SHIFT) & hcal_SECTION_MASK

# Get layerID from hcal hit
def hcal_layer(hit):
    return (hit.getID() >> hcal_LAYER_SHIFT) & hcal_LAYER_MASK

# Get stripID from hcal hit
def hcal_strip(hit):
    return (hit.getID() >> hcal_STRIP_SHIFT) & hcal_STRIP_MASK

###########################
# Get e/gamma SP hit info
###########################

# Get electron target scoringplane hit
def electronTargetSPHit(targetSPHits):

    targetSPHit = None
    pmax = 0
    for hit in targetSPHits:

        if hit.getPosition()[2] > sp_thickness + sp_thickness + 0.5 or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1 or\
                hit.getPdgID() != 11:
            continue

        if mag(hit.getMomentum()) > pmax:
            targetSPHit = hit
            pmax = mag(targetSPHit.getMomentum())

    return targetSPHit

# Get electron ecal scoringplane hit
def electronEcalSPHit(ecalSPHits):

    eSPHit = None
    pmax = 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > sp_ecal_front_z + sp_thickness/2 or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1 or\
                hit.getPdgID() != 11:
            continue

        if mag(hit.getMomentum()) > pmax:
            eSPHit = hit
            pmax = mag(eSPHit.getMomentum())

    return eSPHit

# Get electron target and ecal SP hits
def electronSPHits(ecalSPHits, targetSPHits):

    ecalSPHit   = electronEcalSPHit(ecalSPHits)
    targetSPHit = electronTargetSPHit(tartgetSPHits)

    return ecalSPHit, targetSPHit

# Return photon position and momentum at target
def gammaTargetInfo(eTargetSPHit):

    gTarget_pvec = np.array([0,0,4000]) - np.array(eTargetSPHit.getMomentum())

    return eTargetSPHit.getPosition(), gTarget_pvec

# Get photon ecal scoringplane hit
def gammaEcalSPHit(ecalSPHits):

    gSPHit = None
    pmax = 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > sp_ecal_front_z + sp_thickness/2 or\
                hit.getMomentum()[2] <= 0 or\
                not (hit.getPdgID() in [-22,22]):
            continue

        if mag(hit.getMomentum()) > pmax:
            gSPHit = hit
            pmax = mag(gSPHit.getMomentum())

    return gSPHit

# Get electron and photon ecal scoringplane hits
def elec_gamma_ecalSPHits(ecalSPHits):

    eSPHit = electronEcalSPHit(ecalSPHits)
    gSPHit = gammaEcalSPHit(ecalSPHits)

    return eSPHit, gSPHit
