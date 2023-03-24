import math
import numpy as np
import copy

# Constant defining the clearance between volumes
clearance = 0.001
# Thickness of scoring planes
sp_thickness = 0.001

# Target GDML
# Position
target_z = 0.0
# Tungsten X0 = .3504 cm
# Target thickness = .1X0
target_thickness = 0.3504
# Target dimensions
target_dim_x = 40.
target_dim_y = 100.

# Surround the target with scoring planes
sp_target_down_z = target_z + target_thickness/2 + sp_thickness/2 + clearance
sp_target_up_z = target_z - target_thickness/2 - sp_thickness/2 - clearance

# Trigger scintillator GDML
# Trigger scintillator positions
trigger_pad_thickness = 4.5
trigger_pad_bar_thickness = 2
trigger_pad_bar_gap = 0.3
trigger_pad_dim_x = target_dim_x
trigger_pad_dim_y = target_dim_y
trigger_bar_dx = 40
trigger_bar_dy = 3
number_of_bars = 25

trigger_pad_offset = (target_dim_y - (number_of_bars*trigger_bar_dy + (number_of_bars - 1)*trigger_pad_bar_gap))/2

# Trigger pad distance from the target is -2.4262mm
trigger_pad_up_z = target_z - (target_thickness/2) - (trigger_pad_thickness/2) - clearance
# Trigger pad distance from the target is 2.4262mm
trigger_pad_down_z = target_z + (target_thickness/2) + (trigger_pad_thickness/2) + clearance

# Place scoring planes downstream of each trigger scintillator array
sp_trigger_pad_down_l1_z = trigger_pad_down_z - trigger_pad_bar_gap/2 + sp_thickness/2 + clearance
sp_trigger_pad_down_l2_z = trigger_pad_down_z + trigger_pad_bar_gap/2 + trigger_pad_bar_thickness + sp_thickness/2 + clearance
sp_trigger_pad_up_l1_z = trigger_pad_up_z - trigger_pad_bar_gap/2 + sp_thickness/2 + clearance
sp_trigger_pad_up_l2_z = trigger_pad_up_z + trigger_pad_bar_gap/2 + trigger_pad_bar_thickness + sp_thickness/2 + clearance

# ECal GDML
# ECal layer thicknesses
Wthick_A_dz = 0.75
W_A_dz = 0.75
Wthick_B_dz = 2.25
W_B_dz = 1.5
Wthick_C_dz = 3.5
W_C_dz = 1.75
Wthick_D_dz = 7.0
W_D_dz = 3.5
CFMix_dz = 0.05
CFMixThick_dz = 0.2
PCB_dz = 1.5
Si_dz = 0.5
C_dz = 0.5
Al_dz = 2.0

# Air separating sheets of Al or W with PCB motherboard
# Limited by construction abilities 
FrontTolerance = 0.5

# Gap between layers
BackTolerance = 0.5

# Air separating PCBs from PCB MotherBoards
PCB_Motherboard_Gap = 2.3

# Air separating Carbon sheets in the middle of a layer
CoolingAirGap = 4.0

# Preshower thickness is 20.1mm
preshower_Thickness = Al_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                      + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + CoolingAirGap\
                      + 2.*C_dz + CFMixThick_dz + Si_dz + CFMix_dz + PCB_dz\
                      + PCB_Motherboard_Gap + PCB_dz + BackTolerance

# Layer A thickness is 20.35mm
layer_A_Thickness = Wthick_A_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_A_dz + C_dz\
                    + CoolingAirGap + C_dz + W_A_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Layer B thickness is 22.35mm
# GDML comment indicates that this is 22.35mm, but the actual value is 23.35mm!
layer_B_Thickness = Wthick_B_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_B_dz + C_dz\
                    + CoolingAirGap + C_dz + W_B_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Layer C thickness is 25.1mm
layer_C_Thickness = Wthick_C_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_C_dz + C_dz\
                    + CoolingAirGap + C_dz + W_C_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Layer D thickness is 32.1mm
layer_D_Thickness = Wthick_D_dz + FrontTolerance + PCB_dz + PCB_Motherboard_Gap\
                    + PCB_dz + CFMix_dz + Si_dz + CFMixThick_dz + W_D_dz + C_dz\
                    + CoolingAirGap + C_dz + W_D_dz + CFMixThick_dz + Si_dz\
                    + CFMix_dz + PCB_dz + PCB_Motherboard_Gap + PCB_dz\
                    + BackTolerance

# Number of layers
ecal_A_layers = 1
ecal_B_layers = 1
ecal_C_layers = 9
ecal_D_layers = 5

# ECal thickness is 449.2mm
# GDML comment indicates that this is 449.2mm, but the actual value is 450.2mm!
ECal_dz = preshower_Thickness\
          + layer_A_Thickness*ecal_A_layers\
          + layer_B_Thickness*ecal_B_layers\
          + layer_C_Thickness*ecal_C_layers\
          + layer_D_Thickness*ecal_D_layers

sqrt3 = np.sqrt(3)

module_gap = 1.5  # Flat-to-flat gap between modules
module_radius = 85.  # Center-to-flat radius of one module

# ECal width and height
ECal_dx = module_radius*10./sqrt3 + sqrt3*module_gap
ECal_dy = module_radius*6. + module_gap*2.

# Distance from target to the ECal parent volume
# The calorimeter is an additional .5mm downstream at 220.5mm
# GDML comment indicates that this is 220.5mm, but the actual value is clearly 240.5mm!
ecal_front_z = 240.

side_Ecal_dx = 800.
side_Ecal_dy = 600.

# Dimensions of ECal parent volume
# The size is set to be 1mm larger than the thickness of the ECal calculated above
ecal_envelope_x = side_Ecal_dx
ecal_envelope_y = side_Ecal_dy
ecal_envelope_z = ECal_dz + 1

# Surround the ECal with scoring planes
# sp_ecal_front_z = ecal_front_z + (ecal_envelope_z - ECal_dz)/2 - sp_thickness/2 + clearance
sp_ecal_front_z = ecal_front_z - sp_thickness/2 - clearance      # v14
sp_ecal_back_z = ecal_front_z + ECal_dz + (ecal_envelope_z - ECal_dz)/2 + sp_thickness/2
sp_ecal_top_y = ECal_dy/2 + sp_thickness/2
sp_ecal_bot_y = -ECal_dy/2 - sp_thickness/2
sp_ecal_left_x = -ECal_dx/2 - sp_thickness/2
sp_ecal_right_x = ECal_dx/2 + sp_thickness/2
sp_ecal_mid_z = ecal_front_z + ECal_dz/2 + (ecal_envelope_z - ECal_dz)/2

sin60 = np.sin(np.radians(60))
module_side = module_radius/sin60
cell_radius = 5
cellWidth = 8.7

# ECal DetDescr
ecal_LAYER_MASK = 0x3F  # Space for up to 64 layers
ecal_LAYER_SHIFT = 17
ecal_MODULE_MASK = 0x1F  # Space for up to 32 modules/layer
ecal_MODULE_SHIFT = 12
ecal_CELL_MASK = 0xFFF  # Space for 4096 cells/module
ecal_CELL_SHIFT = 0

# ecal_layerZs = ecal_front_z  + (ecal_envelope_z - ECal_dz)/2 + np.array([7.850,   13.300,  26.400,  33.500,  47.950,
#                                                                          56.550,  72.250,  81.350,  97.050,  106.150,
#                                                                          121.850, 130.950, 146.650, 155.750, 171.450,
#                                                                          180.550, 196.250, 205.350, 221.050, 230.150,
#                                                                          245.850, 254.950, 270.650, 279.750, 298.950,
#                                                                          311.550, 330.750, 343.350, 362.550, 375.150,
#                                                                          394.350, 406.950, 426.150, 438.750])

## v14 ecal layer zpos
Si_zpos = [7.932,   14.532,  32.146,  40.746,  58.110, 
           67.710,  86.574,  96.774,  115.638, 125.838,
           144.702, 154.902, 173.766, 183.966, 202.830,
           213.030, 231.894, 242.094, 260.958, 271.158,
           290.022, 300.222, 319.086, 329.286, 351.650,
           365.250, 387.614, 401.214, 423.578, 437.178,
           459.542, 473.142, 495.506, 509.106]
ecal_layerZs = ecal_front_z + np.array(Si_zpos)

# HCal GDML
# Width and height of the envelope for the side and back HCal
# Must be the maximum of back hcal dx and side hcal dx
hcal_envelope_dx = 3100.
hcal_envelope_dy = 3100.

# Common HCal components
air_thick = 2.
scint_thick = 20.

# Back HCal Layer component
# Layer 1 has no absorber, layers 2 and 3 have absorber of different thickness
hcal_back_dx = 3100.
hcal_back_dy = 3100.
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
side_abso_thick = 20.

# side_dz has to be greater than side_Ecal_dz
hcal_side_dz = 600

# Total calorimeter thickness
hcal_dz = hcal_back_dz + hcal_side_dz

back_start_z = 860.5 # ecal_front_z + hcal_side_dz + 20

# Surround the HCal with scoring planes
sp_hcal_front_z = ecal_front_z - sp_thickness/2 + clearance
sp_hcal_back_z = ecal_front_z + hcal_back_dz + hcal_side_dz + sp_thickness/2
sp_hcal_top_y = hcal_back_dy/2 + sp_thickness/2
sp_hcal_bot_y = -hcal_back_dy/2 - sp_thickness/2
sp_hcal_left_x = -hcal_back_dx/2 - sp_thickness/2
sp_hcal_right_x = hcal_back_dx/2 + sp_thickness/2
sp_hcal_mid_z = ecal_front_z + hcal_dz/2

# HCal DetDescr
# HcalSection BACK = 0, TOP = 1, BOTTOM = 2, LEFT = 4, RIGHT = 3
hcal_SECTION_MASK = 0x7  # Space for up to 7 sections
hcal_SECTION_SHIFT = 18
hcal_LAYER_MASK = 0xFF  # Space for up to 255 layers
hcal_LAYER_SHIFT = 10
hcal_STRIP_MASK = 0xFF  # Space for 255 strips/layer
hcal_STRIP_SHIFT = 0

ecal_zs_round = [round(z) for z in ecal_layerZs]
ecal_rz2layer = {}
for rz, i in zip(ecal_zs_round,range(1,35)):
    ecal_rz2layer[rz] = i

# For v12 reconstruction
mipSiEnergy = 0.130 # MeV
secondOrderEnergyCorrection = 4000./4010.
layerWeights = [1.675,  2.724,  4.398,  6.039,  7.696,
                9.077,  9.630,  9.630,  9.630,  9.630,
                9.630,  9.630,  9.630,  9.630,  9.630,
                9.630,  9.630,  9.630,  9.630,  9.630,
                9.630,  9.630,  9.630,  13.497, 17.364,
                17.364, 17.364, 17.364, 17.364, 17.364,
                17.364, 17.364, 17.364, 8.990]

# Arrays holding 68% containment radius/layer for different bins in momentum/angle
radius68_thetalt10_plt500 = [4.045666158618167,  4.086393662224346,  4.359141107602775,  4.666549994726691,  5.8569181911416015,
                             6.559716356124256,  8.686967529043072,  10.063482736354674, 13.053528344041274, 14.883496407943747,
                             18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406,  28.329169392203216,
                             29.468032123356345, 34.03271241527079,  35.03747443690781,  38.50748727211848,  39.41576583301171,
                             42.63622296033334,  45.41123601592071,  48.618139095742876, 48.11801717451056,  53.220539860213655,
                             58.87753380915155,  66.31550881539764,  72.94685877928593,  85.95506228335348,  89.20607201266672,
                             93.34370253818409,  96.59471226749734,  100.7323427930147,  103.98335252232795]
radius68_thetalt10_pgt500 = [4.081926458777424,  4.099431732299409,  4.262428482867968,  4.362017581473145,  4.831341579961153,
                             4.998346041276382,  6.2633736512415705, 6.588371889265881,  8.359969947444522,  9.015085558044309,
                             11.262722588206483, 12.250305471269183, 15.00547660437276,  16.187264014640103, 19.573764900578503,
                             20.68072032434797,  24.13797140783321,  25.62942209291236,  29.027596514735617, 30.215039667389316,
                             33.929540248019585, 36.12911729771914,  39.184563500620946, 42.02062468386282,  46.972125628650204,
                             47.78214816041894,  55.88428562462974,  59.15520134927332,  63.31816666637158,  66.58908239101515,
                             70.75204770811342,  74.022963432757,    78.18592874985525,  81.45684447449884]
radius68_theta10to20 = [4.0251896715647115, 4.071661598616328, 4.357690094817289,  4.760224640141712,  6.002480766325418,
                        6.667318981016246,  8.652513285172342, 9.72379373302137,   12.479492693251478, 14.058548828317289,
                        17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302,
                        31.768946746958296, 35.79247330197688, 37.27810357669942,  41.657281051476545, 42.628141392692626,
                        47.94208483539388,  49.9289473559796,  54.604030254423975, 53.958762417361655, 53.03339560920388,
                        57.026277390001425, 62.10810455035879, 66.10098633115634,  71.1828134915137,   75.17569527231124,
                        80.25752243266861,  84.25040421346615, 89.33223137382352,  93.32511315462106]
radius68_thetagt20 = [4.0754238481177705, 4.193693485630508,  5.14209420056253,   6.114996249971468,  7.7376807326481645,
                      8.551663213602291,  11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604,
                      25.04088272634407,  28.853696411302344, 34.72538105333071,  40.21218694947545,  46.07344239520299,
                      50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047,  74.82378572939959,
                      89.4528387422834,   93.18228303096758,  92.51751129204555,  98.80228884380018,  111.17537347472128,
                      120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185,
                      177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696]

## v14 RoC in 5 binnings (layer 26-34 from fitting)
radius68_thetalt10 = [  10.12233413, 9.921772, 11.38255086, 11.67991867, 13.14337347, 
                                13.17120624, 16.80994665, 17.83787244, 22.44684374, 23.74239886, 
                                28.60564083, 30.27889678, 34.86404888, 36.39009394, 41.29309474, 
                                43.34682279, 48.55982854, 50.80565589, 55.29496257, 57.92737879, 
                                60.64828824, 65.51760517, 68.26709803, 76.32877518, 84.61219467, 
                                103.3649691, 111.1692293, 119.2928089, 127.7357081, 136.4979268, 
                                145.579465, 154.9803228, 164.7005, 174.7399968 ]
radius68_theta10to15 = [ 10.82307758, 11.17850518, 16.2185281, 18.62488713, 22.63408229, 
                                24.71769042, 30.11217538, 32.69939046, 37.99753196, 40.81619543, 
                                45.89054775, 49.03066318, 54.00440948, 59.31733555, 63.40789682, 
                                64.77580021, 73.00113678, 73.25561396, 78.8914776, 86.73962133, 
                                97.05926327, 96.6932739, 111.6226151, 106.5960265, 109.477541, 
                                144.2545942, 153.7581461, 163.5921179, 173.7565094, 184.2513208, 
                                195.076552, 206.2322029, 217.7182737, 229.5347642 ]
radius68_theta15to20 = [ 12.79450901, 13.02698578, 21.27450933, 25.66008312, 31.78592103, 
                                35.99689874, 44.37101115, 48.82709363, 55.05972458, 59.68948687, 
                                65.39866214, 70.59280337, 76.06007787, 82.22695257, 87.50371819, 
                                90.60099831, 96.34848268, 101.4928478, 106.7157092, 105.0540604, 
                                110.0653355, 148.3428736, 133.1449443, 146.997265, 173.3954389, 
                                185.1307166, 196.1408667, 207.4772729, 219.1399351, 231.1288534, 
                                243.4440277, 256.085458, 269.0531444, 282.3470868 ]
radius68_theta20to30 = [ 14.16989595, 15.4488322, 28.31044668, 37.54285657, 48.57288885, 
                                57.04243339, 68.99836079, 75.33388728, 85.00572867, 91.52574074, 
                                102.5044698, 106.5315986, 116.2341378, 127.1121442, 133.8866375, 
                                144.5121759, 162.1726963, 160.2986579, 171.386638, 182.5653112, 
                                205.5853241, 196.3113071, 200.5907513, 228.7275694, 234.0298491, 
                                253.7990618, 263.6872702, 273.5754785, 283.4636869, 293.3518953, 
                                303.2401036, 313.128312, 323.0165203, 332.9047287 ]
radius68_theta30to60 = [ 22.50983127, 26.44537503, 58.24642887, 90.59076279, 130.0592014, 
                                157.4611392, 184.2187293, 202.6994588, 225.3488816, 243.3454167, 
                                269.2456428, 280.6119298, 303.8591523, 322.0522722, 335.1780181, 
                                350.3398234, 353.7763544, 373.9942362, 382.9453608, 401.9703438, 
                                441.6281859, 432.5241826, 455.2878243, 492.2888656, 502.6653722, 
                                519.9101788, 539.1604349, 558.410691, 577.6609471, 596.9112032, 
                                616.1614593, 635.4117154, 654.6619715, 673.9122276 ]

# For longitudinal segmentation
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
def distPtToLine(x, y1, y2):

    norm = np.linalg.norm(y1 - y2)
    if norm == 0: return np.sqrt(np.sum((x - y1)**2))

    return np.linalg.norm(np.cross(x - y1, y1 - y2))/norm

# Minimum distance between lines, each line defined by two points
def distTwoLines(x1, x2, y1, y2):

    cross = np.cross(x1 - x2, y1 - y2)
    norm = np.linalg.norm(cross)

    if norm == 0:

        xnorm = np.linalg.norm(x1 - x2)
        ynorm = np.linalg.norm(y1 - y2)

        if xnorm == 0 and ynorm == 0: return np.sqrt(np.sum((x1 - y1)**2))
        elif ynorm == 0: return np.linalg.norm(np.cross(x1 - x2, x1 - y1))/xnorm

        return np.linalg.norm(np.cross(x1 - y1, y1 - y2))/ynorm

    return abs(np.dot(cross, x1 - y1)/norm)

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
# Get e/g SP hit info
###########################

# Get electron hit with max p at any scoring plane
def maxPElectronSPHit(SPHits, sp_z):
    
    p_max = 0
    Hit_maxP = None
    for hit in SPHits:

        if abs(hit.getPosition()[2] - sp_z) > 0.5*sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1 or\
                hit.getPdgID() != 11:
            continue

        if mag(hit.getMomentum()) > p_max:
            Hit_maxP = hit
            p_max = mag(hit.getMomentum())
    
    return p_max, Hit_maxP

# Get electron target scoring plane hit
# Method 1: look at pz
def electronTargetSPHit(targetSPHits):
    
    E_threshold = 3000    # 3 GeV threshold, may need to modify
    targetSPHit = None
    pmax = 0
    
    # 1. Interact @ Target
    pmax, targetSPHit = maxPElectronSPHit(targetSPHits, sp_target_down_z)
    if pmax > E_threshold:
        # 2. Interact @ Trigger scin l1
        pmax, targetSPHit = maxPElectronSPHit(targetSPHits, sp_trigger_pad_down_l1_z)
        if pmax > E_threshold:
            # 3. Interact @ Trigger scin l2
            pmax, targetSPHit = maxPElectronSPHit(targetSPHits, sp_trigger_pad_down_l2_z)

    # Assume e- interacting @ Target   
    # for hit in targetSPHits:

    #     if abs(hit.getPosition()[2] - sp_target_down_z) > 0.5*sp_thickness or\
    #             hit.getMomentum()[2] <= 0 or\
    #             hit.getTrackID() != 1 or\
    #             hit.getPdgID() != 11:
    #         continue

    #     if mag(hit.getMomentum()) > pmax:
    #         targetSPHit = hit
    #         pmax = mag(targetSPHit.getMomentum())

    return targetSPHit

# Method 2: look at delta pz
def electronTargetSPHit_deltaPz(targetSPHits):
    
    dpz_threshold = 2400    # Delta pz threshold
    targetSPHit = None
    pmax = 0
    
    # 1. Interact @ Target
    pmax_target_down_z, targetSPHit_down = maxPElectronSPHit(targetSPHits, sp_target_down_z)
    pmax_ts_down_l1, targetSPHit_ts_down_l1 = maxPElectronSPHit(targetSPHits, sp_trigger_pad_down_l1_z)
    pmax_ts_down_l2, targetSPHit_ts_down_l2 = maxPElectronSPHit(targetSPHits, sp_trigger_pad_down_l2_z)
    targetSPHit = targetSPHit_down
    if pmax_target_down_z - pmax_ts_down_l1 > dpz_threshold:
        # 2. Interact @ Trigger scin l1
        # print('Found dpz = {} > 2.4 GeV @ Trigger scin l1'.format(pmax_target_down_z - pmax_ts_down_l1))
        targetSPHit = targetSPHit_ts_down_l1
    elif pmax_ts_down_l1 - pmax_ts_down_l2 > dpz_threshold:
        # 3. Interact @ Trigger scin l2
        # print('Found dpz = {} > 2.4 GeV @ Trigger scin l2'.format(pmax_ts_down_l1 - pmax_ts_down_l2))
        targetSPHit = targetSPHit_ts_down_l2

    return targetSPHit

# Get electron ecal scoring plane hit
def electronEcalSPHit(ecalSPHits):

    eSPHit = None
    pmax = 0
    for hit in ecalSPHits:

        if abs(hit.getPosition()[2] - sp_ecal_front_z) > 0.5*sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                hit.getPdgID() != 11:
                # hit.getTrackID() != 1 or\  # doesn't work for v14 sample
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

def gammaTargetInfo_true(eTargetSPHit, eTargetUpSPHit):
    # Use upstream target sp to decide incident e- momentum
    
    gTarget_pvec = np.array(eTargetUpSPHit.getMomentum()) - np.array(eTargetSPHit.getMomentum())

    return eTargetSPHit.getPosition(), gTarget_pvec

# Get photon ecal scoring plane hit
def gammaEcalSPHit(ecalSPHits):

    gSPHit = None
    pmax = 0
    for hit in ecalSPHits:

        if abs(hit.getPosition()[2] - sp_ecal_front_z) > 0.5*sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                not (hit.getPdgID() in [-22,22]):
            continue

        if mag(hit.getMomentum()) > pmax:
            gSPHit = hit
            pmax = mag(gSPHit.getMomentum())

    return gSPHit

# Get electron and photon ecal scoring plane hits
def elec_gamma_ecalSPHits(ecalSPHits):

    eSPHit = electronEcalSPHit(ecalSPHits)
    gSPHit = gammaEcalSPHit(ecalSPHits)

    return eSPHit, gSPHit
