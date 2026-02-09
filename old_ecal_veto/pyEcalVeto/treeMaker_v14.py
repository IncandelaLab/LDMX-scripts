'''
treeMaker to:
    - calculate SegMIP & Gabrielle BDT variables
    - flatten LDMX_Event trees
    - with v14 geometry
'''
import os
import sys
import math
import ROOT as r
import numpy as np
sys.path.insert(1, '/home/xinyi_xu/ldmx-sw_4_23_24/LDMX-scripts/pyEcalVeto/mods')
from mods import ROOTmanager as manager
from mods import physTools, mipTracking
cellMap = np.loadtxt('/home/xinyi_xu/ldmx-sw_4_23_24/LDMX-scripts/pyEcalVeto/mods/cellmodule.txt')
r.gSystem.Load('libFramework.so')

# TreeModel to build here
branches_info = {
        # Base variables
        'nReadoutHits':              {'rtype': int,   'default': 0 },
        'summedDet':                 {'rtype': float, 'default': 0.},
        'summedDetTrig':             {'rtype': float, 'default': 0.},
        'summedTightIso':            {'rtype': float, 'default': 0.},
        'maxCellDep':                {'rtype': float, 'default': 0.},
        'showerRMS':                 {'rtype': float, 'default': 0.},
        'xStd':                      {'rtype': float, 'default': 0.},
        'yStd':                      {'rtype': float, 'default': 0.},
        'avgLayerHit':               {'rtype': float, 'default': 0.},
        'stdLayerHit':               {'rtype': float, 'default': 0.},
        'deepestLayerHit':           {'rtype': int,   'default': 0 },
        'ecalBackEnergy':            {'rtype': float, 'default': 0.},
        # MIP tracking variables
        'straight4':                 {'rtype': int,   'default': 0 },
        'firstNearPhLayer':          {'rtype': int,   'default': 33},
        'nNearPhHits':               {'rtype': int,   'default': 0 },
        'fullElectronTerritoryHits': {'rtype': int,   'default': 0 },
        'fullPhotonTerritoryHits':   {'rtype': int,   'default': 0 },
        'fullTerritoryRatio':        {'rtype': float, 'default': 1.},
        'electronTerritoryHits':     {'rtype': int,   'default': 0 },
        'photonTerritoryHits':       {'rtype': int,   'default': 0 },
        'TerritoryRatio':            {'rtype': float, 'default': 1.},
        'epSep':                     {'rtype': float, 'default': 0.},
        'epDot':                     {'rtype': float, 'default': 0.},
        'epAng':                     {'rtype': float, 'default': 0.}
        }

for i in range(1, physTools.nSegments + 1):

    # Longitudinal segment variables
    branches_info['energy_s{}'.format(i)]          = {'rtype': float, 'default': 0.}
    branches_info['nHits_s{}'.format(i)]           = {'rtype': int,   'default': 0 }
    branches_info['xMean_s{}'.format(i)]           = {'rtype': float, 'default': 0.}
    branches_info['yMean_s{}'.format(i)]           = {'rtype': float, 'default': 0.}
    branches_info['layerMean_s{}'.format(i)]       = {'rtype': float, 'default': 0.}
    branches_info['xStd_s{}'.format(i)]            = {'rtype': float, 'default': 0.}
    branches_info['yStd_s{}'.format(i)]            = {'rtype': float, 'default': 0.}
    branches_info['layerStd_s{}'.format(i)]        = {'rtype': float, 'default': 0.}

    for j in range(1, physTools.nRegions + 1):

        # Electron RoC variables
        branches_info['eContEnergy_x{}_s{}'.format(j,i)]    = {'rtype': float, 'default': 0.}
        branches_info['eContNHits_x{}_s{}'.format(j,i)]     = {'rtype': int,   'default': 0 }
        branches_info['eContXMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['eContYMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['eContLayerMean_x{}_s{}'.format(j,i)] = {'rtype': float, 'default': 0.}
        branches_info['eContXStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['eContYStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['eContLayerStd_x{}_s{}'.format(j,i)]  = {'rtype': float, 'default': 0.}

        # Photon RoC variables
        branches_info['gContEnergy_x{}_s{}'.format(j,i)]    = {'rtype': float, 'default': 0.}
        branches_info['gContNHits_x{}_s{}'.format(j,i)]     = {'rtype': int,   'default': 0 }
        branches_info['gContXMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['gContYMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['gContLayerMean_x{}_s{}'.format(j,i)] = {'rtype': float, 'default': 0.}
        branches_info['gContXStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['gContYStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['gContLayerStd_x{}_s{}'.format(j,i)]  = {'rtype': float, 'default': 0.}

        # Outside RoC variables
        branches_info['oContEnergy_x{}_s{}'.format(j,i)]    = {'rtype': float, 'default': 0.}
        branches_info['oContNHits_x{}_s{}'.format(j,i)]     = {'rtype': int,   'default': 0 }
        branches_info['oContXMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['oContYMean_x{}_s{}'.format(j,i)]     = {'rtype': float, 'default': 0.}
        branches_info['oContLayerMean_x{}_s{}'.format(j,i)] = {'rtype': float, 'default': 0.}
        branches_info['oContXStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['oContYStd_x{}_s{}'.format(j,i)]      = {'rtype': float, 'default': 0.}
        branches_info['oContLayerStd_x{}_s{}'.format(j,i)]  = {'rtype': float, 'default': 0.}

# Gabrielle variables
for j in range(1, physTools.nRegions + 1):
    # Electron RoC variables
    branches_info['eContEnergy_x{}'.format(j)]    = {'rtype': float, 'default': 0.}
    # branches_info['eContEnergy_x{}_0p5'.format(j)]    = {'rtype': float, 'default': 0.}
    # Photon RoC variables 
    branches_info['gContEnergy_x{}'.format(j)]    = {'rtype': float, 'default': 0.}
    # branches_info['gContEnergy_x{}_0p5'.format(j)]    = {'rtype': float, 'default': 0.}
    # Outside RoC variables
    branches_info['oContEnergy_x{}'.format(j)]    = {'rtype': float, 'default': 0.}
    branches_info['oContNHits_x{}'.format(j)]     = {'rtype': int,   'default': 0 }
    branches_info['oContXMean_x{}'.format(j)]     = {'rtype': float, 'default': 0.}
    branches_info['oContYMean_x{}'.format(j)]     = {'rtype': float, 'default': 0.}
    branches_info['oContXStd_x{}'.format(j)]      = {'rtype': float, 'default': 0.}
    branches_info['oContYStd_x{}'.format(j)]      = {'rtype': float, 'default': 0.}
    # branches_info['oContEnergy_x{}_0p5'.format(j)]    = {'rtype': float, 'default': 0.}
    # branches_info['oContNHits_x{}_0p5'.format(j)]     = {'rtype': int,   'default': 0 }
    # branches_info['oContXMean_x{}_0p5'.format(j)]     = {'rtype': float, 'default': 0.}
    # branches_info['oContYMean_x{}_0p5'.format(j)]     = {'rtype': float, 'default': 0.}
    # branches_info['oContXStd_x{}_0p5'.format(j)]      = {'rtype': float, 'default': 0.}
    # branches_info['oContYStd_x{}_0p5'.format(j)]      = {'rtype': float, 'default': 0.}

# Flatten tree variables
branches_flatten = {
        # SimParticles
        'SimParticles_size':            {'rtype': int,              'default': 0 },
        'SimParticles_trackID':         {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'SimParticles_energy':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_pdgID':           {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'SimParticles_x':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_y':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_z':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_time':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_mass':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_endX':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_endY':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_endZ':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_px':              {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_py':              {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_pz':              {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_endPX':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_endPY':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_endPZ':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'SimParticles_daughters':       {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'SimParticles_parents':         {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'SimParticles_processType':     {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'SimParticles_vertexVolume':    {'rtype': 'vector<string>', 'default': r.std.vector('string')(['']) },
        # EcalSimHits
        'EcalSimHits_size':                 {'rtype': int,              'default': 0 },
        'EcalSimHits_id':                   {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalSimHits_edep':                 {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalSimHits_x':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalSimHits_y':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalSimHits_z':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalSimHits_time':                 {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalSimHits_trackIDContribs':      {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'EcalSimHits_incidentIDContribs':   {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'EcalSimHits_pdgIDContribs':        {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'EcalSimHits_edepContribs':         {'rtype': 'vv<double>',     'default': r.std.vector('std::vector<double>')([[0.]]) },
        'EcalSimHits_timeContribs':         {'rtype': 'vv<double>',     'default': r.std.vector('std::vector<double>')([[0.]]) },
        'EcalSimHits_nContribs':            {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalSimHits_velocity':             {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        # TargetSimHits
        'TargetSimHits_size':               {'rtype': int,              'default': 0 },
        'TargetSimHits_id':                 {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'TargetSimHits_edep':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetSimHits_x':                  {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetSimHits_y':                  {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetSimHits_z':                  {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetSimHits_time':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetSimHits_trackIDContribs':    {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'TargetSimHits_incidentIDContribs': {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'TargetSimHits_pdgIDContribs':      {'rtype': 'vv<int>',        'default': r.std.vector('std::vector<int>')([[0]]) },
        'TargetSimHits_edepContribs':       {'rtype': 'vv<double>',     'default': r.std.vector('std::vector<double>')([[0.]]) },
        'TargetSimHits_timeContribs':       {'rtype': 'vv<double>',     'default': r.std.vector('std::vector<double>')([[0.]]) },
        'TargetSimHits_nContribs':          {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'TargetSimHits_velocity':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        # EcalScoringPlaneHits
        'EcalScoringPlaneHits_size':        {'rtype': int,              'default': 0 },
        'EcalScoringPlaneHits_id':          {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalScoringPlaneHits_layerID':     {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalScoringPlaneHits_moduleID':    {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalScoringPlaneHits_edep':        {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_energy':      {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_x':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_y':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_z':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_time':        {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_px':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_py':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_pz':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalScoringPlaneHits_trackID':     {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalScoringPlaneHits_pdgID':       {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        # TargetScoringPlaneHits
        'TargetScoringPlaneHits_size':        {'rtype': int,              'default': 0 },
        'TargetScoringPlaneHits_id':          {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'TargetScoringPlaneHits_layerID':     {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'TargetScoringPlaneHits_moduleID':    {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'TargetScoringPlaneHits_edep':        {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_energy':      {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_x':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_y':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_z':           {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_time':        {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_px':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_py':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_pz':          {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'TargetScoringPlaneHits_trackID':     {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'TargetScoringPlaneHits_pdgID':       {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        # EcalRecHits
        'EcalRecHits_size':                 {'rtype': int,              'default': 0 },
        'EcalRecHits_id':                   {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'EcalRecHits_amplitude':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalRecHits_energy':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalRecHits_time':                 {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalRecHits_x':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalRecHits_y':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalRecHits_z':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'EcalRecHits_isNoise':              {'rtype': 'vector<bool>',   'default': r.std.vector('bool')([0]) },
        # HCalRecHits
        'HCalRecHits_size':                 {'rtype': int,              'default': 0 },
        'HCalRecHits_id':                   {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'HCalRecHits_amplitude':            {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_energy':               {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_time':                 {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_x':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_y':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_z':                    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_isNoise':              {'rtype': 'vector<bool>',   'default': r.std.vector('bool')([0]) },
        'HCalRecHits_pe':                   {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_minpe':                {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'HCalRecHits_section':              {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'HCalRecHits_layer':                {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'HCalRecHits_strip':                {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'HCalRecHits_end':                  {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },
        'HCalRecHits_isADC':                {'rtype': 'vector<int>',    'default': r.std.vector('int')([0]) },             
        # ECalVeto
        'ECalVeto_passesVeto':              {'rtype': int,                'default': 0 },
        'ECalVeto_nReadoutHits':            {'rtype': int,                'default': 0 },
        'ECalVeto_deepestLayerHit':         {'rtype': int,                'default': 0 },
        'ECalVeto_summedDet':               {'rtype': float,              'default': 0. },
        'ECalVeto_summedTightIso':          {'rtype': float,              'default': 0. },
        'ECalVeto_maxCellDep':              {'rtype': float,              'default': 0. },
        'ECalVeto_showerRMS':               {'rtype': float,              'default': 0. },
        'ECalVeto_xStd':                    {'rtype': float,              'default': 0. },
        'ECalVeto_yStd':                    {'rtype': float,              'default': 0. },
        'ECalVeto_avgLayerHit':             {'rtype': float,              'default': 0. },
        'ECalVeto_stdLayerHit':             {'rtype': float,              'default': 0. },
        'ECalVeto_ecalBackEnergy':          {'rtype': float,              'default': 0. },
        'ECalVeto_nStraightTracks':         {'rtype': int,                'default': 0 },
        'ECalVeto_nLinregTracks':           {'rtype': int,                'default': 0 },
        'ECalVeto_firstNearPhLayer':        {'rtype': int,                'default': 0 },
        'ECalVeto_epAng':                   {'rtype': float,              'default': 0. },
        'ECalVeto_epSep':                   {'rtype': float,              'default': 0. },
        'ECalVeto_electronContainmentEnergy':   {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'ECalVeto_photonContainmentEnergy':     {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'ECalVeto_outsideContainmentEnergy':    {'rtype': 'vector<double>', 'default': r.std.vector('double')([0.]) },
        'ECalVeto_outsideContainmentNHits': {'rtype': 'vector<int>',      'default': r.std.vector('int')([0]) },
        'ECalVeto_outsideContainmentXStd':  {'rtype': 'vector<double>',   'default': r.std.vector('double')([0.]) },
        'ECalVeto_outsideContainmentYStd':  {'rtype': 'vector<double>',   'default': r.std.vector('double')([0.]) },
        'ECalVeto_discValue':               {'rtype': float,              'default': 0. },
        'ECalVeto_recoilPx':                {'rtype': float,              'default': 0. },
        'ECalVeto_recoilPy':                {'rtype': float,              'default': 0. },
        'ECalVeto_recoilPz':                {'rtype': float,              'default': 0. },
        'ECalVeto_recoilX':                 {'rtype': float,              'default': 0. },
        'ECalVeto_recoilY':                 {'rtype': float,              'default': 0. },
        # HCalVeto
        'HCalVeto_passesVeto':              {'rtype': int,                'default': 0 },
        'HCalVeto_maxPEHit_id':             {'rtype': int,                'default': 0 },
        'HCalVeto_maxPEHit_pe':             {'rtype': float,              'default': 0. },
        'HCalVeto_maxPEHit_layer':          {'rtype': int,                'default': 0 },
        'HCalVeto_maxPEHit_strip':          {'rtype': int,                'default': 0 }
}

# merge two dictionaries
branches_info.update(branches_flatten)

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    batch_mode = pdict['batch']
    separate = pdict['separate']
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    startEvent = pdict['startEvent']
    maxEvents = pdict['maxEvents']
    # Should maybe put in parsing eventually and make event_process *arg

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels,inlist):
        procs.append( manager.TreeProcess(event_process, group,
                                          ID=gl, batch=batch_mode, pfreq=100) )

    # Process jobs
    for proc in procs:

        # Move into appropriate scratch dir
        os.chdir(proc.tmp_dir)

        # tag
        tag = "sim"  # PN bkg
        # tag = "signal"  # signal
        
        # Branches needed
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_{}'.format(tag))
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_{}'.format(tag))
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_{}'.format(tag))
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_{}'.format(tag))
        proc.simParticles = proc.addBranch('SimParticle', 'SimParticles_{}'.format(tag))
        proc.ecalSimHits = proc.addBranch('SimCalorimeterHit', 'EcalSimHits_{}'.format(tag))
        proc.targetSimHits = proc.addBranch('SimCalorimeterHit', 'TargetSimHits_{}'.format(tag))
        proc.hcalRecHits = proc.addBranch('HcalHit', 'HcalRecHits_{}'.format(tag))
        proc.hcalVeto = proc.addBranch('HcalVetoResult', 'HcalVeto_{}'.format(tag))

        # Tree/Files(s) to make
        print('\nRunning %s'%(proc.ID))

        proc.separate = separate

        proc.tfMakers = {'unsorted': None}
        if proc.separate:
            proc.tfMakers = {
                'egin': None,
                'ein': None,
                'gin': None,
                'none': None
                }

        for tfMaker in proc.tfMakers:
            proc.tfMakers[tfMaker] = manager.TreeMaker(group_labels[procs.index(proc)]+\
                                        '.root',\
                                        "EcalVeto_flatten",\
                                        branches_info,\
                                        outlist[procs.index(proc)]
                                        )
                                        # Don't add this prefix
                                        # '_{}.root'.format(tfMaker),\

        # Gets executed at the end of run()
        proc.extrafs = [ proc.tfMakers[tfMaker].wq for tfMaker in proc.tfMakers ]

        # RUN
        proc.run(strEvent=startEvent, maxEvents=maxEvents)

    # Remove scratch directory if there is one
    if not batch_mode:     # Don't want to break other batch jobs when one finishes
        manager.rmScratch()

    print('\nDone!\n')


# Process an event
def event_process(self):

    # Initialize BDT input variables w/ defaults
    feats = next(iter(self.tfMakers.values())).resetFeats()

    # Assign pre-computed variables
    feats['nReadoutHits']       = self.ecalVeto.getNReadoutHits()
    feats['summedDet']          = self.ecalVeto.getSummedDet()
    feats['summedTightIso']     = self.ecalVeto.getSummedTightIso()
    feats['maxCellDep']         = self.ecalVeto.getMaxCellDep()
    feats['showerRMS']          = self.ecalVeto.getShowerRMS()
    feats['xStd']               = self.ecalVeto.getXStd()
    feats['yStd']               = self.ecalVeto.getYStd()
    feats['avgLayerHit']        = self.ecalVeto.getAvgLayerHit()
    feats['stdLayerHit']        = self.ecalVeto.getStdLayerHit()
    feats['deepestLayerHit']    = self.ecalVeto.getDeepestLayerHit() 
    feats['ecalBackEnergy']     = self.ecalVeto.getEcalBackEnergy()
    
    # Flatten tree variables
    ## SimParticles collection
    feats['SimParticles_size'] = self.simParticles.size()
    feats['SimParticles_trackID'].clear()
    feats['SimParticles_energy'].clear()
    feats['SimParticles_pdgID'].clear()
    feats['SimParticles_x'].clear()
    feats['SimParticles_y'].clear()
    feats['SimParticles_z'].clear()
    feats['SimParticles_time'].clear()
    feats['SimParticles_mass'].clear()
    feats['SimParticles_endX'].clear()
    feats['SimParticles_endY'].clear()
    feats['SimParticles_endZ'].clear()
    feats['SimParticles_px'].clear()
    feats['SimParticles_py'].clear()
    feats['SimParticles_pz'].clear()
    feats['SimParticles_endPX'].clear()
    feats['SimParticles_endPY'].clear()
    feats['SimParticles_endPZ'].clear()
    feats['SimParticles_daughters'].clear()
    feats['SimParticles_parents'].clear()
    feats['SimParticles_processType'].clear()
    feats['SimParticles_vertexVolume'].clear()
    for (tid, sp) in self.simParticles:
        feats['SimParticles_trackID'].push_back(tid)
        feats['SimParticles_energy'].push_back(sp.getEnergy()) # [MeV]
        feats['SimParticles_pdgID'].push_back(sp.getPdgID())
        feats['SimParticles_x'].push_back(sp.getVertex()[0]) # [mm]
        feats['SimParticles_y'].push_back(sp.getVertex()[1])
        feats['SimParticles_z'].push_back(sp.getVertex()[2])
        feats['SimParticles_time'].push_back(sp.getTime()) # [ns]
        feats['SimParticles_mass'].push_back(sp.getMass()) # [GeV]
        feats['SimParticles_endX'].push_back(sp.getEndPoint()[0]) # [mm]
        feats['SimParticles_endY'].push_back(sp.getEndPoint()[1])
        feats['SimParticles_endZ'].push_back(sp.getEndPoint()[2])
        feats['SimParticles_px'].push_back(sp.getMomentum()[0]) # [MeV]
        feats['SimParticles_py'].push_back(sp.getMomentum()[1])
        feats['SimParticles_pz'].push_back(sp.getMomentum()[2])
        feats['SimParticles_endPX'].push_back(sp.getEndPointMomentum()[0]) # [MeV]
        feats['SimParticles_endPY'].push_back(sp.getEndPointMomentum()[1])
        feats['SimParticles_endPZ'].push_back(sp.getEndPointMomentum()[2])
        feats['SimParticles_daughters'].push_back(sp.getDaughters())
        feats['SimParticles_parents'].push_back(sp.getParents())
        feats['SimParticles_processType'].push_back(sp.getProcessType())
        feats['SimParticles_vertexVolume'].push_back(sp.getVertexVolume())
    
    ## EcalSimHits & TargetSimHits collection
    trackIDContribs = r.std.vector('int')([])
    incidentIDContribs = r.std.vector('int')([])
    pdgIDContribs = r.std.vector('int')([])
    edepContribs = r.std.vector('double')([])
    timeContribs = r.std.vector("double")([])
    
    for bname in ['EcalSimHits', 'TargetSimHits']:
        if bname == 'EcalSimHits':
            hits = self.ecalSimHits
        else:
            hits = self.targetSimHits

        feats[bname+'_size'] = hits.size()
        feats[bname+'_id'].clear()
        feats[bname+'_edep'].clear()
        feats[bname+'_x'].clear()
        feats[bname+'_y'].clear()
        feats[bname+'_z'].clear()
        feats[bname+'_time'].clear()
        feats[bname+'_trackIDContribs'].clear()
        feats[bname+'_incidentIDContribs'].clear()
        feats[bname+'_pdgIDContribs'].clear()
        feats[bname+'_edepContribs'].clear()
        feats[bname+'_timeContribs'].clear()
        feats[bname+'_nContribs'].clear()
        feats[bname+'_velocity'].clear()
    
        for hit in hits:
            feats[bname+'_id'].push_back(hit.getID())
            feats[bname+'_edep'].push_back(hit.getEdep())
            feats[bname+'_x'].push_back(hit.getPosition()[0])
            feats[bname+'_y'].push_back(hit.getPosition()[1])
            feats[bname+'_z'].push_back(hit.getPosition()[2])
            feats[bname+'_time'].push_back(hit.getTime()) # [ns]
            trackIDContribs.clear()
            incidentIDContribs.clear()
            pdgIDContribs.clear()
            edepContribs.clear()
            timeContribs.clear()
            for i in range(hit.getNumberOfContribs()):
                trackIDContribs.push_back(hit.getContrib(i).trackID)
                incidentIDContribs.push_back(hit.getContrib(i).incidentID)
                pdgIDContribs.push_back(hit.getContrib(i).pdgCode)
                edepContribs.push_back(hit.getContrib(i).edep)
                timeContribs.push_back(hit.getContrib(i).time)
            feats[bname+'_trackIDContribs'].push_back(trackIDContribs)
            feats[bname+'_incidentIDContribs'].push_back(incidentIDContribs)
            feats[bname+'_pdgIDContribs'].push_back(pdgIDContribs)
            feats[bname+'_edepContribs'].push_back(edepContribs)
            feats[bname+'_timeContribs'].push_back(timeContribs)
            feats[bname+'_nContribs'].push_back(hit.getNumberOfContribs())
            feats[bname+'_velocity'].push_back(hit.getVelocity()) # [mm/ns]
        
    ## EcalScoringPlaneHits & TargetScoringPlaneHits collections
    for bname in ['EcalScoringPlaneHits', 'TargetScoringPlaneHits']:
        if bname == 'EcalScoringPlaneHits':
            hits = self.ecalSPHits
        else:
            hits = self.targetSPHits
        feats[bname+'_size'] = hits.size()
        feats[bname+'_id'].clear()
        feats[bname+'_layerID'].clear()
        feats[bname+'_moduleID'].clear()
        feats[bname+'_edep'].clear()
        feats[bname+'_energy'].clear()
        feats[bname+'_x'].clear()
        feats[bname+'_y'].clear()
        feats[bname+'_z'].clear()
        feats[bname+'_time'].clear()
        feats[bname+'_px'].clear()
        feats[bname+'_py'].clear()
        feats[bname+'_pz'].clear()
        feats[bname+'_trackID'].clear()
        feats[bname+'_pdgID'].clear()
        for hit in hits:
            feats[bname+'_id'].push_back(hit.getID())
            feats[bname+'_layerID'].push_back(hit.getLayerID())
            feats[bname+'_moduleID'].push_back(hit.getModuleID())
            feats[bname+'_edep'].push_back(hit.getEdep())
            feats[bname+'_energy'].push_back(hit.getEnergy())
            feats[bname+'_x'].push_back(hit.getPosition()[0])
            feats[bname+'_y'].push_back(hit.getPosition()[1])
            feats[bname+'_z'].push_back(hit.getPosition()[2])
            feats[bname+'_time'].push_back(hit.getTime())
            feats[bname+'_px'].push_back(hit.getMomentum()[0])
            feats[bname+'_py'].push_back(hit.getMomentum()[1])
            feats[bname+'_pz'].push_back(hit.getMomentum()[2])
            feats[bname+'_trackID'].push_back(hit.getTrackID())
            feats[bname+'_pdgID'].push_back(hit.getPdgID())
            
    ## EcalRecHits & HCalRecHits collections
    for bname in ['EcalRecHits', 'HCalRecHits']:
        if bname == 'EcalRecHits':
            hits = self.ecalRecHits
        else:
            hits = self.hcalRecHits
            feats[bname+'_pe'].clear()
            feats[bname+'_minpe'].clear()
            feats[bname+'_section'].clear()
            feats[bname+'_layer'].clear()
            feats[bname+'_strip'].clear()
            feats[bname+'_end'].clear()
            feats[bname+'_isADC'].clear()
        feats[bname+'_size'] = hits.size()
        feats[bname+'_id'].clear()
        feats[bname+'_amplitude'].clear()
        feats[bname+'_energy'].clear()
        feats[bname+'_time'].clear()
        feats[bname+'_x'].clear()
        feats[bname+'_y'].clear()
        feats[bname+'_z'].clear()
        feats[bname+'_isNoise'].clear()

        for hit in hits:
            feats[bname+'_id'].push_back(hit.getID())
            feats[bname+'_amplitude'].push_back(hit.getAmplitude())
            feats[bname+'_energy'].push_back(hit.getEnergy())
            feats[bname+'_x'].push_back(hit.getXPos())
            feats[bname+'_y'].push_back(hit.getYPos())
            feats[bname+'_z'].push_back(hit.getZPos())
            feats[bname+'_time'].push_back(hit.getTime())
            feats[bname+'_isNoise'].push_back(hit.isNoise())
            if bname == 'HCalRecHits':
                feats[bname+'_pe'].push_back(hit.getPE())
                feats[bname+'_minpe'].push_back(hit.getMinPE())
                feats[bname+'_section'].push_back(hit.getSection())
                feats[bname+'_layer'].push_back(hit.getLayer())
                feats[bname+'_strip'].push_back(hit.getStrip())
                feats[bname+'_end'].push_back(hit.getEnd())
                feats[bname+'_isADC'].push_back(hit.getIsADC())
        
    ## ECalVeto collection
    feats['ECalVeto_nReadoutHits']              = feats['nReadoutHits']
    feats['ECalVeto_summedDet']                 = feats['summedDet']
    feats['ECalVeto_summedTightIso']            = feats['summedTightIso']
    feats['ECalVeto_maxCellDep']                = feats['maxCellDep']
    feats['ECalVeto_showerRMS']                 = feats['showerRMS']
    feats['ECalVeto_xStd']                      = feats['xStd']
    feats['ECalVeto_yStd']                      = feats['yStd']
    feats['ECalVeto_avgLayerHit']               = feats['avgLayerHit']
    feats['ECalVeto_stdLayerHit']               = feats['stdLayerHit']
    feats['ECalVeto_deepestLayerHit']           = feats['deepestLayerHit']
    feats['ECalVeto_ecalBackEnergy']            = feats['ecalBackEnergy']
    feats['ECalVeto_passesVeto']                = self.ecalVeto.passesVeto()
    feats['ECalVeto_nStraightTracks']           = self.ecalVeto.getNStraightTracks()
    feats['ECalVeto_firstNearPhLayer']          = self.ecalVeto.getFirstNearPhLayer()
    feats['ECalVeto_epAng']                     = self.ecalVeto.getEPAng()
    feats['ECalVeto_epSep']                     = self.ecalVeto.getEPSep()
    feats['ECalVeto_electronContainmentEnergy'] = self.ecalVeto.getElectronContainmentEnergy()
    feats['ECalVeto_photonContainmentEnergy']   = self.ecalVeto.getPhotonContainmentEnergy()
    feats['ECalVeto_outsideContainmentEnergy']  = self.ecalVeto.getOutsideContainmentEnergy()
    feats['ECalVeto_outsideContainmentNHits']   = self.ecalVeto.getOutsideContainmentNHits()
    feats['ECalVeto_outsideContainmentXStd']    = self.ecalVeto.getOutsideContainmentXStd()
    feats['ECalVeto_outsideContainmentYStd']    = self.ecalVeto.getOutsideContainmentYStd()
    feats['ECalVeto_discValue']                 = self.ecalVeto.getDisc()
    feats['ECalVeto_recoilPx']                  = self.ecalVeto.getRecoilMomentum()[0]
    feats['ECalVeto_recoilPy']                  = self.ecalVeto.getRecoilMomentum()[1]
    feats['ECalVeto_recoilPz']                  = self.ecalVeto.getRecoilMomentum()[2]
    feats['ECalVeto_recoilX']                   = self.ecalVeto.getRecoilX()
    feats['ECalVeto_recoilY']                   = self.ecalVeto.getRecoilY()
    
    ## HCalVeto collection
    feats['HCalVeto_passesVeto']     = self.hcalVeto.passesVeto()
    feats['HCalVeto_maxPEHit_id']    = self.hcalVeto.getMaxPEHit().getID()
    feats['HCalVeto_maxPEHit_pe']    = self.hcalVeto.getMaxPEHit().getPE()
    feats['HCalVeto_maxPEHit_layer'] = self.hcalVeto.getMaxPEHit().getLayer()
    feats['HCalVeto_maxPEHit_strip'] = self.hcalVeto.getMaxPEHit().getStrip()
    
    ###################################
    # Determine event type
    ###################################

    # Get e position and momentum from EcalSP
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()

    # Photon Info from targetSP
    e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
    if e_targetHit != None:
        g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
    else:  # Should about never happen -> division by 0 in g_traj
        print('no e at targ!')
        g_targPos = g_targP = np.zeros(3)

    # Get electron and photon trajectories
    e_traj = g_traj = None

    if e_ecalHit != None:
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)

    if e_targetHit != None:
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    # Fiducial categories (filtered into different output trees)
    if self.separate:
        e_fid = g_fid = False

        if e_traj != None:
            for cell in cellMap:
                if physTools.dist( cell[1:], e_traj[0] ) <= physTools.cell_radius:
                    e_fid = True
                    break

        if g_traj != None:
            for cell in cellMap:
                if physTools.dist( cell[1:], g_traj[0] ) <= physTools.cell_radius:
                    g_fid = True
                    break

    ###################################
    # Compute extra BDT input variables
    ###################################

    # Find epSep and epDot, and prepare electron and photon trajectory vectors
    if e_traj != None and g_traj != None:

        # Create arrays marking start and end of each trajectory
        e_traj_ends = [np.array([e_traj[0][0], e_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([e_traj[-1][0], e_traj[-1][1], physTools.ecal_layerZs[-1] ])]
        g_traj_ends = [np.array([g_traj[0][0], g_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([g_traj[-1][0], g_traj[-1][1], physTools.ecal_layerZs[-1] ])]

        e_norm  = physTools.unit( e_traj_ends[1] - e_traj_ends[0] )
        g_norm  = physTools.unit( g_traj_ends[1] - g_traj_ends[0] )
        feats['epSep'] = physTools.dist( e_traj_ends[0], g_traj_ends[0] )
        feats['epDot'] = physTools.dot(e_norm,g_norm)
        # Add epAng
        feats['epAng'] = math.acos(physTools.dot(e_norm,g_norm)) * 180.0 / math.pi

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_ends   = [np.array([999 ,999 ,0   ]), np.array([999 ,999 ,999 ]) ]
        g_traj_ends   = [np.array([1000,1000,0   ]), np.array([1000,1000,1000]) ]

        feats['epSep'] = 10.0 + 1.0 # Don't cut on these in this case
        feats['epDot'] = 3.0 + 1.0 # ? This default value should be assigned to an angle
        feats['epAng'] = 3.0 + 1.0

    # Territory setup (consider missing case)
    gToe    = physTools.unit( e_traj_ends[0] - g_traj_ends[0] )
    origin  = g_traj_ends[0] + 0.5*8.7*gToe

    # Recoil electron momentum magnitude and angle with z-axis
    recoilPMag  = physTools.mag(  e_ecalP )                 if e_ecalHit != None else -1.0
    recoilTheta = physTools.angle(e_ecalP, units='degrees') if recoilPMag > 0    else -1.0

    # Set electron RoC binnings0
    ## v9 RoC
    # e_radii = physTools.radius68_thetalt10_plt500
    # if recoilTheta < 10 and recoilPMag >= 500: e_radii = physTools.radius68_thetalt10_pgt500
    # elif recoilTheta >= 10 and recoilTheta < 20: e_radii = physTools.radius68_theta10to20
    # elif recoilTheta >= 20: e_radii = physTools.radius68_thetagt20
    
    ## v14 RoC
    
    #e_radii = physTools.radius68_thetalt10
    #if recoilTheta >= 10 and recoilTheta < 15:
    #    e_radii = physTools.radius68_theta10to15
    #elif recoilTheta >= 15 and recoilTheta < 20:
    #    e_radii = physTools.radius68_theta15to20
    #elif recoilTheta >= 20 and recoilTheta < 30:
    #    e_radii = physTools.radius68_theta20to30
    #elif recoilTheta >= 30:
    #    e_radii = physTools.radius68_theta30to60

    ## half v14 RoC
    # e_radii = 0.5 * np.array(physTools.radius68_thetalt10)
    # if recoilTheta >= 10 and recoilTheta < 15:
    #     e_radii = 0.5 * np.array(physTools.radius68_theta10to15)
    # elif recoilTheta >= 15 and recoilTheta < 20:
    #     e_radii = 0.5 * np.array(physTools.radius68_theta15to20)
    # elif recoilTheta >= 20 and recoilTheta < 30:
    #     e_radii = 0.5 * np.array(physTools.radius68_theta20to30)
    # elif recoilTheta >= 30:
    #     e_radii = 0.5 * np.array(physTools.radius68_theta30to60)
    
    #v14 8 gev
    e_radii = physTools.radius68_thetalt10
    if recoilTheta >= 10 and recoilTheta < 15:
        e_radii = physTools.radius68_theta10to15
    elif recoilTheta >= 15 and recoilTheta < 25:
        e_radii = physTools.radius68_theta15to25
    elif recoilTheta >= 25 and recoilTheta < 30:
        e_radii = physTools.radius68_theta25to30
    elif recoilTheta >= 30 and recoilTheta < 40:
        e_radii = physTools.radius68_theta30to40
    elif recoilTheta >= 40 and recoilTheta < 50:
        e_radii = physTools.radius68_theta40to50

    # Always use default binning for photon RoC
    ## v9 RoC
    # g_radii = physTools.radius68_thetalt10_plt500
    ## v14 RoC
    #g_radii = physTools.radius68_thetalt10
    ## half v14 RoC
    # g_radii = 0.5 * np.array(physTools.radius68_thetalt10)

    #v14 8gev
    g_radii = physTools.radius68_thetalt10
    # Big data
    trackingHitList = []

    # Major ECal loop
    for hit in self.ecalRecHits:
        
        if hit.getEnergy() > 0:
            
            layer = physTools.ecal_layer(hit)
            xy_pair = ( hit.getXPos(), hit.getYPos() )

            # Territory selections
            hitPrime = physTools.pos(hit) - origin
            if np.dot(hitPrime, gToe) > 0: feats['fullElectronTerritoryHits'] += 1
            else: feats['fullPhotonTerritoryHits'] += 1

            # Distance to electron trajectory
            if e_traj != None:
                xy_e_traj = ( e_traj[layer][0], e_traj[layer][1] )
                distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
            else: distance_e_traj = -1.0

            # Distance to photon trajectory
            if g_traj != None:
                xy_g_traj = ( g_traj[layer][0], g_traj[layer][1] )
                distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
            else: distance_g_traj = -1.0
            
            # Trigger energy
            if layer <= 20:
                feats['summedDetTrig'] += hit.getEnergy()

            # Decide which longitudinal segment the hit is in and add to sums
            for i in range(1, physTools.nSegments + 1):

                if (physTools.segLayers[i - 1] <= layer)\
                  and (layer <= physTools.segLayers[i] - 1):
                    feats['energy_s{}'.format(i)] += hit.getEnergy()
                    feats['nHits_s{}'.format(i)] += 1
                    feats['xMean_s{}'.format(i)] += xy_pair[0]*hit.getEnergy()
                    feats['yMean_s{}'.format(i)] += xy_pair[1]*hit.getEnergy()
                    feats['layerMean_s{}'.format(i)] += layer*hit.getEnergy()

                    # Decide which containment region the hit is in and add to sums
                    for j in range(1, physTools.nRegions + 1):

                        if ((j - 1)*e_radii[layer] <= distance_e_traj)\
                          and (distance_e_traj < j*e_radii[layer]):
                            feats['eContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['eContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['eContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['eContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['eContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        if ((j - 1)*g_radii[layer] <= distance_g_traj)\
                          and (distance_g_traj < j*g_radii[layer]):
                            feats['gContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['gContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['gContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['gContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['gContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        if (distance_e_traj > j*e_radii[layer])\
                          and (distance_g_traj > j*g_radii[layer]):
                            feats['oContEnergy_x{}_s{}'.format(j,i)] += hit.getEnergy()
                            feats['oContNHits_x{}_s{}'.format(j,i)] += 1
                            feats['oContXMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[0]*hit.getEnergy()
                            feats['oContYMean_x{}_s{}'.format(j,i)] +=\
                                                                xy_pair[1]*hit.getEnergy()
                            feats['oContLayerMean_x{}_s{}'.format(j,i)] +=\
                                                                layer*hit.getEnergy()

                        # double granularity
                        # if ((j - 1)*e_radii[layer]*0.5 <= distance_e_traj)\
                        #   and (distance_e_traj < j*e_radii[layer]*0.5):
                        #     feats['eContEnergy_x{}_0p5'.format(j)] += hit.getEnergy()
                              
                        # if ((j - 1)*g_radii[layer]*0.5 <= distance_g_traj)\
                        #   and (distance_g_traj < j*g_radii[layer]*0.5):
                        #     feats['gContEnergy_x{}_0p5'.format(j)] += hit.getEnergy()
                        
                        # if (distance_e_traj > j*e_radii[layer]*0.5)\
                        #   and (distance_g_traj > j*g_radii[layer]*0.5):
                        #     feats['oContEnergy_x{}_0p5'.format(j)] += hit.getEnergy()
                        #     feats['oContNHits_x{}_0p5'.format(j)] += 1
                        #     feats['oContXMean_x{}_0p5'.format(j)] +=\
                        #                                         xy_pair[0]*hit.getEnergy()
                        #     feats['oContYMean_x{}_0p5'.format(j)] +=\
                        #                                         xy_pair[1]*hit.getEnergy()
                        
            # Build MIP tracking hit list; (outside electron region or electron missing)
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                trackingHitList.append(hit) 

    # Sum over segments to get total energy, oContNHits per region
    for j in range(1, physTools.nRegions + 1):
        for i in range(1, physTools.nSegments + 1):
            feats['eContEnergy_x{}'.format(j)] += feats['eContEnergy_x{}_s{}'.format(j,i)]
            feats['gContEnergy_x{}'.format(j)] += feats['gContEnergy_x{}_s{}'.format(j,i)]
            feats['oContEnergy_x{}'.format(j)] += feats['oContEnergy_x{}_s{}'.format(j,i)]
            feats['oContNHits_x{}'.format(j)] += feats['oContNHits_x{}_s{}'.format(j,i)]
            feats['oContXMean_x{}'.format(j)] += feats['oContXMean_x{}_s{}'.format(j,i)]
            feats['oContYMean_x{}'.format(j)] += feats['oContYMean_x{}_s{}'.format(j,i)]

    # If possible, quotient out the total energy from the means
    for i in range(1, physTools.nSegments + 1):

        if feats['energy_s{}'.format(i)] > 0:
            feats['xMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]
            feats['yMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]
            feats['layerMean_s{}'.format(i)] /= feats['energy_s{}'.format(i)]

        for j in range(1, physTools.nRegions + 1):

            if feats['eContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['eContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]
                feats['eContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]
                feats['eContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['eContEnergy_x{}_s{}'.format(j,i)]

            if feats['gContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['gContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]
                feats['gContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]
                feats['gContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['gContEnergy_x{}_s{}'.format(j,i)]

            if feats['oContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['oContXMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]
                feats['oContYMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]
                feats['oContLayerMean_x{}_s{}'.format(j,i)] /=\
                                                    feats['oContEnergy_x{}_s{}'.format(j,i)]

    for j in range(1, physTools.nRegions + 1):
        if feats['oContEnergy_x{}'.format(j)] > 0:
            feats['oContXMean_x{}'.format(j)] /= feats['oContEnergy_x{}'.format(j)]
            feats['oContYMean_x{}'.format(j)] /= feats['oContEnergy_x{}'.format(j)]
        # if feats['oContEnergy_x{}_0p5'.format(j)] > 0:
        #     feats['oContXMean_x{}_0p5'.format(j)] /= feats['oContEnergy_x{}_0p5'.format(j)]
        #     feats['oContYMean_x{}_0p5'.format(j)] /= feats['oContEnergy_x{}_0p5'.format(j)]
    
    # Loop over hits again to calculate the standard deviations
    for hit in self.ecalRecHits:

        layer = physTools.ecal_layer(hit)
        xy_pair = (hit.getXPos(), hit.getYPos())

        # Distance to electron trajectory
        if e_traj != None:
            xy_e_traj = (e_traj[layer][0], e_traj[layer][1])
            distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
        else:
            distance_e_traj = -1.0

        # Distance to photon trajectory
        if g_traj != None:
            xy_g_traj = (g_traj[layer][0], g_traj[layer][1])
            distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
        else:
            distance_g_traj = -1.0

        # Decide which longitudinal segment the hit is in and add to sums
        for i in range(1, physTools.nSegments + 1):

            if (physTools.segLayers[i - 1] <= layer) and\
                    (layer <= physTools.segLayers[i] - 1):
                feats['xStd_s{}'.format(i)] += ((xy_pair[0] -\
                        feats['xMean_s{}'.format(i)])**2)*hit.getEnergy()
                feats['yStd_s{}'.format(i)] += ((xy_pair[1] -\
                        feats['yMean_s{}'.format(i)])**2)*hit.getEnergy()
                feats['layerStd_s{}'.format(i)] += ((layer -\
                        feats['layerMean_s{}'.format(i)])**2)*hit.getEnergy()

                # Decide which containment region the hit is in and add to sums
                for j in range(1, physTools.nRegions + 1):

                    if ((j - 1)*e_radii[layer] <= distance_e_traj)\
                      and (distance_e_traj < j*e_radii[layer]):
                        feats['eContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['eContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['eContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['eContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['eContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['eContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

                    if ((j - 1)*g_radii[layer] <= distance_g_traj)\
                      and (distance_g_traj < j*g_radii[layer]):
                        feats['gContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['gContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['gContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['gContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['gContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['gContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()

                    if (distance_e_traj > j*e_radii[layer])\
                      and (distance_g_traj > j*g_radii[layer]):
                        feats['oContXStd_x{}_s{}'.format(j,i)] += ((xy_pair[0] -\
                                feats['oContXMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['oContYStd_x{}_s{}'.format(j,i)] += ((xy_pair[1] -\
                                feats['oContYMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        feats['oContLayerStd_x{}_s{}'.format(j,i)] += ((layer -\
                            feats['oContLayerMean_x{}_s{}'.format(j,i)])**2)*hit.getEnergy()
                        # Gabrielle
                        feats['oContXStd_x{}'.format(j)] += ((xy_pair[0] -\
                                feats['oContXMean_x{}'.format(j)])**2)*hit.getEnergy()
                        feats['oContYStd_x{}'.format(j)] += ((xy_pair[1] -\
                                feats['oContYMean_x{}'.format(j)])**2)*hit.getEnergy()
                    
                    # double granularity
                    # if (distance_e_traj > j*e_radii[layer]*0.5)\
                    #   and (distance_g_traj > j*g_radii[layer]*0.5):
                    #     feats['oContXStd_x{}_0p5'.format(j)] += ((xy_pair[0] -\
                    #             feats['oContXMean_x{}_0p5'.format(j)])**2)*hit.getEnergy()
                    #     feats['oContYStd_x{}_0p5'.format(j)] += ((xy_pair[1] -\
                    #             feats['oContYMean_x{}_0p5'.format(j)])**2)*hit.getEnergy()

    # Sum over segments to get total oContXStd, oContYStd per region
    # for j in range(1, physTools.nRegions + 1):
    #     for i in range(1, physTools.nSegments + 1):
    #         feats['oContXStd_x{}'.format(j)] += feats['oContXStd_x{}_s{}'.format(j,i)]
    #         feats['oContYStd_x{}'.format(j)] += feats['oContYStd_x{}_s{}'.format(j,i)]

    # Quotient out the total energies from the standard deviations if possible and take root
    for i in range(1, physTools.nSegments + 1):

        if feats['energy_s{}'.format(i)] > 0:
            feats['xStd_s{}'.format(i)] = math.sqrt(feats['xStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])
            feats['yStd_s{}'.format(i)] = math.sqrt(feats['yStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])
            feats['layerStd_s{}'.format(i)] = math.sqrt(feats['layerStd_s{}'.format(i)]/\
                    feats['energy_s{}'.format(i)])

        for j in range(1, physTools.nRegions + 1):

            if feats['eContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['eContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContXStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])
                feats['eContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContYStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])
                feats['eContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['eContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['eContEnergy_x{}_s{}'.format(j,i)])

            if feats['gContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['gContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContXStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])
                feats['gContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContYStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])
                feats['gContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['gContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['gContEnergy_x{}_s{}'.format(j,i)])

            if feats['oContEnergy_x{}_s{}'.format(j,i)] > 0:
                feats['oContXStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContXStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])
                feats['oContYStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContYStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])
                feats['oContLayerStd_x{}_s{}'.format(j,i)] =\
                        math.sqrt(feats['oContLayerStd_x{}_s{}'.format(j,i)]/\
                        feats['oContEnergy_x{}_s{}'.format(j,i)])

    for j in range(1, physTools.nRegions + 1):
        if feats['oContEnergy_x{}'.format(j)] > 0:
            feats['oContXStd_x{}'.format(j)] =\
                    math.sqrt(feats['oContXStd_x{}'.format(j)]/\
                    feats['oContEnergy_x{}'.format(j)])
            feats['oContYStd_x{}'.format(j)] =\
                    math.sqrt(feats['oContYStd_x{}'.format(j)]/\
                    feats['oContEnergy_x{}'.format(j)])
            # feats['oContXStd_x{}_0p5'.format(j)] =\
            #         math.sqrt(feats['oContXStd_x{}_0p5'.format(j)]/\
            #         feats['oContEnergy_x{}_0p5'.format(j)])
            # feats['oContYStd_x{}_0p5'.format(j)] =\
            #         math.sqrt(feats['oContYStd_x{}_0p5'.format(j)]/\
            #         feats['oContEnergy_x{}_0p5'.format(j)])

    # Find the first layer of the ECal where a hit near the projected photon trajectory
    # AND the total number of hits around the photon trajectory
    if g_traj != None: # If no photon trajectory, leave this at the default

        # First currently unusued; pending further study; performance drop from  v9 and v12
        #print(trackingHitList, g_traj)
        feats['firstNearPhLayer'], feats['nNearPhHits'] = mipTracking.nearPhotonInfo(
                                                            trackingHitList, g_traj )
    else: feats['nNearPhHits'] = feats['nReadoutHits']


    # Territories limited to trackingHitList
    if e_traj != None:
        for hit in trackingHitList:
            hitPrime = physTools.pos(hit) - origin
            if np.dot(hitPrime, gToe) > 0: feats['electronTerritoryHits'] += 1
            else: feats['photonTerritoryHits'] += 1
    else:
        feats['photonTerritoryHits'] = feats['nReadoutHits']
        feats['TerritoryRatio'] = 10
        feats['fullTerritoryRatio'] = 10
    if feats['electronTerritoryHits'] != 0:
        feats['TerritoryRatio'] = feats['photonTerritoryHits']/feats['electronTerritoryHits']
    if feats['fullElectronTerritoryHits'] != 0:
        feats['fullTerritoryRatio'] = feats['fullPhotonTerritoryHits']/\
                                            feats['fullElectronTerritoryHits']


    # Find MIP tracks
    feats['straight4'], trackingHitList = mipTracking.findStraightTracks(
                                trackingHitList, e_traj_ends, g_traj_ends,
                                mst = 4, returnHitList = True)

    # Fill the tree (according to fiducial category) with values for this event
    if not self.separate:
        self.tfMakers['unsorted'].fillEvent(feats)
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)

if __name__ == "__main__":
    main()
