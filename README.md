# LDMX-scripts

Please get started with going to https://github.com/IncandelaLab/LDMX-scripts/wiki

The tutorials can be found at https://github.com/IncandelaLab/LDMX-scripts/blob/master/TutorialFiles/README.md

Specific Notes Regarding WAB:
- Everything related to the WAB is under pyEcalVeto
- There are several .lhe files located under the WABSample directory. They are the most relevant WAB samples with selected form factors. We only consider form factors 2 and 4 because others are either too easy to veto or are too unlikely to happen. 
- The quickTreeMaker.py quickly extracts the recoil electron 3-momentum for you.
- The wab_sample_analyzer.py and the wab_sample_analyzer_v12.py are the main two python files that do the WAB analysis. It figures out how deep the recoil electron penetrates through the recoil tracker and what is the correct photon and electron kinematic information. This information is directly obtained from the Target Scoring Plane, so they are true values and are not accessible in the real experiment. Looking at them simply helps us figure out what is actually going on. Do NOT USE the branch "gTheta" and "egTheta", unless you want to know how bad the usual way to find photon kinematic information is. gTheta is the angle of photon inferred from the electron kinematic information (basically just [0,0,4000MeV] - recoil electron 3-momentum, so this is to some extent an experimentally accessible value). This is how treeMaker.py usually computes photon angle, but it will give you a horrible estimate of the photon kinematics in the WAB cases because the target or other phenomena will take energy away from the initial 4GeV. So please use "gThetaNew", "gPMagnitude", "gPx","gx"... to look for photon information. The exact meaning of each of the branches is commented on the file wab_sample_analyzer.py
