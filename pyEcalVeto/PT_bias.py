from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import font_manager
import ROOT as r


import ROOT
file1 = ROOT.TFile.Open('bdt_test_0/0.1_PT_evalout.root')
tree = file1.Get('EcalVeto')
c=0
last_layer= 0.17669999


event_p=[]
event_p_disc=[]
for event in tree:
    if event.summedDet<3000:
        max_p = 0 
        max_p_disc = 0
        p_transverse = -1 
        p_transverse_disc = -1  
        for n, val in enumerate(event.TargetScoringPlaneHits_z):
            if val >= last_layer and event.TargetScoringPlaneHits_pdgID[n] == 11 and event.TargetScoringPlaneHits_pz[n]>=0:
                p_squre = event.TargetScoringPlaneHits_px[n]**2 + event.TargetScoringPlaneHits_py[n]**2 + event.TargetScoringPlaneHits_pz[n]**2
                p = np.sqrt(p_squre)
                if p > max_p:
                    max_p = p
                    p_transverse = np.sqrt(max_p**2-event.TargetScoringPlaneHits_pz[n]**2)

        if event.discValue_EcalVeto > 0.991946 :#and event.HCalVeto_passesVeto == 1:
            p_transverse_disc=p_transverse
            c=c+1

        event_p.append(p_transverse)
        event_p_disc.append(p_transverse_disc)
#print("c",c)
#print(len(event_p))
#print(len(event_p_disc))

a=0
for event in tree:
    #print(event.discValue_EcalVeto)
    if event.discValue_EcalVeto > 0.991946: #and event.HCalVeto_passesVeto == 1:
        a=a+1
print(a)
        

ROOT.gROOT.SetBatch(True)
min_p = 0.0
max_p=500
bin_width = 5

print(max_p)
n_bins = int((max_p - min_p) / bin_width)

# histogram for event_p
hist_p = ROOT.TH1F("hist_p", "Number of Events vs. Transverse Momentum", n_bins, min_p, max_p)
for value in event_p:
    if value >= max_p:
        hist_p.Fill(max_p - 0.001)  
    else:
        hist_p.Fill(value)

hist_p.SetLineColor(ROOT.kRed)  
hist_p.SetLineWidth(2)  

# histogram for event_p_disc
hist_p_disc = ROOT.TH1F("hist_p_disc", "Ratio", n_bins, min_p, max_p)
for value in event_p_disc:
    if value >= max_p:
        hist_p_disc.Fill(max_p - 0.001)  
    else:
        hist_p_disc.Fill(value)

hist_p_disc.SetLineColor(ROOT.kOrange+1)  
hist_p_disc.SetLineWidth(2)  


canvas = ROOT.TCanvas("canvas", "Histogram", 800, 800)
canvas.Divide(1, 2)  # two pads

# Upper original distribution
canvas.cd(1)
ROOT.gPad.SetPad(0, 0.3, 1, 1)  
ROOT.gPad.SetBottomMargin(0)  
ROOT.gPad.SetLogy()  
hist_p.Draw("HIST")
hist_p_disc.Draw("HIST SAME")

# axis labels legend for the upper
hist_p.GetXaxis().SetTitle("Recoil pT")
hist_p.GetYaxis().SetTitle("Number of Events")
hist_p.GetXaxis().SetTitleSize(0.04)  
hist_p.GetYaxis().SetTitleSize(0.04)
hist_p.SetStats(0)
hist_p_disc.SetStats(0)

legend = ROOT.TLegend(0.15, 0.7, 0.4, 0.9)
legend.SetTextSize(0.04)  
legend.AddEntry(hist_p, "All event")
legend.AddEntry(hist_p_disc, "segmipX cut (1 bkg left)")
legend.SetX1(0.15)  
legend.SetX2(0.725)
legend.SetY1(0.82)  
legend.SetY2(0.90)
legend.Draw()

# Lower for the ratio plot
canvas.cd(2)
ROOT.gPad.SetPad(0, 0, 1, 0.3)  
ROOT.gPad.SetTopMargin(0) 
ROOT.gPad.SetBottomMargin(0.25) 
hist_ratio = hist_p_disc.Clone()
hist_ratio.Divide(hist_p)  

hist_ratio.Draw("HIST")

# Set the axis labels for the ratio plot
hist_ratio.GetXaxis().SetTitle("Recoil pT")
hist_ratio.GetYaxis().SetTitle("Ratio")
hist_ratio.GetXaxis().SetTitleSize(0.08)  
hist_ratio.GetYaxis().SetTitleSize(0.06)  

hist_ratio.GetXaxis().SetLabelSize(0.08)  
hist_ratio.GetYaxis().SetLabelSize(0.08) 
hist_ratio.GetYaxis().SetRangeUser(0, 1.2)
hist_ratio.SetStats(0)


canvas.Update()
canvas.SaveAs("0.1_pT_bias_8GeV_1pn_left_disc.png")







