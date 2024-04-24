from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import font_manager
import ROOT as r


import ROOT
file1 = ROOT.TFile.Open('/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/bdt_test_0/0.001_PT_evalout.root')
file2 = ROOT.TFile.Open('/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/bdt_test_0/0.01_PT_evalout.root')
file3 = ROOT.TFile.Open('/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/bdt_test_0/0.1_PT_evalout.root')
file4 = ROOT.TFile.Open('/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/bdt_test_0/1_PT_evalout.root')
file5 = ROOT.TFile.Open('/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/pyEcalVeto/bdt_test_0/pn_PT_evalout.root')
tree_0001 = file1.Get('EcalVeto')
tree_001 = file2.Get('EcalVeto')
tree_01 = file3.Get('EcalVeto')
tree_1 = file4.Get('EcalVeto')
tree_pn = file5.Get('EcalVeto')
c=0
last_layer= 0.17669999

#0.001
event_p_0001=[]
event_p_disc_0001=[]
for event in tree_0001:
    if event.summedDet<3000:
        max_p_0001 = 0 
        max_p_disc_0001 = 0
        p_transverse_0001 = 0 
        p_transverse_disc_0001 = 0  
        for n, val in enumerate(event.TargetScoringPlaneHits_z):
            #print(val)
            if val >= last_layer and event.TargetScoringPlaneHits_pdgID[n] == 11 and event.TargetScoringPlaneHits_pz[n]>0:
                p_squre_0001 = event.TargetScoringPlaneHits_px[n]**2 + event.TargetScoringPlaneHits_py[n]**2 + event.TargetScoringPlaneHits_pz[n]**2
                p_0001 = np.sqrt(p_squre_0001)
                #print(val)
                if p_0001 > max_p_0001:
                    max_p_0001 = p_0001
                    p_transverse_0001 = np.sqrt(max_p_0001**2-event.TargetScoringPlaneHits_pz[n]**2)
                if event.discValue_EcalVeto > 0.991946:
                    c=c+1
                    p_disc_0001 = p_0001
                    if p_disc_0001 > max_p_disc_0001:
                        max_p_disc_0001 = p_disc_0001
                        p_transverse_disc_0001 = np.sqrt(max_p_disc_0001**2-event.TargetScoringPlaneHits_pz[n]**2)    

        event_p_0001.append(p_transverse_0001)
        event_p_disc_0001.append(p_transverse_disc_0001)

print(len(event_p_0001))
#print(len(event_p_disc_0001))
print("c",c)

#0.01
event_p_001=[]
event_p_disc_001=[]
for event in tree_001:
    if event.summedDet<3000:
        max_p_001 = 0 
        max_p_disc_001 = 0
        p_transverse_001 = 0 
        p_transverse_disc_001 = 0  
        for n, val in enumerate(event.TargetScoringPlaneHits_z):
            #print(val)
            if val >= last_layer and event.TargetScoringPlaneHits_pdgID[n] == 11 and event.TargetScoringPlaneHits_pz[n]>0:
                p_squre_001 = event.TargetScoringPlaneHits_px[n]**2 + event.TargetScoringPlaneHits_py[n]**2 + event.TargetScoringPlaneHits_pz[n]**2
                p_001 = np.sqrt(p_squre_001)
                #print(val)
                if p_001 > max_p_001:
                    max_p_001 = p_001
                    p_transverse_001 = np.sqrt(max_p_001**2-event.TargetScoringPlaneHits_pz[n]**2)
                if event.discValue_EcalVeto > 0.991946:
                    c=c+1
                    p_disc_001 = p_001
                    if p_disc_001 > max_p_disc_001:
                        max_p_disc_001 = p_disc_001
                        p_transverse_disc_001 = np.sqrt(max_p_disc_001**2-event.TargetScoringPlaneHits_pz[n]**2)    

        event_p_001.append(p_transverse_001)
        event_p_disc_001.append(p_transverse_disc_001)

print(len(event_p_001))
#print(len(event_p_disc_001))
print("c",c)


#0.1
event_p_01=[]
event_p_disc_01=[]
for event in tree_01:
    if event.summedDet<3000:
        max_p_01 = 0 
        max_p_disc_01 = 0
        p_transverse_01 = 0 
        p_transverse_disc_01 = 0  
        for n, val in enumerate(event.TargetScoringPlaneHits_z):
            #print(val)
            if val >= last_layer and event.TargetScoringPlaneHits_pdgID[n] == 11 and event.TargetScoringPlaneHits_pz[n]>0:
                p_squre_01 = event.TargetScoringPlaneHits_px[n]**2 + event.TargetScoringPlaneHits_py[n]**2 + event.TargetScoringPlaneHits_pz[n]**2
                p_01 = np.sqrt(p_squre_01)
                #print(val)
                if p_01 > max_p_01:
                    max_p_01 = p_01
                    p_transverse_01 = np.sqrt(max_p_01**2-event.TargetScoringPlaneHits_pz[n]**2)
                if event.discValue_EcalVeto > 0.991946:
                    c=c+1
                    p_disc_01 = p_01
                    if p_disc_01 > max_p_disc_01:
                        max_p_disc_01 = p_disc_01
                        p_transverse_disc_01 = np.sqrt(max_p_disc_01**2-event.TargetScoringPlaneHits_pz[n]**2)    

        event_p_01.append(p_transverse_01)
        event_p_disc_01.append(p_transverse_disc_01)



#1.0
event_p_1=[]
event_p_disc_1=[]
for event in tree_1:
    if event.summedDet<3000:
        max_p_1 = 0 
        max_p_disc_1 = 0
        p_transverse_1 = 0 
        p_transverse_disc_1 = 0  
        for n, val in enumerate(event.TargetScoringPlaneHits_z):
            #print(val)
            if val >= last_layer and event.TargetScoringPlaneHits_pdgID[n] == 11 and event.TargetScoringPlaneHits_pz[n]>0:
                p_squre_1 = event.TargetScoringPlaneHits_px[n]**2 + event.TargetScoringPlaneHits_py[n]**2 + event.TargetScoringPlaneHits_pz[n]**2
                p_1 = np.sqrt(p_squre_1)
                #print(val)
                if p_1 > max_p_1:
                    max_p_1 = p_1
                    p_transverse_1 = np.sqrt(max_p_1**2-event.TargetScoringPlaneHits_pz[n]**2)
                if event.discValue_EcalVeto > 0.991946:
                    c=c+1
                    p_disc_1 = p_1
                    if p_disc_1 > max_p_disc_1:
                        max_p_disc_1 = p_disc_1
                        p_transverse_disc_1 = np.sqrt(max_p_disc_1**2-event.TargetScoringPlaneHits_pz[n]**2)    

        event_p_1.append(p_transverse_1)
        event_p_disc_1.append(p_transverse_disc_1)

#pn
event_p_pn=[]
event_p_disc_pn=[]
for event in tree_pn:
    if event.summedDet<3000:
        max_p_pn = 0 
        max_p_disc_pn = 0
        p_transverse_pn = 0 
        p_transverse_disc_pn = 0  
        for n, val in enumerate(event.TargetScoringPlaneHits_z):
            #print(val)
            if val >= last_layer and event.TargetScoringPlaneHits_pdgID[n] == 11 and event.TargetScoringPlaneHits_pz[n]>0:
                p_squre_pn = event.TargetScoringPlaneHits_px[n]**2 + event.TargetScoringPlaneHits_py[n]**2 + event.TargetScoringPlaneHits_pz[n]**2
                p_pn = np.sqrt(p_squre_pn)
                #print(val)
                if p_pn > max_p_pn:
                    max_p_pn = p_pn
                    p_transverse_pn = np.sqrt(max_p_pn**2-event.TargetScoringPlaneHits_pz[n]**2)
                if event.discValue_EcalVeto > 0.991946:
                    c=c+1
                    p_disc_pn = p_pn
                    if p_disc_pn > max_p_disc_pn:
                        max_p_disc_pn = p_disc_pn
                        p_transverse_disc_pn = np.sqrt(max_p_disc_pn**2-event.TargetScoringPlaneHits_pz[n]**2)    

        event_p_pn.append(p_transverse_pn)
        event_p_disc_pn.append(p_transverse_disc_pn)




ROOT.gROOT.SetBatch(True)


bmin_p = 0.0
bmax_p = np.max(event_p_1)
bin_width= 10

n_bins= int((bmax_p - bmin_p) / bin_width)
print(n_bins)
print(bmax_p)
print(bmin_p)
print(bin_width)

#0.001
# Create a ROOT histogram for event_p
hist_p_0001 = ROOT.TH1F("hist_p_0001", "Number of Events vs. Transverse Momentum", n_bins, bmin_p, bmax_p)
for value in event_p_0001:
    hist_p_0001.Fill(value)  # Put values above x=40 in the last bin
    #else:
hist_p_0001.Scale(1.0 / hist_p_0001.Integral("width"))      


#hist_p_disc_0001 = ROOT.TH1F("hist_p_disc_0001", "", n_bins, bmin_p, bmax_p)
#for value in event_p_disc_0001:
#    if value <= bmax_p_0001:
#        hist_p_disc_0001.Fill(value) # Put values above x=40 in the last bin
    #else:
        


#0.01
hist_p_001 = ROOT.TH1F("hist_p_001", "Number of Events vs. Transverse Momentum", n_bins, bmin_p, bmax_p)
for value in event_p_001:
    hist_p_001.Fill(value)  # Put values above x=40 in the last bin
hist_p_001.Scale(1.0 / hist_p_001.Integral("width"))  
       


#hist_p_disc_001 = ROOT.TH1F("hist_p_disc_001", "Ratio", n_bins_001, bmin_p, bmax_p)
##\for value in event_p_disc_001:
#    if value <= bmax_p_001:
#         hist_p_disc_001.Fill(value)

#0.1
hist_p_01 = ROOT.TH1F("hist_p_01", "Number of Events vs. Transverse Momentum", n_bins, bmin_p, bmax_p)
for value in event_p_01:
    hist_p_01.Fill(value)
hist_p_01.Scale(1.0 / hist_p_01.Integral("width"))   


#hist_p_disc_01 = ROOT.TH1F("hist_p_disc_01", "Ratio", n_bins_01, bmin_p, bmax_p)
#for value in event_p_disc_01:
#    if value <= bmax_p_01:
#        hist_p_disc_01.Fill(value)



#1.0
hist_p_1 = ROOT.TH1F("hist_p_1", "Number of Events vs. Transverse Momentum", n_bins, bmin_p, bmax_p)
for value in event_p_1:
    hist_p_1.Fill(value)
hist_p_1.Scale(1.0 / hist_p_1.Integral("width"))   


#hist_p_disc_1 = ROOT.TH1F("hist_p_disc_1", "Ratio", n_bins_1, bmin_p, bmax_p)
#for value in event_p_disc_1:
#    if value <= bmax_p_1:
#        hist_p_disc_1.Fill(value)



#pn
hist_p_pn = ROOT.TH1F("hist_p_pn", "Number of Events vs. Transverse Momentum", n_bins, bmin_p, bmax_p)
for value in event_p_pn:
    hist_p_pn.Fill(value)
hist_p_pn.Scale(1.0 / hist_p_pn.Integral("width"))   

# Create a ROOT histogram for event_p_disc
#hist_p_disc_pn = ROOT.TH1F("hist_p_disc_pn", "Ratio", n_bins_pn, bmin_p, bmax_p)
#for value in event_p_disc_pn:
#    if value <= bmax_p_pn:
#        hist_p_disc_pn.Fill(value)


canvas = ROOT.TCanvas("canvas", "Histogram", 800, 600)
#hist_p_0001.SetTitle("pT distribution")
hist_p_0001.SetTitleSize(0.08)

canvas.cd()
ROOT.gPad.SetPad(0, 0, 1, 1) 
ROOT.gPad.SetTopMargin(0.1)  
ROOT.gPad.SetBottomMargin(0.1)
canvas.SetLeftMargin(0.13)
canvas.SetRightMargin(0.18)
canvas.SetLogy() 


hist_p_0001.SetLineColor(ROOT.kAzure+2)  
hist_p_0001.Draw("HIST")


  
hist_p_001.SetLineColor(ROOT.kOrange-3)  
hist_p_001.Draw("HIST SAME")


  
hist_p_01.SetLineColor(ROOT.kViolet+2)  
hist_p_01.Draw("HIST SAME")


  
hist_p_1.SetLineColor(ROOT.kGreen+3)  
hist_p_1.Draw("HIST SAME")


 
hist_p_pn.SetLineColor(ROOT.kRed+2)  
hist_p_pn.Draw("HIST SAME")

hist_p_0001.GetXaxis().SetTitle("Recoil pT (MeV)")
#hist_p_0001.GetYaxis().SetTitle("# of events")
hist_p_0001.GetXaxis().SetTitleSize(0.05)  
hist_p_0001.GetYaxis().SetTitleSize(0.05)  
hist_p_0001.GetYaxis().SetTitleOffset(0.8) 
hist_p_0001.GetXaxis().SetLabelSize(0.03)  
hist_p_0001.GetYaxis().SetLabelSize(0.03) 
#hist_p_0001.GetXaxis().SetRangeUser(0, 600)
hist_p_0001.GetYaxis().SetRangeUser(1e-5, 1e0)


legend = ROOT.TLegend(0.67,0.6,0.77,0.8)  
legend.SetNColumns(1)  
legend.SetTextSize(0.04)  
legend.SetBorderSize(0)
legend.SetFillStyle(0)

legend.AddEntry(hist_p_0001, "0.001 GeV", "L")  
legend.AddEntry(hist_p_001, "0.01 GeV", "L")  
legend.AddEntry(hist_p_01, "0.1 GeV", "L")  
legend.AddEntry(hist_p_1, "1.0 GeV", "L")  
legend.AddEntry(hist_p_pn, "pn", "L")  
legend.Draw()

hist_p_0001.SetLineWidth(2)
hist_p_001.SetLineWidth(2)
hist_p_01.SetLineWidth(2)
hist_p_1.SetLineWidth(2)
hist_p_pn.SetLineWidth(2)

hist_p_0001.SetStats(0)

# Update the canvas
canvas.Update()

# Save the histogram to a file
canvas.SaveAs("PT_distribution_8GeV_new.png")

