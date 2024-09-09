import uproot
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# load files and calculate rec energy without HIPs
def process_directory_p(file_path, tree_name='LDMX_Events'):
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        summed_det = tree['EcalVeto_sim/summedDet_'].array(library='np')
        ecalsimH_pdgCodeContribs = tree["EcalSimHits_sim/EcalSimHits_sim.pdgCodeContribs_"].array(library='np')
        ecalsimH_x = tree["EcalSimHits_sim/EcalSimHits_sim.x_"].array(library='np')
        ecalsimH_y = tree["EcalSimHits_sim/EcalSimHits_sim.y_"].array(library='np')
        ecalsimH_z = tree["EcalSimHits_sim/EcalSimHits_sim.z_"].array(library='np')
            

        ecalrecH_amp = tree["EcalRecHits_sim.amplitude_"].array(library='np')
        ecalrecH_z = tree["EcalRecHits_sim.zpos_"].array(library='np')
        ecalrecH_x = tree["EcalRecHits_sim.xpos_"].array(library='np')
        ecalrecH_y = tree["EcalRecHits_sim.ypos_"].array(library='np')
        ecalrecH_energy = tree["EcalRecHits_sim.energy_"].array(library='np')
        ecalrecH_id = tree["EcalRecHits_sim/EcalRecHits_sim.id_"].array(library='np')
        layer_info = (ecalrecH_id >> 17) & 0x3f
            

        ecalrecH_amp_no_n_p = [[0 for _ in sublist] for sublist in ecalrecH_amp]
        layer_weights_for_each_hit=[[0 for _ in sublist] for sublist in ecalrecH_amp]
        ecalrecH_energy_no_p_n=[[0 for _ in sublist] for sublist in ecalrecH_amp]
        re_summed_Det = []   
            
        ecalrecH_energy_re=[[1 for _ in sublist] for sublist in ecalrecH_energy]


        layer_weights = np.array([2.312, 4.312, 6.522, 7.490, 8.595, 10.253, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 10.915, 14.783, 18.539, 18.539, 18.539, 18.539, 18.539, 18.539, 18.539, 18.539, 18.539, 9.938])
        mip_si_energy = 0.130
        secondOrderEnergyCorrection = 4000. / 3940.5 


        for i in range(len(summed_det)):  
            for j in range(len(ecalsimH_pdgCodeContribs[i])):
                if any((pdg_code == 2212 or pdg_code > 1000000) for pdg_code in ecalsimH_pdgCodeContribs[i][j]):    
                    for k in range(len(ecalrecH_x[i])):
                        if (ecalrecH_x[i][k] == ecalsimH_x[i][j]) and (ecalrecH_y[i][k] == ecalsimH_y[i][j]) and (ecalrecH_z[i][k] == ecalsimH_z[i][j]):
                            ecalrecH_amp_no_n_p[i][k]=-999.
                            
                else:
                    for k in range(len(ecalrecH_x[i])):
                        if (ecalrecH_x[i][k] == ecalsimH_x[i][j]) and (ecalrecH_y[i][k] == ecalsimH_y[i][j]) and (ecalrecH_z[i][k] == ecalsimH_z[i][j]):
                            ecalrecH_amp_no_n_p[i][k]=ecalrecH_amp[i][k]

                    
        for i in range(len(summed_det)):
            for j in range(len(layer_info[i])):
                k=layer_info[i][j]
                layer_weights_for_each_hit[i][j]=layer_weights[k]
                    

            
        for i in range(len(ecalrecH_amp_no_n_p)):
            for j in range(len(ecalrecH_amp_no_n_p[i])):
                if ecalrecH_amp_no_n_p[i][j]==-999:
                    ecalrecH_energy_no_p_n[i][j]=0
                else:
                    ecalrecH_energy_no_p_n[i][j]=((1 + layer_weights_for_each_hit[i][j] / mip_si_energy)*ecalrecH_amp_no_n_p[i][j])

        for i in range (len(summed_det)):
            re_summed_Det.append(np.sum(ecalrecH_energy_no_p_n[i])* secondOrderEnergyCorrection)
            
#re_summed_Det for reconstructed energy of events without HIPs, summed_det for all events
    return np.array(re_summed_Det), np.array(summed_det)

pn_025_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/025_0degree_z0.root'
pn_050_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/050_0degree_z0.root'
pn_075_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/075_0degree_z0.root'
pn_125_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/125_0degree_z0.root'
pn_150_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/150_0degree_z0.root'
pn_175_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/175_0degree_z0.root'
pn_100_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/100_0degree_z0.root'
pn_225_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/2_25GeV_0mm_Vertex.root'
pn_250_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/250_0degree_z0.root'
pn_275_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/2_75GeV_0mm_Vertex.root'
pn_200_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/200_0degree_z0.root'
pn_325_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/3_25GeV_0mm_Vertex.root'
pn_350_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/3_5GeV_0mm_Vertex.root'
pn_375_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/3_75GeV_0mm_Vertex.root'
pn_300_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/3GeV_0mm_Vertex.root'
pn_400_dir='/home/xinyi_xu/ldmx-sw/ldmx-sw/LDMX-scripts/electron_gun/outputs/4GeV_0mm_Vertex.root'

re_summed_Det_025,summed_det_025=process_directory_p(pn_025_dir)
re_summed_Det_050,summed_det_050=process_directory_p(pn_050_dir)
re_summed_Det_075,summed_det_075=process_directory_p(pn_075_dir)
re_summed_Det_100,summed_det_100=process_directory_p(pn_100_dir)
re_summed_Det_125,summed_det_125=process_directory_p(pn_125_dir)
re_summed_Det_150,summed_det_150=process_directory_p(pn_150_dir)
re_summed_Det_175,summed_det_175=process_directory_p(pn_175_dir)
re_summed_Det_200,summed_det_200=process_directory_p(pn_200_dir)
re_summed_Det_225,summed_det_225=process_directory_p(pn_225_dir)
re_summed_Det_250,summed_det_250=process_directory_p(pn_250_dir)
re_summed_Det_275,summed_det_275=process_directory_p(pn_275_dir)
re_summed_Det_300,summed_det_300=process_directory_p(pn_300_dir)
re_summed_Det_325,summed_det_325=process_directory_p(pn_325_dir)
re_summed_Det_350,summed_det_350=process_directory_p(pn_350_dir)
re_summed_Det_375,summed_det_375=process_directory_p(pn_375_dir)
re_summed_Det_400,summed_det_400=process_directory_p(pn_400_dir)
print("data processed !")



# Calculate the mean and standard deviation of different energy in the unit of GeV
def mean_std(E):
    GeV_E=np.array(E)/1000
    mean_E = sum(GeV_E) / len(GeV_E)
    std_dev_E = (sum((x - mean_E) ** 2 for x in GeV_E) / len(GeV_E)) ** 0.5
    return mean_E, std_dev_E
# calculate data points that need to be fited
def x_y_calculation(mean_E, std_dev_E):
    x=mean_E
    y=std_dev_E/mean_E
    x_error= std_dev_E/50000**0.5   
    return x,y,x_error
re_025_mean,re_025_std=mean_std(re_summed_Det_025)
re_050_mean,re_050_std=mean_std(re_summed_Det_050)
re_075_mean,re_075_std=mean_std(re_summed_Det_075)
re_100_mean,re_100_std=mean_std(re_summed_Det_100)
re_125_mean,re_125_std=mean_std(re_summed_Det_125)
re_150_mean,re_150_std=mean_std(re_summed_Det_150)
re_175_mean,re_175_std=mean_std(re_summed_Det_175)
re_200_mean,re_200_std=mean_std(re_summed_Det_200)
re_225_mean,re_225_std=mean_std(re_summed_Det_225)
re_250_mean,re_250_std=mean_std(re_summed_Det_250)
re_275_mean,re_275_std=mean_std(re_summed_Det_275)
re_300_mean,re_300_std=mean_std(re_summed_Det_300)
re_325_mean,re_325_std=mean_std(re_summed_Det_325)
re_350_mean,re_350_std=mean_std(re_summed_Det_350)
re_375_mean,re_375_std=mean_std(re_summed_Det_375)
re_400_mean,re_400_std=mean_std(re_summed_Det_400)

x_025,y_025,x_025_error=x_y_calculation(re_025_mean,re_025_std)
x_050,y_050,x_050_error=x_y_calculation(re_050_mean,re_050_std)
x_075,y_075,x_075_error=x_y_calculation(re_075_mean,re_075_std)
x_100,y_100,x_100_error=x_y_calculation(re_100_mean,re_100_std)
x_125,y_125,x_125_error=x_y_calculation(re_125_mean,re_125_std)
x_150,y_150,x_150_error=x_y_calculation(re_150_mean,re_150_std)
x_175,y_175,x_175_error=x_y_calculation(re_175_mean,re_175_std)
x_200,y_200,x_200_error=x_y_calculation(re_200_mean,re_200_std)
x_225,y_225,x_225_error=x_y_calculation(re_225_mean,re_225_std)
x_250,y_250,x_250_error=x_y_calculation(re_250_mean,re_250_std)
x_275,y_275,x_275_error=x_y_calculation(re_275_mean,re_275_std)
x_300,y_300,x_300_error=x_y_calculation(re_300_mean,re_300_std)
x_325,y_325,x_325_error=x_y_calculation(re_325_mean,re_325_std)
x_350,y_350,x_350_error=x_y_calculation(re_350_mean,re_350_std)
x_375,y_375,x_375_error=x_y_calculation(re_375_mean,re_375_std)
x_400,y_400,x_400_error=x_y_calculation(re_400_mean,re_400_std)


x_data=[x_025,x_050,x_075,x_100,x_125,x_150,x_175,x_200,x_225,x_250,x_275,x_300,x_325,x_350,x_375,x_400]
y_data=[y_025,y_050,y_075,y_100,y_125,y_150,y_175,y_200,y_225,y_250,y_275,y_300,y_325,y_350,y_375,y_400]
x_error=[x_025_error,x_050_error,x_075_error,x_100_error,x_125_error,x_150_error,x_175_error,x_200_error,x_225_error,x_250_error,x_275_error,x_300_error,x_325_error,x_350_error,x_375_error,x_400_error]
y_error= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
xmax=np.max(x_data)
xmin=np.min(x_data)


import ROOT
from array import array

graph = ROOT.TGraphErrors(len(x_data), array('d', x_data), array('d', y_data), array('d', x_error), array('d', y_error))

model = "sqrt([0]**2/x + [1]**2 + [2]**2/x**2)"
func = ROOT.TF1("modelFunc", model, xmin, xmax)
func.SetParameters(0.22715, 0.02826, 1.04342e-7)  # initial guesses for parameters s, c, n

# Fit 
fit_result = graph.Fit(func, "SCHI2") 

params = fit_result.Get().GetParams()
errors = fit_result.Get().GetErrors()
chi2 = fit_result.Get().Chi2()
print("Fitted Parameters and Uncertainties:")
for i in range(func.GetNpar()):
    print(f"Parameter {i}: {params[i]} +/- {errors[i]}")
print(f"Chi-squared: {chi2}")

canvas = ROOT.TCanvas("canvas", "Chi-Square Fit", 1200, 800)
graph.SetMarkerStyle(20) 
graph.SetMarkerSize(1)  
graph.Draw("AP")

func.SetLineColor(2) 
func.SetLineWidth(1)  
func.Draw("same") 

leg = ROOT.TLegend(0.65, 0.75, 0.9, 0.85) 
leg.AddEntry(graph, "Events w/o HIP", "lep")
leg.Draw()

graph.SetTitle("#frac{#sigma}{#LT E_{meas}#GT} #propto #frac{s}{#sqrt{#LT E_{meas}#GT}} #oplus C #oplus #frac{n}{#LT E_{meas}#GT};#LT E_{meas}#GT (GeV);#frac{#sigma}{#LT E_{meas}#GT}")
canvas.SetTopMargin(0.14)

canvas.Draw() 
canvas.SaveAs("energy_resolution_no_p_n.png")


