from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
matplotlib.use('Agg')
#font_path = font_manager.findfont(font_manager.FontProperties(family='Helvetica'))
font_path = 'arial.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family']='arial'
plt.rcParams['xtick.labelsize' ]= 16
plt.rcParams['ytick.labelsize' ]= 16
plt.rcParams['axes.titlesize' ]= 16
plt.rcParams['axes.labelsize' ]= 16
def create_roc_curve(labels, scores, positive_label): #positive_label=1
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=positive_label)
    #roc_auc = auc(fpr, tpr)
    fig =plt.figure(figsize = (18,18))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])
s1=([0.1, 0.4, 0.35, 0.3])



def createAllRoc(labels, scores, positive_label,graphlabel,colors): #positive_label=1
    plt.rcParams["lines.linewidth"]=1.35
    fig =plt.figure(figsize = (8,6.6))
    plt.title('ROC Analysis')
    coun = 0
    for i in range(len(labels)):
        fpr, tpr, thresholds = metrics.roc_curve(labels[i], scores[i], pos_label=positive_label)
        print(thresholds)
        if coun < 4:
            plt.plot(fpr, tpr, 'b', label=graphlabel[i], color=colors[i])
        elif coun < 8:
            plt.plot(fpr, tpr, '--', label=graphlabel[i], color=colors[i], dashes=(3, 1.5), solid_capstyle='round')
        else:
            plt.plot(fpr, tpr, ':', label=graphlabel[i], color=colors[i])
   
        coun = coun+1
    plt.legend(loc='lower right')
    #plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,0.0010])
    plt.ylim([0.,1.])
    #plt.xlim([-0.1,1.2])
    #plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.savefig('ROC_8gev_1st_train_same_format_as_8GevPaper.png',dpi=400)
    plt.show()
    

#createAllRoc([y,y], [s1,scores], 1,['h1','h2'])

import ROOT
file1 = ROOT.TFile.Open('1.0_tree.root')
sig1 = file1.Get('EcalVeto')
file2 = ROOT.TFile.Open('0.1_tree.root')
sig01 = file2.Get('EcalVeto')
file3 = ROOT.TFile.Open('0.01_tree.root')
sig001 = file3.Get('EcalVeto')
file4 = ROOT.TFile.Open('0.001_tree.root')
sig0001 = file4.Get('EcalVeto')
file5 = ROOT.TFile.Open('bkg_tree.root')
bkg = file5.Get('EcalVeto')



sig1preds = []
for i in sig1:
    sig1preds +=[i.discValue_EcalVeto]
sig01preds = []
for i in sig01:
    sig01preds +=[i.discValue_EcalVeto]
sig001preds = []
for i in sig001:
    sig001preds +=[i.discValue_EcalVeto]
sig0001preds = []
for i in sig0001:
    sig0001preds +=[i.discValue_EcalVeto]
bkgpreds = []
for i in bkg:
    bkgpreds +=[i.discValue_EcalVeto]
bkgmirror = np.zeros(len(bkgpreds),dtype=int)
sig1mirror = np.zeros(len(sig1preds),dtype=int)+1
sig01mirror = np.zeros(len(sig01preds),dtype=int)+1
sig001mirror = np.zeros(len(sig001preds),dtype=int)+1
sig0001mirror = np.zeros(len(sig0001preds),dtype=int)+1
sig1roc = bkgpreds+sig1preds
sig01roc = bkgpreds+sig01preds
sig001roc = bkgpreds+sig001preds
sig0001roc = bkgpreds+sig0001preds

sig1label =np.concatenate((bkgmirror,sig1mirror))
sig01label =np.concatenate((bkgmirror,sig01mirror))
sig001label =np.concatenate((bkgmirror,sig001mirror))
sig0001label =np.concatenate((bkgmirror,sig0001mirror))


out = createAllRoc([sig1label,sig01label,sig001label,sig0001label], [sig1roc,sig01roc,sig001roc,sig0001roc], 1,['1 GeV Sig Seg','0.1 GeV Sig Seg','0.01 GeV Sig Seg','0.001 GeV Sig Seg'],['darkgreen','indigo','darkorange','lightskyblue'])
print(out)
