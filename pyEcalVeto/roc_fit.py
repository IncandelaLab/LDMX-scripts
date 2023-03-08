import ROOT as r
import numpy as np
import uproot
import matplotlib.pyplot as plt

filename = "sig_test_unsorted.root"
t = uproot.open(filename)['EcalVeto']

nbin = 8
nlayer = 34

average = np.zeros((nbin,nlayer))

for i in range(nbin):
    for j in range(nlayer):
        roc = t['roc68_binning{}_layer{}'.format(i+1, j+1)].array(library="np")
        roc_nonzero = roc[np.nonzero(roc)]
        print("non zero roc", roc_nonzero)
        average[i,j] = np.average(roc_nonzero)

print("average: ",average)

for i in range(nbin):
    roc_average = average[i]
    layers = np.linspace(1, 34, num=34, endpoint=True)
    
    coeffs = np.polyfit(layers, roc_average, 2)
    print("Binning {}:".format(i+1))
    print('a2 is {:.3f}, a1 is {:.3f}, a0 is {:.3f}'.format(coeffs[0], coeffs[1], coeffs[2]))
    
    p = np.poly1d(coeffs)
    roc_fit = np.array([p(x) for x in layers])
    
    plt.figure(i)
    plt.plot(layers, roc_average,'k.' , label='data')
    plt.plot(layers, roc_fit , label='2d poly fit')
    plt.xlabel("Layer")
    plt.ylabel("RoC")
    plt.title("Binning {}".format(i+1))

    # plt.show()
    plt.savefig("RoC_binning{}".format(i+1))
