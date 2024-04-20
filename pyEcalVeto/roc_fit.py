import ROOT as r
import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model

def poly2(x, a0, a1, a2):
    return a2 * x * x + a1 * x + a0

def poly1(x, a0, a1):
    return a1 * x + a0

poly2model = Model(poly2, nan_policy='omit')
print("Poly2model: ")
print(poly2model.param_names, poly2model.independent_vars)

poly1model = Model(poly1, nan_policy='omit')
print("Poly1model: ")
print(poly1model.param_names, poly1model.independent_vars)

filename = "sig_test_unsorted.root"
t = uproot.open(filename)['EcalVeto']

nbin = 8
nlayer = 34

average = np.zeros((nbin,nlayer))
sigma = np.zeros((nbin,nlayer))

for i in range(nbin):
    for j in range(nlayer):
        roc = t['roc68_binning{}_layer{}'.format(i+1, j+1)].array(library="np")
        roc_nonzero = roc[np.nonzero(roc)]
        # print("non zero roc", roc_nonzero)
        average[i,j] = np.average(roc_nonzero)
        sigma[i,j] = np.std(roc_nonzero)

print("average: ",average)

for i in range(nbin):
    roc_average = average[i]
    roc_error = sigma[i]
    weight = np.reciprocal(roc_error)
    layers = np.linspace(1, 34, num=34, endpoint=True)
    
    # lmfit
    deg = 0
    if i < 3:
        # binning 1, 2, 3, use 1st order
        polymodel = poly2model
        deg = 2
    else:
        # binning >= 4, use 1st order
        polymodel = poly1model
        deg = 1
    
    params = polymodel.make_params(a0=8, a1=0.2, a2=0.1)
    result = polymodel.fit(roc_average, params, x=layers, weights=weight)
    layer_nonzero = layers[np.logical_not(np.isnan(roc_average))]
    print("================ lmfit {} ================".format(i))
    print(result.fit_report())
    
    # p = np.poly1d(coeffs)
    # roc_fit = np.array([p(x) for x in layers])
    
    plt.figure(i)
    plt.plot(layers, roc_average,'k.' , label='data')
    plt.errorbar(layers, roc_average, yerr=roc_error, fmt='ko', label='data')
    plt.plot(layer_nonzero, result.best_fit, 'r-', label='poly fit deg = {}'.format(deg))
    plt.xlabel("Layer")
    plt.ylabel("RoC")
    plt.title("Binning {}".format(i+1))

    # plt.show()
    plt.savefig("RoC_binning{}".format(i+1))
