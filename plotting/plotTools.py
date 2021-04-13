import ROOT as rt
from array import array

def computeEffVsCutGraph(hist, reversecutdir=False):
    # hist has to be TH1 derivative
  
    nbins = hist.GetNbinsX()
    xbins = array('d',[0.0]*nbins)

    for ibin in range(nbins):
      xbins[ibin] = hist.GetBinLowEdge(ibin+1)

    npasssig = array('d',[0.0]*nbins)
    effsig = array('d',[0.0]*nbins)

    sigtotal = hist.Integral(0, nbins+1)

    for ibin in range(nbins):
      if reversecutdir:
        npasssig[ibin] = hist.Integral(0, ibin)
      else:
        npasssig[ibin] = hist.Integral(ibin, nbins+1)

      effsig[ibin] = npasssig[ibin]/sigtotal

 
    gr = rt.TGraph(nbins, xbins, effsig)
    gr.SetTitle('')
    gr.GetXaxis().SetTitle(hist.GetXaxis().GetTitle())
    gr.GetYaxis().SetTitle('Efficiency')
  
    return gr

def getCutValueForEfficiency(hist, targeteff, reversecutdir=False):
    nbins = hist.GetNbinsX()
    xbins = array('d',[0.0]*(nbins+1))
    binw = array('d',[0.0]*nbins)

    for ibin in range(nbins):
      xbins[ibin] = hist.GetBinLowEdge(ibin+1)
      binw[ibin] = hist.GetBinWidth(ibin+1)

    xbins[nbins] = xbins[nbins-1] + binw[nbins-1];

    npass = array('d',[0.0]*nbins)
    eff = array('d',[0.0]*nbins)

    total = hist.Integral(0, nbins+1)

    effdiff = 1.0;
    nbin = -1;

    for ibin in range(nbins):
      if reversecutdir:
        npass[ibin] = hist.Integral(0, ibin)
      else:
        npass[ibin] = hist.Integral(ibin, nbins+1)
      eff[ibin] = npass[ibin]/total
      tmpdiff = abs(eff[ibin] - targeteff)
      if tmpdiff < effdiff:
        effdiff = tmpdiff
        nbin = ibin

    return (xbins[nbin],eff[nbin])

def getEfficiencyForCutValue(hist, cut, reversecutdir=False):
    nbins = hist.GetNbinsX()
    xbins = array('d',[0.0]*(nbins+1))
    binw = array('d',[0.0]*nbins)

    for ibin in range(nbins):
      xbins[ibin] = hist.GetBinLowEdge(ibin+1)
      binw[ibin] = hist.GetBinWidth(ibin+1)

    xbins[nbins] = xbins[nbins-1] + binw[nbins-1];

    npass = array('d',[0.0]*nbins)
    eff = array('d',[0.0]*nbins)

    total = hist.Integral(0, nbins+1)

    diff = 1.0;
    nbin = -1;

    for ibin in range(nbins):
      if reversecutdir:
        npass[ibin] = hist.Integral(0, ibin)
      else:
        npass[ibin] = hist.Integral(ibin, nbins+1)
      eff[ibin] = npass[ibin]/total
      tmpdiff = abs(xbins[ibin] - cut);
      if tmpdiff < diff:
        diff = tmpdiff;
        nbin = ibin;

    return (eff[nbin],xbins[nbin])

