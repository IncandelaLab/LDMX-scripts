import os
import sys
import ast
from configparser import ConfigParser
from collections import OrderedDict
from ROOT import gROOT,gStyle,gSystem,TFile,TTree,TH1,TH1D,TH2D,TGraph,TCanvas,TLegend,TLine
import styleTools as st
import plotTools as pt

def main():
    # Default conf file, if running with a different one just provide it as an argument
    configFile = 'plots.conf'
    args = sys.argv[1:]
    if len(args) >= 1:
        configFile = args[0]
    if os.path.exists(configFile):
        print ('running with config file', configFile)
    else:
        print ('you are trying to use a config file (' + configFile + ') that does not exist!')
        sys.exit(1)

    # Parse the conf file
    cfg = ConfigParser(dict_type=OrderedDict)
    cfg.optionxform = str

    cfg.read(configFile)

    # Set up plotting
    inputdir = cfg.get('setup','inputdir')
    outputdir = cfg.get('setup','outputdir')
    procs = cfg.get('setup','processes').replace(' ', '').split(',')
    treename = cfg.get('setup','treename')
    treename2d = cfg.get('setup','treename2d')
    comparetype = cfg.get('setup','comparetype')
    if not comparetype in ['processes', 'sels']:
        print ('comparetype must be either "processes" or "sels"!')
        sys.exit(1)
    plotnames = cfg.get('plotting','plotnames').replace(' ', '').split(',')
    plotnames2d = cfg.get('plotting','plotnames2d').replace(' ', '').split(',')
    effplotnames = cfg.get('plotting','effplotnames').replace(' ', '').split(',')
    reversecutvars = cfg.get('plotting','reversecutvars').replace(' ', '').split(',')
    logvars = cfg.get('plotting','logvars').replace(' ', '').split(',')
    expr = {k:v for k,v in cfg['expressions'].items()}
    sel = {k:v for k,v in cfg['sels'].items()}
    sel2d = {k:v for k,v in cfg['sels2d'].items()}
    proclabels = {k:v for k,v in cfg['proclabels'].items()}
    plotlabels = {k:v for k,v in cfg['plotlabels'].items()}
    binning = {k:ast.literal_eval(v) for k,v in cfg['binning'].items()}
    colors = {k:st.colors[v] for k,v in cfg['colors'].items()}

    # ROOT setup stuff
    gROOT.SetBatch(True)

    st.SetTDRStyle()
    st.SetupColors()

    gStyle.SetLabelSize(0.03,"XYZ")

    gSystem.mkdir(outputdir, True)

    print ('Making plots!')

    files = {proc:TFile.Open(inputdir+'/'+proc+'_tree.root') for proc in procs}

    # Loop over 1D variables to be plotted
    for n in plotnames:
        if n == '' or not n in binning:
            continue

        print ('plotting',n)
        
        (outerloopitems, innerloopitems) = (procs, sel) if comparetype == 'sels' else (sel, procs)

        # Loop over outer loop items
        for x in outerloopitems:

            hists = []
            ymax = -1.

            infile = None
            tree = None
            selexp = ''

            if comparetype == 'processes':
                selexp = sel[x]
                print ('with selection', selexp)

            # Loop over inner loop items
            for y in innerloopitems:
                if comparetype == 'sels':
                    infile = files[x]
                    selexp = sel[y]
                    print ('with selection',selexp)
                else:
                    infile = files[y]

                tree = infile.FindObjectAny(treename)

                hist = TH1D('_'.join(['h',n,x,y]),'',binning[n][0],binning[n][1],binning[n][2])
                hist.SetLineColor(colors[y])

                # Check if variable name corresponds to an expression
                if n in expr:
                    tree.Draw(expr[n]+'_'.join(['>>h',n,x,y]),selexp,'histnorm')
                else:
                    tree.Draw(n+'_'.join(['>>h',n,x,y]),selexp,'histnorm')

                # Histogram setup
                st.addOverFlow(hist)

                hist.GetXaxis().SetTitle(plotlabels[n])
                hist.GetXaxis().SetTitleSize(0.04)
                hist.GetYaxis().SetTitle("A.U.")
                hist.GetYaxis().SetTitleSize(0.05)
                hist.SetLineWidth(3)
                hist.SetLineColor(colors[y])
                if(hist.GetMaximum() > ymax):
                    ymax = hist.GetMaximum()
                tempmax = 0.0
                maxbin = 0
                for ibin in range(1,hist.GetNbinsX()+1):
                    binc = hist.GetBinContent(ibin)
                    if binc > tempmax:
                        tempmax = binc
                        maxbin = hist.GetBinLowEdge(ibin)

                # Add this histogram to the list
                hists.append(hist)

            # Setup canvas
            c = st.MakeCanvas("c","",600,600)
            leg = TLegend(0.65,0.7,0.95,0.9)
            st.SetLegendStyle(leg)
            leg.SetTextSize(0.04)

            # Check if plotting in log scale
            logy = (n in logvars)
            if(logy):
                c.SetLogy()

            labels = []

            # Draw histograms and add each entry to the legend
            for hist in hists:
                if(logy):
                    hist.GetYaxis().SetRangeUser(1e-04,15*ymax)
                else:
                    hist.GetYaxis().SetRangeUser(0,1.2*ymax)

                hist.Draw('histsame')

                hname = str(hist.GetName())
                labelname = hname[hname.rindex('_')+1:]

                leg.AddEntry(hist,proclabels[labelname],'L')

            # Draw legend and save the plot
            leg.Draw('same')
            st.LDMX_lumi(c,0,'Simulation')

            c.SaveAs(outputdir+'/'+n+'_'+x+('_log.pdf' if logy else '.pdf'))


    # Loop over 2D plots
    for n in plotnames2d:
        if n == '':
            continue

        xvar = n[n.rindex('_')+1:]
        yvar = n[0:n.index('_')]

        if not xvar in binning or not yvar in binning:
            continue

        print ('plotting',yvar,'vs',xvar)

        # Loop over processes
        for proc in procs:
            infile = files[proc]
            tree = infile.FindObjectAny(treename2d)

            c = st.MakeCanvas('c','',600,600)

            # Loop over cut strings
            for seln in sel2d:
                selexp = sel2d[seln]
                print ('with selection',selexp)

                hist = TH2D('_'.join(['h',n,proc,seln]),'',binning[xvar][0],binning[xvar][1],binning[xvar][2],binning[yvar][0],binning[yvar][1],binning[yvar][2])

                c.cd()

                logx,logy = False,False

                if xvar in logvars:
                    logx = True
                if yvar in logvars:
                    logy = True

                if logx:
                    c.SetLogx()
                if logy:
                    c.SetLogy()

                c.SetLogz()
                c.SetLeftMargin(0.13);
                c.SetRightMargin(0.18);

                print ('Drawing',expr[n])
                if n in expr:
                    #tree.Draw(expr[n]+'_'.join(['>>h',n,proc,seln]),selexp,'COLZnorm')
                    tree.Draw(expr[n]+'_'.join(['>>h',n,proc,seln]),selexp,'COLZ')
                else:
                    #tree.Draw(n+'_'.join(['>>h',n,proc,seln]),selexp,'COLZnorm')
                    tree.Draw(n+'_'.join(['>>h',n,proc,seln]),selexp,'COLZ')

                # Histogram setup
                hist.GetXaxis().SetTitle(plotlabels[xvar])
                hist.GetXaxis().SetTitleSize(0.05)
                hist.GetYaxis().SetTitle(plotlabels[yvar])
                hist.GetYaxis().SetTitleSize(0.05)

                # Save plot
                c.SaveAs(outputdir+'/'+n+'_'+proc+'_'+seln+('_log.pdf' if logx or logy else '.pdf'))

    # Loop over efficiency variables to be plotted
    for n in effplotnames:
        if n == '' or not n in binning:
            continue

        print ('plotting efficiency vs',n)

        (outerloopitems, innerloopitems) = (procs, sel) if comparetype == 'sels' else (sel, procs)
        # Loop over outer loop items
        for x in outerloopitems:

            hists = []
            effs = []

            infile = None
            tree = None
            selexp = ''

            if comparetype == 'sels':
                infile = files[x]
                tree = infile.FindObjectAny(treename)
            else:
                selexp = sel[x]
                print ('with selection', selexp)

            isel = -1
            # Loop over inner loop items
            for y in innerloopitems:
                if comparetype == 'sels':
                    selexp = sel[y]
                    print ('with selection',selexp)
                else:
                    infile = files[y]
                    tree = infile.FindObjectAny(treename)

                isel += 1

                hist = TH1D('_'.join(['h',n,x,y]),'',binning[n][0],binning[n][1],binning[n][2])
                hist.SetLineColor(colors[y])
                hist.SetMarkerColor(colors[y])

                # Check if variable name corresponds to an expression
                if n in expr:
                    tree.Draw(expr[n]+'_'.join(['>>h',n,x,y]),selexp)
                else:
                    tree.Draw(n+'_'.join(['>>h',n,x,y]),selexp)

                # Histogram setup
                st.addOverFlow(hist)
                hist.GetXaxis().SetTitle(plotlabels[n])
                hists.append(hist)

            # Setup canvas
            c = st.MakeCanvas("c","",600,600)
            leg = TLegend(0.65,0.7,0.95,0.9)
            st.SetLegendStyle(leg)
            leg.SetTextSize(0.04)

            logy = (n in logvars)
            if(logy):
                c.SetLogy()

            if len(hists):
                xmin = hists[0].GetXaxis().GetXmin()
                xmax = hists[0].GetXaxis().GetXmax()
                c.DrawFrame(xmin,1e-6 if logy else 0,xmax,1.1)

            graphs = []
            emptyhist = None

            ihist = -1
            for hist in hists:
                ihist += 1
             
                hname = str(hist.GetName())
                labelname = hname[hname.rindex('_')+1:]

                xmin = hist.GetXaxis().GetXmin()
                xmax = hist.GetXaxis().GetXmax()

                effgr = pt.computeEffVsCutGraph(hist, n in reversecutvars)
                effgr.SetLineWidth(3)
                effgr.SetLineColor(colors[labelname])
                effgr.SetMarkerColor(colors[labelname])
                effgr.GetXaxis().SetTitle(plotlabels[n])
                effgr.GetYaxis().SetTitle('Efficiency')
                effgr.GetXaxis().SetTitleSize(0.04)
                effgr.GetYaxis().SetTitleSize(0.05)
                effgr.GetHistogram().GetXaxis().SetLimits(xmin,xmax)
                effgr.GetHistogram().GetYaxis().SetRangeUser(1e-6 if logy else 0.0,1.1)
                c.cd()
                effgr.Draw('Csame')
                if ihist == 0:
                    emptyhist = effgr.GetHistogram()
                leg.AddEntry(effgr,proclabels[labelname],'L')

                graphs.append(effgr)

            if emptyhist:
                emptyhist.Draw('AXIS')

            for graph in graphs:
                graph.Draw('Csame')

            c.cd()
            leg.Draw('same')
            st.LDMX_lumi(c,0,'Simulation')

            c.SaveAs(outputdir+'/eff_vs_'+n+'_'+y+'.pdf')

    for infile in files.values():
        infile.Close()


if __name__ == '__main__': main()
