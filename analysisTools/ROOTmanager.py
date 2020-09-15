import sys
import ROOT as r

#TODO: Add a move to scratch dir function
#TODO: Make options for no output or based on input
#TODO: Make nolists independant for in and out

colors = {
        'kBlue': 600,
        'kGreen': 417,
        'kMagenta': 616,
        'kYellow': 400,
        'kViolet': 880,
        'kRed': 632,
        'kBlack': 1,
        'kCyan': 432,
        'kOrange': 800
        }

color_list = [colors[key] for key in colors]

lineStyles = {
        'kSolid': 1,
        'kDashed': 2,
        'kDotted': 3,
        'kDashDotted': 4
        }

lineStyle_list = [i for i in range(1,11)]

def parse(nolist = False):

    import glob
    import argparse

    ###########################
    # Interactive not rlly
    ###########################
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', action='store', dest='infiles', default=[],
            help='input file(s)')
    parser.add_argument('--indirs', nargs='+', action='store', dest='indirs', default='',
            help='Director(y/ies) of input files')
    parser.add_argument('-g','-groupls', nargs='+', action='store', dest='group_labels',
            default='', help='Human readable sample labels e.g. for legends')
    parser.add_argument('-o', nargs='+', action='store', dest='outfiles', default=[],
            help='outfile(s)')
    parser.add_argument('--outdir', action='store', dest='oudir', default='',
            help='outfile')
    parser.add_argument('--notlist', action='store_true', dest='nolist',
            help="can't be bothered rn")
    parser.add_argument('-m','--max', type=int, action='store', dest='maxEvent',
            default=-1, help='max events to run over for EACH group')
    args = parser.parse_args()

    ###########################
    # Input
    ###########################
    if args.infiles != []:
        inlist = [[f] for f in args.infiles] # Makes general loading easier
        if nolist or args.nolist == True:
            inlist = inlist[0]
    elif args.indirs != '':
        inlist = [glob.glob(indir + '/*.root') for indir in args.indirs]
    else:
        sys.exit('provide input')

    ###########################
    # Output
    ###########################
    if args.outfiles != []:
        outlist = args.outfiles
        if nolist or args.nolist == True:
            outlist = outlist[0]
    elif args.outdir != '':
        outlist = glob.glob(args.outdir + '/*.root')
    else:
        sys.exit('provide output')
    
    pdict = {
            'inlist':inlist,
            'groupls':args.group_labels,
            'outlist':outlist,
            'maxEvent':args.maxEvent
            }

    return pdict

def load(group,treeName='LDMX_Events'):

    tree = r.TChain(treeName)
    for f in group:
        tree.Add(f)

    return tree

class Histogram:

    def __init__(self, hist, title='', xlabel='x', ylabel='y',\
            color=1, lineStyle=1, fillStyle=1):
        self.hist = hist
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color = color
        self.lineStyle = lineStyle
        self.fillStyle = fillStyle

class TreeProcess:

    def __init__(self, event_process, tree=None, ID = '', color=1,\
            maxEvents=-1, pfreq=1000):

        self.event_process = event_process
        self.tree = tree
        self.ID = ID
        self.color = color
        self.maxEvents = maxEvents
        self.pfreq = pfreq

    def addBranch(self, ldmx_class, branch_name):

        if self.tree == None:
            sys.exit('Set tree')

        if ldmx_class == 'EventHeader':
            branch = r.ldmx.EventHeader()
        elif ldmx_class == 'SimParticle':
            branch = r.map(int, 'ldmx::'+ldmx_class)() 
        else:
            branch = r.std.vector('ldmx::'+ldmx_class)()

        self.tree.SetBranchAddress(branch_name,r.AddressOf(branch))

        return branch
 
    def run(self, maxEvents=-1, pfreq=1000):

        if maxEvents != -1: self.maxEvents = maxEvents
        else: self.maxEvents = self.tree.GetEntries()
        if pfreq != 1000: self.pfreq = pfreq
        
        self.event_count = 0
        while self.event_count < self.maxEvents:
            self.tree.GetEntry(self.event_count)
            if self.event_count%self.pfreq == 0:
                print('Processing Event: %s'%(self.event_count))
            self.event_process(self)
            self.event_count += 1

