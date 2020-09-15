import ROOT as r
import ROOTmanager as manager # Suggested standard import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
r.gSystem.Load('<your-path-to>/ldmx-sw/install/lib/libEvent.so')

# NOTE: Not currently working (For me, (Juan))
# Option 1: It looks like we have to add some packages to the ldmx-sw container
# Option 2: Update ROOT to 6.22/03. This should be easy but it's not on the
          # oficcial site so I'm not sure how it got into the derivative.
          # Also, it might just already work with samples produced by older
          # versions of the software. I tested this with WAB samples made with
          # in the unification derivative.

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    maxEvent = pdict['maxEvent']

    # Sample trees
    trees = {}
    for  gl, group in zip(group_labels, inlist): 
        trees[gl] = manager.load(group)
    
    # Construct tree processes
    procs = []
    for tkey in trees:
        procs.append(manager.TreeProcess(event_process, trees[tkey], ID=tkey))

    for proc in procs:
        
        # Branches
        proc.ecalSimHits = proc.addBranch('EcalHit', 'EcalSimHits_wab')

        # RUN
        # NOTE: In a most other scripts, the event process that is fed to
        # the process object and runs here would be filling histograms also
        # defined above (in this loop with the branches they need or eariler
        # depending on how the data is meant to be grouped and the full
        # histograms would then be fed to a unique plotting function defined
        # (ideally), below event_process and if name == '__main__':. Here,
        # the goal is just to plot
        proc.run(maxEvents=maxEvent)

def event_process(self):

    # 3D-ify
    fig=plt.figure()
    ax=Axes3D(fig)

    # Get Info
    x, y, z = [], [], []
    for sp in self.ecalSimHits:
       x.append( sp.getXPos() )
       y.append( sp.getYPos() )
       z.append( sp.getZPos() )

    # Plot whatever hits/tracks
    # Plotting X Y and Z in this order gives a more natural output
    # (i.e. Z points back instead of up)
    ax.scatter(z, x , y , c = 'b', marker='o')
    #ax.plot3D(zl,xl,yl, c = 'r')
    
    # Labels
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    
    # Invert x (y) axis view for our c-system
    buff = 2
    ax.set_ylim(max(x) + buff, min(x) - buff)
    plt.show()

if __name__ == "__main__":
    main()
