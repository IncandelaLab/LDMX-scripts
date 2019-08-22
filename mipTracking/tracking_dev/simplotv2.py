import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sig_file='hits_Sig0_1000sig.txt'
bkg_file='hits_Bkg0_bkg.txt'
#adds = 'Boop0'
filePath = "/nfs/slac/g/ldmx/users/pmasters/ldmx-sw/scripts/EcalVeto/NewVars/hitPlotting/"
#fileName = "hits_" + str(adds) + "_1000sig.txt"
fileArr = np.loadtxt(bkg_file)

#Go through fileArr, find indices i and j where the next desired event is
#plot over those indices

N=4  #Create N plots consecutively
n=1
#i=1  #number of current event
i_start=0
i_end=0
for i in range(1,len(fileArr[:,0])):  #i=current event being examined
    #points=np.zeros((300,3))
    #print(i_end)
    #print("Before loop:  hit1="+str(fileArr[i_end,:]))
    #print("Looping")
    while i==fileArr[i_end,0]:
        #print('i_end='+str(i_end))
        print(fileArr[i_end,:])
        i_end+=1
    if i_start==i_end:  #no events found
        print("No hits found for event "+str(i))
        continue
    print("Plotting event "+str(i))
    #Events found.  Plot them:
    #print(i_start)
    #print(i_end)
    fig=plt.figure(i)
    ax=Axes3D(fig)
    ax.scatter(fileArr[i_start:i_end-1,3],fileArr[i_start:i_end-1,1],fileArr[i_start:i_end-1,2], c = 'r', marker='o')
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    plt.show()
    i_start=i_end
    #N+=1
    #if n>N: break
