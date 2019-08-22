import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


adds = 'Boop0'
filePath = "/nfs/slac/g/ldmx/users/jmlazaro/ldmx-sw/scripts/EcalVeto/NewVars/outHitTxts/"
fileName = "hits_" + str(adds) + "_1000sig.txt"
fileArr = np.loadtxt(filePath + fileName)

def rowCount(csv_file):
    rowCount = 0
    for fileArr[rowCount] in fileArr:
	rowCount += 1
    return rowCount

rows = rowCount(filePath + fileName)

eventNumb = 0
xPos = 1
yPos = 2
zPos = 3
hitE = 4

currentRow = 0

x = {'x'+str(fileArr[currentRow][eventNumb]): [fileArr[currentRow][xPos]]}
y = {'y'+str(fileArr[currentRow][eventNumb]): [fileArr[currentRow][yPos]]}
z = {'z'+str(fileArr[currentRow][eventNumb]): [fileArr[currentRow][zPos]]}

print(x['x3.0'])
print(y['y3.0'])
print(z['z3.0'])

#IDEA:  Add radii of containment to current plot
#Need list of radii and centroids...

currentRow = 1
while currentRow < rows:
    #add hit to event
    if fileArr[currentRow][eventNumb] == fileArr[currentRow - 1][eventNumb]:                  
        x['x'+str(fileArr[currentRow][eventNumb])].append(fileArr[currentRow][xPos])
        y['y'+str(fileArr[currentRow][eventNumb])].append(fileArr[currentRow][yPos]) 
        z['z'+str(fileArr[currentRow][eventNumb])].append(fileArr[currentRow][zPos])
        currentRow += 1 	
    # start a new event and plot the one that just finished
    else:
        fig = plt.figure(fileArr[currentRow - 1][eventNumb])
        ax = fig.add_subplot(111, projection = '3d')
	print(str(x['x'+str(fileArr[currentRow - 1][eventNumb])]) + '\n' + str(y['y'+str(fileArr[currentRow - 1][eventNumb])]) + '\n' + str(z['z'+str(fileArr[currentRow - 1][eventNumb])]))
        ax.scatter(x['x'+str(fileArr[currentRow - 1][eventNumb])],y['y'+str(fileArr[currentRow - 1][eventNumb])],z['z'+str(fileArr[currentRow - 1][eventNumb])], c = 'r', marker='o')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z Layer')
	plt.show()
	print('I made a plot dude')
	currentRow += 1
	x['x'+str(fileArr[currentRow][eventNumb])] = [fileArr[currentRow][xPos]]
	y['y'+str(fileArr[currentRow][eventNumb])] = [fileArr[currentRow][yPos]]
	z['z'+str(fileArr[currentRow][eventNumb])] = [fileArr[currentRow][zPos]]
	# plot this event if it is the last one
	if currentRow + 1 == rows:
	    fig = plt.figure(fileArr[currentRow][eventNumb])
            ax.scatter(x['x'+str(fileArr[currentRow][xPos])],y['y'+str(fileArr[currentRow][yPos])],z['z'+str(fileArr[currentRow][zPos])], c = 'r', marker='o')
	    ax.set_xlabel('X')
	    ax.set_ylabel('Y')
	    ax.set_zlabel('Z Layer')
	    plt.show()
	    print('I made a plot dude')

	"""
	x = []
	y =[]

	for event in range(eventNumb + 1):

	fileList.append(filePath + "hit_" + str(adds) + "_" + str(event + 1) + ".txt")   
	except:
	 pass 
	for event in fileList:  
	i = 1
	with open(event) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=' ')
	for row in csv_reader:
	    x.append(float(row[1]))
	    y.append(float(row[2]))    

	    # Plot
	    plt.figure(i)
	    plt.scatter(x, y)
	    plt.title("1 GeV A' hits outside68x3 containment")
	    plt.xlabel('x')
	    plt.ylabel('y')
	    plt.show()
	    i += 1
	"""
