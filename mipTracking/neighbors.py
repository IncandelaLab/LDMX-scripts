import csv

# returns true if (x1,y1) and (x2,y2) are adjacent pixels
# Not yet validated for edges / partial pixels
def isNeighbor(x1, y1, x2, y2):
    if ((float(x1)-float(x2))*(float(x1)-float(x2))+(float(y1)-float(y2))*(float(y1)-float(y2))<76):
        return True
    return False

# neighborList.txt lists adjacent neighbors for each pixel numbered 0 through 3966
# each row is the list of neighbors for the first entry of the row
# rows with no neighbors listed represent indexes of non-existant pixels (ex: 7)
# this is done so that the neighbors of pixel x can always be found on line x

with open('neighborList.txt','w') as write_file:
  writer = csv.writer(write_file)
  numList = [0,0,0,0,0,0,0] # Stores number of pixels with 0, 1, 2, 3, 4, 5, or 6 neighbors, used for validation purposes
  for i in range(4000):
    neighborList = []
    neighborList.append(i)
    with open('cellmodule.txt') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=' ')
      for row in csv_reader:
        if int(row[0])==i:
          with open('cellmodule2.txt') as csv_file2: # sorry for the sloppy coding: I don't know how to safely parce the same file twice simultaneously, so I created 'cellmodule2.txt', which is and identical copy of 'cellmodule.txt'
            csv_reader2 = csv.reader(csv_file2, delimiter=' ')
            numb = 0
            for row2 in csv_reader2:
              if isNeighbor(row[1],row[2],row2[1],row2[2]) and not row[0]==row2[0]:
                neighborList.append(row2[0])
                numb+=1
            numList[numb] = numList[numb]+1
    for n in range(7-len(neighborList)):
      neighborList.append(-1)
    for entries in neighborList:
      write_file.write(str(entries))
      write_file.write(" ")
    write_file.write('\n')
  print numList
