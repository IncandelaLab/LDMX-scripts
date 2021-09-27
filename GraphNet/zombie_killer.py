import uproot
import glob
import os

filepath = '/home/duncansw/GraphNet_input/v12/processed/*.root'

kills = 0
for filename in glob.glob(filepath):
	with uproot.open(filename) as file:
		if len(file.keys()) == 0:
			print("Found zombie: {} KILLING...".format(filename))
			os.remove(filename)
			kills += 1
			print("KILLED ZOMBIE")
		
print("Level cleared. {} zombies killed".format(kills))