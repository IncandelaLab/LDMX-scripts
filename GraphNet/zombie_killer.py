import uproot
import glob
import os
from multiprocessing import Pool

pathname = '/home/duncansw/GraphNet_input/v12/processed/*.root'

kills = 0
for filename in glob.glob(pathname):
	with uproot.open(filename) as file:
		IsZombie = False
		if len(file.keys()) == 0:
			IsZombie = True
		if IsZombie:
			print("Found zombie: {} KILLING...".format(filename))
			os.remove(filename)
			kills += 1
			print("KILLED ZOMBIE")
			
print("Done. {} zombies killed".format(kills))
