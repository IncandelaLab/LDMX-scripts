import uproot
import glob
import os
from multiprocessing import Pool

pathname = '/home/duncansw/GraphNet_input/v12/processed/*.root'

def killZombie(filename):
	with uproot.open(filename) as file:
		IsZombie = False
		if len(file.keys()) == 0:
			IsZombie = True
		if IsZombie:
			print("Found zombie: {} KILLING...".format(filename))
			os.remove(filename)
			print("KILLED ZOMBIE")
			return 1
		else:
			return 0 

if __name__ == '__main__':
	with Pool(20) as p:
		zombies = p.map(killZombie, f) for f in glob.glob(pathname)
	kills = sum(zombies)
	print("Done. {} zombies killed".format(kills))