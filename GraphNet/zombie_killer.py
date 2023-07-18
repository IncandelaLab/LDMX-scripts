import uproot
import glob
import os
from tqdm import tqdm
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(30)

filepath = '/home/duncansw/GraphNet_input/v14/v3_tskim/XCal_total/*.root'

kills = 0
filelist = glob.glob(filepath)
nFiles = len(filelist)
for filename in tqdm(filelist, total=nFiles):
    with uproot.open(filename, interpretation_executor=executor) as file:
        if len(file.keys()) == 0:
            #print("Found zombie: {} KILLING...".format(filename), flush=True)
            #os.remove(filename)
            kills += 1
            #print("KILLED ZOMBIE", flush=True)

print("\nLevel cleared. {} zombies killed".format(kills), flush=True)