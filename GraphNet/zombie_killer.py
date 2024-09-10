## Loop through a specified filepath, and delete all zombie ROOT files
## Change "filepath" variable and uncomment the os.remove lines
## Also handles another rare uproot error on particular files (calling these files vampires)

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
    try:
        with uproot.open(filename, interpretation_executor=executor) as file:
            if len(file.keys()) == 0:
                #print("Found zombie: {} KILLING...".format(filename), flush=True)
                #os.remove(filename)
                kills += 1
                #print("KILLED ZOMBIE", flush=True)

    except OSError:
        #print("Found zombie: {} KILLING...".format(filename), flush=True)
        #os.remove(filename)
        kills +=1
        #print("KILLED ZOMBIE", flush=True)
        continue

    except uproot.deserialization.DeserializationError:
        #os.remove(filename)
        others += 1
        #print("KILLED VAMPIRE", flush=True)
        continue

print("\nLevel cleared. {} zombies killed".format(kills), flush=True)
print("\n {} vampires killed".format(others), flush=True)
