import ROOT
#ROOT.gSystem.Load('libFramework.so')
ROOT.gSystem.Load('libTracking_Event.so') 
ROOT.gSystem.Load('libRecon_Event.so')
ROOT.gSystem.Load('libEcal_Event.so') 
ROOT.gSystem.Load('libHcal_Event.so') 
ROOT.gSystem.Load('libTrigger_Event.so') 
ROOT.gSystem.Load('libSimCore_Event.so')

# Open the ROOT file
file_name = "../CutBasedDM/ecalPnIn.root"
root_file = ROOT.TFile.Open(file_name)

# Get the tree from the file
tree_name = "LDMX_Events"
tree = root_file.Get(tree_name)

# Check if the tree is loaded correctly
if not tree:
    print(f"Error: Tree '{tree_name}' not found in the file '{file_name}'")
    exit(1)

# Get the number of entries in the tree
n_entries = tree.GetEntries()
print(f"Number of entries in the tree: {n_entries}")

# Loop over the events
for i in range(n_entries):
    tree.GetEntry(i)
    
    # Access the EcalSimHits_sim branch
    ecal_sim_hits = getattr(tree, 'EcalSimHits_sim', None)
    if ecal_sim_hits:
        # Example of how to access properties of the EcalSimHits_sim branch
        # Adjust this section according to the structure of EcalSimHits_sim
        print(f"Event {i}: EcalSimHits_sim contains {ecal_sim_hits.size()} hits")

        for hit in ecal_sim_hits:
            # I looked up the function here
            # https://github.com/LDMX-Software/ldmx-sw/blob/trunk/SimCore/include/SimCore/Event/SimCalorimeterHit.h
            print(f"  Hit energy: {hit.getEdep()}, Position: ({hit.getPosition()[0]}, {hit.getPosition()[1]}, {hit.getPosition()[2]})")

    else:
        print(f"Event {i}: No EcalSimHits_sim data")

# Close the file
root_file.Close()
