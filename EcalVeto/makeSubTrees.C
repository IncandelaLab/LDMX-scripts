
void makeSubTrees(){

vector<TString> procs = {"1.0","0.1","0.01","0.001"};
//vector<TString> procs = {"bkg"};

TString location = "bdttrain_kf_newvars_040419";

for(auto proc : procs) {
  TFile* infile = new TFile(location + "/" + proc + "_tree.root");
  assert(infile);
  TFile* outfile = new TFile(location + "/" + proc + "_bdttrain_subtree.root","RECREATE"); // BDT training trees
  //TFile* outfile = new TFile(location + "/" + proc + "_bdttest_subtree.root","RECREATE"); // BDT evaluation trees

  TTree* intree = (TTree*)infile->Get("EcalVeto");
  assert(intree);

  intree->SetBranchStatus("*",1);

  TTree* outtree = intree->CloneTree(0);

  // For signal: use first 312500 events of each tree to train, hadd signal bdttrain sub-trees to get combined signal training file
  // For signal: use events > 312500 of each tree to evaluate BDT performance
  // For bkg: use first 1250000 events of tree to train
  // For bkg: use events > 1250000 of tree to evaluate BDT performance
  for(int ientry = 0; ientry < 312500; ++ientry) { // signal training samples
  //for(int ientry = 0; ientry < 1250000; ++ientry) { // bkg training sample
  //for(int ientry = 312500; ientry < intree->GetEntries(); ++ientry) { // proc testing samples
  //for(int ientry = 1250000; ientry < intree->GetEntries(); ++ientry) { // bkg testing sample
    intree->GetEntry(ientry);
    outtree->Fill();
  }

  outfile->cd();

  outtree->Write();

  outfile->Close();

  delete infile;
  delete outfile;

}

}
