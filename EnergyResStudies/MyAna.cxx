#include "Framework/EventProcessor.h"
#include "Ecal/Event/EcalVetoResult.h"
#include "Ecal/Event/EcalHit.h"

#include <math.h>

class MyAna : public framework::Analyzer {
public:
  MyAna(const std::string& name, framework::Process& p)
  : framework::Analyzer(name, p) {}
  ~MyAna() = default;
  void onProcessStart();
  void analyze(const framework::Event& event) final;
};



void MyAna::onProcessStart(){
  getHistoDirectory();
  histograms_.create("SummedDet", "Summed ECAL energy [MeV]", 100, 0.0, 8000.0);
  histograms_.create("SummedDetRecHits", "Summed ECAL energy [MeV]", 100, 0.0, 8000.0);  
}

void MyAna::analyze(const framework::Event& event) {
  // std::cout << " ---------------------------------------------" << std::endl;
  auto ecalVeto{event.getObject<ldmx::EcalVetoResult>("EcalVeto","")};
  std::vector<ldmx::EcalHit> ecalRecHits = event.getCollection<ldmx::EcalHit>("EcalRecHits", "");

  std::sort(ecalRecHits.begin(), ecalRecHits.end(),
            [](const ldmx::EcalHit &lhs, const ldmx::EcalHit &rhs) {
              return lhs.getID() < rhs.getID();
            });

  double totalRecEnergy{0.};
  for (const ldmx::EcalHit &recHit : ecalRecHits) {
    // skip anything that digi flagged as noise
    if (recHit.isNoise()) {
      continue;
    }
  totalRecEnergy += recHit.getEnergy();
  }

  // Fill histograms
  histograms_.fill("SummedDet", ecalVeto.getSummedDet() );
  histograms_.fill("SummedDetRecHits", totalRecEnergy );
}

DECLARE_ANALYZER(MyAna);


