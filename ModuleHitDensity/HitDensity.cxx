#include "Framework/EventProcessor.h"
#include "Ecal/Event/EcalVetoResult.h"
#include "Ecal/Event/EcalHit.h"
#include "DetDescr/EcalID.h"
#include "DetDescr/include/DetDescr/EcalGeometry.h"

#include <math.h>

class HitDensity : public framework::Analyzer {
public:
  HitDensity(const std::string& name, framework::Process& p)
  : framework::Analyzer(name, p) {}
  ~HitDensity() = default;
  void onProcessStart();
  void analyze(const framework::Event& event) final;
};

void HitDensity::onProcessStart(){
  getHistoDirectory();
  histograms_.create("NumModWith0Hits", "Num of modules with 0 hit", 239, -0.5, 238.5);
  histograms_.create("NumModWith1Hits", "Num of modules with 1 hit", 239, -0.5, 238.5);
  histograms_.create("NumModWith2Hits", "Num of modules with 2 hits", 239, -0.5, 238.5);
  histograms_.create("NumModWithMoreThan2Hits", "Num of modules with >2 hits", 239, -0.5, 238.5);
  histograms_.create("NumHitsIfMoreThan2", "Num of hits for modules with >2 hits", 39, -0.5, 38.5);

  // histograms_.create("NumCellWith0Hits", "Num of cells with 0 hit", 239, -0.5, 102816.5);
  // histograms_.create("NumCellWith1Hits", "Num of cells with 1 hit", 239, -0.5, 102816.5);
  // histograms_.create("NumCellWith2Hits", "Num of cells with 2 hits", 239, -0.5, 102816.5);
  // histograms_.create("NumCellWithMoreThan2Hits", "Num of cels with >2 hits", 239, -0.5, 102816.5);
}

void HitDensity::analyze(const framework::Event& event) {
  // std::cout << " ---------------------------------------------" << std::endl;

  auto ecalRecHits{event.getCollection<ldmx::EcalHit>("EcalRecHits", "")};

//   const auto& ecal_geometry = getCondition<ldmx::EcalGeometry>(
//       ldmx::EcalGeometry::CONDITIONS_OBJECT_NAME);
// EcalID i_th_ecal_id;
// for (i_th_ecal_id : all_IDs) {
//   auto i_th_id_pos = ecal_geometry.getPosition(i_th_ecal_id);
// }

  // std::sort(ecalRecHits.begin(), ecalRecHits.end(),
  //           [](const ldmx::EcalHit &lhs, const ldmx::EcalHit &rhs) {
  //             return lhs.getID() < rhs.getID();
  //           });

  std::vector<int> myCostumCellIDs;
  std::set<int> myCostumCellIDsSet;
  std::vector<int> myCostumModIDs;
  std::set<int> myCostumModIDsSet;
  for (const ldmx::EcalHit &recHit : ecalRecHits) {
    ldmx::EcalID ecal_id(recHit.getID());
    int layer = ecal_id.layer() + 1;
    int moduleID = ecal_id.getModuleID() + 1;
    int cellID = ecal_id.getCellID() + 1;
    int myModConstumID = layer * 100 + moduleID;
    int myCellConstumID = layer * 1000000 + moduleID * 10000 + cellID;
    int id{recHit.getID()};
    myCostumModIDs.push_back(myModConstumID);
    myCostumModIDsSet.insert(myModConstumID);
    // std::cout << " myModConstumID " << myModConstumID << std::endl;
    myCostumCellIDs.push_back(myCellConstumID);
    myCostumCellIDsSet.insert(myCellConstumID);
    
  }

  std::map<int, int> moduleHits;
  for (const int &myCostumModID : myCostumModIDs) {
    moduleHits[myCostumModID]++;
  }

  int numModWith0Hits{0};
  int numModWith1Hits{0};
  int numModWith2Hits{0};
  int numModWithMoreThan2Hits{0};

  // all modules is 34*7 = 238
  numModWith0Hits = 34*7 - myCostumModIDsSet.size();
  // std::cout << " myCostumModIDsSet.size() " << myCostumModIDsSet.size() << std::endl;
  for (const auto &moduleHit : moduleHits) {
    if (moduleHit.second == 1) {
      numModWith1Hits++;
    }
    else if (moduleHit.second == 2) {
      numModWith2Hits++;
    }
    else if (moduleHit.second > 2) {
      // std::cout << " moduleHit.second  " << moduleHit.second  << std::endl;
      histograms_.fill("NumHitsIfMoreThan2", moduleHit.second);
      numModWithMoreThan2Hits++;
    }
  }


  histograms_.fill("NumModWith0Hits", numModWith0Hits);
  if (numModWith1Hits > 0) histograms_.fill("NumModWith1Hits", numModWith1Hits);
  if (numModWith2Hits > 0) histograms_.fill("NumModWith2Hits", numModWith2Hits);
  if (numModWithMoreThan2Hits > 0) histograms_.fill("NumModWithMoreThan2Hits", numModWithMoreThan2Hits);

  // repeat with cells
  // ---------------------------------------------------------

  // std::map<int, int> cellHits;
  // for (const int &myCellID : myCostumCellIDs) {
  //   cellHits[myCellID]++;
  // }

  // int numCellWith0Hits{0};
  // int numCellWith1Hits{0};
  // int numCellWith2Hits{0};
  // int numCellWithMoreThan2Hits{0};

  // numCellWith0Hits = 34*7*432 - myCostumCellIDsSet.size();

  //   for (const auto &cellHit : cellHits) {
  //   if (cellHit.second == 1) {
  //     numCellWith1Hits++;
  //   }
  //   else if (cellHit.second == 2) {
  //     numCellWith2Hits++;
  //   }
  //   else if (cellHit.second > 2) {
  //     numCellWithMoreThan2Hits++;
  //   }
  // }

  // histograms_.fill("NumCellWith0Hits", numCellWith0Hits);
  // if (numCellWith1Hits > 0) histograms_.fill("NumCellWith1Hits", numCellWith1Hits);
  // if (numCellWith2Hits > 0) histograms_.fill("NumCellWith2Hits", numCellWith2Hits);
  // if (numCellWithMoreThan2Hits > 0) histograms_.fill("NumCellWithMoreThan2Hits", numCellWithMoreThan2Hits);

}

DECLARE_ANALYZER(HitDensity);


