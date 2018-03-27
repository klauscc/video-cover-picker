#ifndef IQA
#define IQA

#include "brisque.h"
#include "faceModel.h"
#include "ittivsmodel.h"
#include "nriqa.h"
#include <iostream>

#define MAX_POOL_SIZE 20

class iqa {
public:
  iqa(std::string allRangeFile = "../model/brisque/allrange",
      std::string allModel = "../model/brisque/allmodel");
  float computeScore(const std::string &imagePath, int thread_idx = 0);
  float getAllScore(const std::string imagePath, std::vector<float> &scores,
                    int thread_idx = 0);
  int getBestImages(std::string imageDir, float &bestScore,
                    std::string &bestImageName, std::string &bestImagePath);
  virtual ~iqa();

private:
  /* data */
  std::vector<Brisque> brisqueVec;
  std::vector<faceModel> faceModelVec;
  std::vector<IttiVSModel> ittiVec;
  std::vector<nriqa> nriqaVec;
  const float w[7] = {0.9, 0.8, 0.2, 0.1, 0.2, 0.2, 0};
};

#endif /* ifndef IQA */
