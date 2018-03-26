#ifndef IQA
#define IQA

#include "brisque.h"
#include "faceModel.h"
#include "ittivsmodel.h"
#include "nriqa.h"
#include <iostream>

class iqa {
public:
  iqa(std::string allRangeFile = "../model/brisque/allrange",
      std::string allModel = "../model/brisque/allmodel");
  float computeScore(const std::string &imagePath);
  float getAllScore(const std::string imagePath, std::vector<float> &scores);
  int getBestImages(std::string imageDir, float &bestScore,
                    std::string &bestImageName, std::string &bestImagePath);
  virtual ~iqa();

private:
  /* data */
  Brisque brisque;
  faceModel facemodel;
  IttiVSModel itti;
  nriqa nrIQA;
  const float w[7] = {0.9, 0.8, 0.2, 0.1, 0.2, 0.2, 0};
};

#endif /* ifndef IQA */
