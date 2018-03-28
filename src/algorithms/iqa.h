/*================================================================
*   God Bless You. 
*   
*   file name: iqa.h
*   author: klaus
*   email: klaus.cheng@qq.com
*   created date: 2018/03/26
*   description:  interface of iqa algorithm
*
================================================================*/

#ifndef IQA
#pragma once
#define IQA

#include "brisque.h"
#include "faceModel.h"
#include "ittivsmodel.h"
#include "nriqa.h"
#include <iostream>

#define MAX_IMAGE_NUM 12

class iqa {
public:
  iqa(std::string modelDir = "/usr/local/iqa"); 
  float computeScore(const std::string &imagePath, int thread_idx = 0);
  float getAllScore(const std::string imagePath, std::vector<float> &scores,
                    int thread_idx = 0);

  /**
  * @brief: given a directory, get the best image name and full path.
  *
  * @param: std::string imageDir. image directory.
  *       : float &bestScore. the score of the best image in imageDir
  *       : std::string &bestImageName.
  *       : std::string &bestImagePath.
  *
  * @return: int
  */
  int getBestImage(std::string imageDir, float &bestScore,
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
