/*================================================================
*   God Bless You.
*
*   file name: brisque.h
*   author: klaus
*   email: klaus.cheng@qq.com
*   created date: 2018/03/26
*   description:
*
================================================================*/

#ifndef IQA_BRESQUE
#pragma once
#define IQA_BRESQUE

#include "brisque/brisque.h"
#include "brisque/libsvm/svm.h"
#include "opencv2/opencv.hpp"

// rescaling based on training data i libsvm

class Brisque {
public:
  Brisque() {}
  float computeScore(cv::Mat image);
  virtual ~Brisque();

  int readRangeFile(std::string rangeFilePath = "../model/brisque/allrange");
  int loadSvmModel(std::string modelFilePath = "../model/brisque/allmodel");

private:
  /* data */
  struct svm_model *model;
  float rescale_vector[36][2];
};

#endif /* ifndef IQA_BRESQUE                                                   \
                                                                               \
                                                                               \
 */
