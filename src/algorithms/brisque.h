#ifndef IQA_BRESQUE
#define IQA_BRESQUE

#include "brisque/brisque.h"
#include "brisque/libsvm/svm.h"
#include "opencv2/opencv.hpp"

// rescaling based on training data i libsvm

class brisque {
public:
  brisque(std::string allRangeFile = "../model/brisque/allrange", std::string allModel = "../model/brisque/allmodel");
  float computeScore(cv::Mat image);
  virtual ~brisque();

private:
  int readRangeFile(std::string rangeFilePath);

private:
  /* data */
  struct svm_model *model;
  struct svm_node x[37];
  float rescale_vector[36][2];
};

#endif /* ifndef IQA_BRESQUE                                                   \
                                                                               \
                                                                               \
 */
