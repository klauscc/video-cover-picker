#ifndef IQA_BRESQUE
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
