#include "brisque.h"

int Brisque::loadSvmModel(std::string modelFilePath)
{
  if ((model = svm_load_model(modelFilePath.c_str())) == 0) {
    fprintf(stderr, "can't open model file allmodel\n");
    exit(1);
    return 1;
  }
  return 0;
}


float Brisque::computeScore(cv::Mat image) {
  double qualityscore;
  int i;
  IplImage orig = image;
  std::vector<double> brisqueFeatures;
  ComputeBrisqueFeature(&orig, brisqueFeatures);
  // rescale the brisqueFeatures vector from -1 to 1
  struct svm_node x[37];
  for (i = 0; i < 36; ++i) {
    float min = rescale_vector[i][0];
    float max = rescale_vector[i][1];
    x[i].value = -1 + (2.0 / (max - min) * (brisqueFeatures[i] - min));
    x[i].index = i + 1;
  }
  x[36].index = -1;
  int nr_class = svm_get_nr_class(model);
  double *prob_estimates = (double *)malloc(nr_class * sizeof(double));
  qualityscore = svm_predict_probability(model, x, prob_estimates);
  free(prob_estimates);
  return qualityscore;
}

int Brisque::readRangeFile(std::string rangeFilePath) {

  // check if file exists
  char buff[100];
  int i;
  FILE *rangeFile = fopen(rangeFilePath.c_str(), "r");
  // assume standard file format for this program
  fgets(buff, 100, rangeFile);
  fgets(buff, 100, rangeFile);
  // now we can fill the array
  for (i = 0; i < 36; ++i) {
    float a, b, c;
    fscanf(rangeFile, "%f %f %f", &a, &b, &c);
    rescale_vector[i][0] = b;
    rescale_vector[i][1] = c;
  }

  return 0;
}

Brisque::~Brisque() {
  if (model != NULL) {
    svm_free_and_destroy_model(&model);
  }
}
