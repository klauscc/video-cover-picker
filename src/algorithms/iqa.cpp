#include "iqa.h"
#include "opencv2/opencv.hpp"
#include "utils/fileUtil.h"

#define METRIC_LENGTH 7

float iqa::computeScore(const std::string &imagePath) {
  std::vector<float> scores;
  getAllScore(imagePath, scores);
  float counted_metrics[METRIC_LENGTH] = {scores[1], scores[2], scores[3],
                                          scores[5], scores[6], scores[7],
                                          scores[8]};
  float score = 0;

  for (int i = 0; i < METRIC_LENGTH; ++i) {
    score += w[i] * counted_metrics[i];
  }
  return score;
}
int iqa::getBestImages(std::string imageDir, float &bestScore,
                       std::string &bestImageName, std::string &bestImagePath) {
  std::vector<std::string> fileNames;
  getFileNamesInDir(imageDir, fileNames);
  if (fileNames.size() == 0) {
    return 1;
  }
  std::vector<float> imgScores(fileNames.size());
  for (size_t j = 0; j < fileNames.size(); ++j) {
    std::string oriImgPath =
        (fs::path(imageDir) / fs::path(fileNames[j])).c_str();
    imgScores[j] = computeScore(oriImgPath);
    std::cout << oriImgPath << ":" << imgScores[j]  << std::endl;
  }

  bestImagePath = (fs::path(imageDir) / fs::path(fileNames[0])).c_str();
  bestImageName = fileNames[0];
  bestScore = 0;
  for (size_t i = 0; i < fileNames.size(); ++i) {
    float score = imgScores[i];
    if (score > bestScore) {
      bestScore = score;
      bestImagePath = (fs::path(imageDir) / fs::path(fileNames[i])).c_str();
      bestImageName = fileNames[i];
    }
  }
  return 0;
}

iqa::iqa(std::string allRangeFile, std::string allModel) {
  brisque.readRangeFile(allRangeFile);
  brisque.loadSvmModel(allModel);
  itti.IttiVSModelInit();
  nrIQA.nriqaInit();
}

float iqa::getAllScore(const std::string imagePath,
                       std::vector<float> &scores) {
  cv::Mat input = cv::imread(imagePath);
  // to_save klausScore, areaScore, posScore, brisqueScore, shrScore, nrss,
  // g_mssim, Rp, g_entropy
  scores.resize(9);

  // calculate klausScore, areaScore, posScore, brisqueScore
  cv::Mat output;
  float areaScore, posScore;
  float faceScore =
      facemodel.calculateFaceScore(input, output, areaScore, posScore);
  float brisqueScore = brisque.computeScore(input);
  brisqueScore = 1 - brisqueScore / 100;
  float score = faceScore * 2 + brisqueScore;
  if (brisqueScore < 0.3) {
    score = 0;
  }
  // calculate shrScore, nrssScore
  Mat vsiMap = itti.getIttiSaliency(input);
  std::vector<float> nriqaScores = nrIQA.nriqaQualityIndex(input, vsiMap);
  scores[0] = score;
  scores[1] = areaScore;
  scores[2] = posScore;
  scores[3] = brisqueScore;
  std::copy(nriqaScores.begin(), nriqaScores.end(), scores.begin() + 4);
  return score;
}

iqa::~iqa() {}
