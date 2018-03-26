#include "iqa.h"
#include "opencv2/opencv.hpp"
#include "utils/fileUtil.h"

float iqa::computeScore(const std::string &imagePath) {

  cv::Mat input = cv::imread(imagePath);
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
  return score;
}
int iqa::getBestImages(std::string imageDir, float &bestScore,
                       std::string &bestImageName, std::string &bestImagePath) {
  std::vector<std::string> fileNames;
  getFileNamesInDir(imageDir, fileNames);
  if (fileNames.size() == 0) {
    return 1;
  }
  bestImagePath = (fs::path(imageDir) / fs::path(fileNames[0])).c_str();
  bestImageName = fileNames[0];
  bestScore = 0;
  for (size_t j = 0; j < fileNames.size(); ++j) {
    std::string oriImgPath =
        (fs::path(imageDir) / fs::path(fileNames[j])).c_str();
    std::vector<float> metrics;
    // float score = computeScore(oriImgPath);
    float score = getAllScore(oriImgPath, metrics);
    std::string metric_str = "";
    for (auto v : metrics) {
      metric_str += std::to_string(v) + ",";
    }
    std::cout << "img: " << fileNames[j] << ". score: " << metric_str
              << std::endl;
    if (score > bestScore) {
      bestScore = score;
      bestImagePath = oriImgPath;
      bestImageName = fileNames[j];
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
