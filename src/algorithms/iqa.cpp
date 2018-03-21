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

std::string iqa::getBestImages(std::string imageDir, float &bestScore,
                               std::string &bestImageName) {
  std::vector<std::string> fileNames;
  getFileNamesInDir(imageDir, fileNames);
  std::string bestImage = (fs::path(imageDir) / fs::path(fileNames[0])).c_str();
  bestImageName = fileNames[0];
  bestScore = 0;
  for (size_t j = 0; j < fileNames.size(); ++j) {
    std::string oriImgPath =
        (fs::path(imageDir) / fs::path(fileNames[j])).c_str();
    float score = computeScore(oriImgPath);
    if (score > bestScore) {
      bestScore = score;
      bestImage = oriImgPath;
      bestImageName = fileNames[j];
    }
  }
  return bestImage;
}

iqa::iqa(std::string allRangeFile, std::string allModel) {
  brisque.readRangeFile(allRangeFile);
  brisque.loadSvmModel(allModel);
}

iqa::~iqa() {}
