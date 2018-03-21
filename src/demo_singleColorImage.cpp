#include <iostream>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <boost/format.hpp>

#include "algorithms/utils/fileUtil.h"
#include "algorithms/brisque.h"
#include "algorithms/faceModel.h"
#include "algorithms/ittivsmodel.h"
#include "algorithms/nriqa.h"

namespace fs = boost::filesystem;

int main(int argc, char **argv) {

  std::string ori_img_dir = "../dataset/singleColor_ori_imgs/";
  std::string qa_img_dir = "../dataset/singleColor_qa_imgs/";

  fs::path qa_path = fs::path(qa_img_dir);
  try {
    if (fs::exists(qa_path)) {
      fs::remove_all(qa_path);
    }
    fs::create_directories(qa_path);
  } catch (fs::filesystem_error const &e) {
    std::cout << "filesystem error" << std::endl;
  }

  faceModel facemodel;
  Brisque brisque;

  int numVideos = 14;
  for (int i = 1; i < numVideos; ++i) {
    std::vector<std::string> fileNames;
    std::string oriVidePath = ori_img_dir + std::to_string(i);
    std::string dstVideoPath = qa_img_dir + std::to_string(i);
    fs::create_directory(fs::path(dstVideoPath));

    getFileNamesInDir(oriVidePath, fileNames);
    for (size_t j = 0; j < fileNames.size(); ++j) {
      std::string oriImgPath = oriVidePath + "/" + fileNames[j];

      std::string dstImgPrefixPattern = "%1%_%2%_%3%_%4%_";

      // compute scores
      Mat input = imread(oriImgPath);
      Mat output;
      float areaScore, posScore;
      float faceScore =
          facemodel.calculateFaceScore(input, output, areaScore, posScore);
      float brisqueScore = brisque.computeScore(input);
      brisqueScore = 1 - brisqueScore / 100;
      float score = faceScore * 2 + brisqueScore;
      if (brisqueScore < 0.3) {
        score = 0;
      }
      std::cout << oriImgPath << "--"
                << "totalScore:" << score << ". brisqueScore: " << brisqueScore
                << std::endl;

      std::string dstImgPrefix = (boost::format(dstImgPrefixPattern) % score %
                                  areaScore % posScore % brisqueScore)
                                     .str();

      std::string dstImgPath = dstVideoPath + "/" + dstImgPrefix + fileNames[j];
      imwrite(dstImgPath, output);
    }
  }
}
