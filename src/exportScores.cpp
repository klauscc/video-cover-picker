#include <boost/filesystem.hpp>
#include <fstream>

#include "algorithms/iqa.h"
#include "algorithms/utils/fileUtil.h"

namespace fs = boost::filesystem;

int main(int argc, char *argv[]) {

  iqa iqa;
  std::string oriImageDir = "../dataset/samples/samples/";
  std::string saveCsvName = "../dataset/samples/score.csv";
  std::ofstream csvStream;
  csvStream.open(saveCsvName);
  std::vector<std::string> videoNames;
  getDirectoriesInDir(oriImageDir, videoNames);
  csvStream << "fileName,klausScore,areaScore,posScore,brisqueScore,shrScore,"
               "nrss,g_mssim,Rp,g_entropy\n";
  for (size_t i = 0; i < videoNames.size(); ++i) {
    std::string videoPath =
        (fs::path(oriImageDir) / fs::path(videoNames[i])).c_str();
    std::vector<std::string> fileNames;
    getFileNamesInDir(videoPath, fileNames);
    for (auto fileName : fileNames) {
      std::string oriImgPath =
          (fs::path(videoPath) / fs::path(fileName)).c_str();
      std::vector<float> metrics;
      float score = iqa.getAllScore(oriImgPath, metrics);
      std::string metric_str = videoNames[i] + "/" + fileName + ",";
      for (size_t i = 0; i < metrics.size() - 1; ++i) {
        metric_str += std::to_string(metrics[i]) + ",";
      }
      metric_str += std::to_string(metrics[metrics.size() - 1]) + "\n";
      std::cout << metric_str << std::endl;
      csvStream << metric_str;
    }
  }
  return 0;
}
