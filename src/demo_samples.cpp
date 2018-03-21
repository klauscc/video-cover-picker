#include <boost/filesystem.hpp>

#include "algorithms/iqa.h"
#include "algorithms/utils/fileUtil.h"

namespace fs = boost::filesystem;

int main(int argc, char *argv[]) {
  iqa iqa;
  std::string oriImageDir = "../dataset/samples/samples/";
  std::string qaImgDir = "../dataset/samples/samples_res/";
  std::vector<std::string> videoNames;
  getDirectoriesInDir(oriImageDir, videoNames);
  for (size_t i = 0; i < videoNames.size(); ++i) {
    std::string videoPath =
        (fs::path(oriImageDir) / fs::path(videoNames[i])).c_str();
    float score;
    std::string bestImageName;
    std::string bestImagePath =
        iqa.getBestImages(videoPath, score, bestImageName);
    bestImageName = "klaus_" + std::to_string(score) + "_" + bestImageName;
    std::string bestImageSavePath =
        (fs::path(qaImgDir) / fs::path(videoNames[i]) / fs::path(bestImageName))
            .c_str();
    fs::copy_file(bestImagePath, bestImageSavePath,
                  fs::copy_option::overwrite_if_exists);
    std::cout << "video:" << videoPath << ". bestImage:" << bestImageName
              << std::endl;
  }
  return 0;
}
