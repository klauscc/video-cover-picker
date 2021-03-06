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
    std::string videoSavePath =
        (fs::path(qaImgDir) / fs::path(videoNames[i])).c_str();
    std::vector<std::string> saveFileName;
    if (fs::exists(fs::path(videoSavePath))) {
      getFileNamesInDir(videoSavePath, saveFileName);
      if (saveFileName.size() == 2) {
        continue;
      }
    } else {
      fs::create_directory(fs::path(videoSavePath));
    }
    float score;
    std::string bestImageName, bestImagePath;
    int res = iqa.getBestImages(videoPath, score, bestImageName, bestImagePath);
    if (res == 0) {
      bestImageName = "klaus_" + std::to_string(score) + "_" + bestImageName;
      std::string bestImageSavePath =
          (fs::path(qaImgDir) / fs::path(videoNames[i]) /
           fs::path(bestImageName))
              .c_str();
      std::cout << "video:" << videoPath << ". bestImage:" << bestImageName
                << std::endl;
      fs::copy_file(bestImagePath, bestImageSavePath,
                    fs::copy_option::overwrite_if_exists);
    } else {
      std::cout << "video:" << videoPath << " is empty!skip.." << std::endl;
    }
  }
  return 0;
}
