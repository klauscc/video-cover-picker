#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

#include "algorithms/iqa_pool.h"
#include "algorithms/utils/fileUtil.h"

namespace fs = boost::filesystem;

void iqaVideo(std::string videoPath, std::string videoSavePath) {
  std::vector<std::string> saveFileName;
  if (fs::exists(fs::path(videoSavePath))) {
    getFileNamesInDir(videoSavePath, saveFileName);
    // if (saveFileName.size() == 2) {
    // continue;
    //}
  } else {
    fs::create_directory(fs::path(videoSavePath));
  }
  float score;
  std::string bestImageName, bestImagePath;
  long t1 = cv::getTickCount();
  int res = getBestImage(videoPath, score, bestImageName, bestImagePath);
  long t2 = cv::getTickCount();
  double t = (t2 - t1) / cv::getTickFrequency();
  if (res == 0) {
    bestImageName = "003_" + std::to_string(score) + "_" + bestImageName;
    std::string bestImageSavePath =
        (fs::path(videoSavePath) / fs::path(bestImageName)).c_str();
    std::cout << "video:" << videoPath << ". bestImage:" << bestImageName
              << ". takes " << t << "s" << std::endl;
    fs::copy_file(bestImagePath, bestImageSavePath,
                  fs::copy_option::overwrite_if_exists);
  } else {
    std::cout << "video:" << videoPath << " is empty!skip.." << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::string oriImageDir = "../dataset/video1000/samples/";
  std::string qaImgDir = "../dataset/video1000/samples_res/";
  std::vector<std::string> videoNames;
  getDirectoriesInDir(oriImageDir, videoNames);

  boost::asio::io_service ioService;
  boost::thread_group threadpool;
  boost::asio::io_service::work work(ioService);

  int thread_num = 3;
  for (int i = 0; i < thread_num; ++i) {
    threadpool.create_thread(
        boost::bind(&boost::asio::io_service::run, &ioService));
  }

  for (size_t i = 0; i < videoNames.size(); ++i) {
    std::string videoPath =
        (fs::path(oriImageDir) / fs::path(videoNames[i])).c_str();
    std::string videoSavePath =
        (fs::path(qaImgDir) / fs::path(videoNames[i])).c_str();
    ioService.post(boost::bind(iqaVideo, videoPath, videoSavePath));
  }
  threadpool.join_all();
  return 0;
}
