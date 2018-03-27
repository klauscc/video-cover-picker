/*================================================================
*   God Bless You. 
*   
*   file name: faceModel.cpp
*   author: klaus
*   email: klaus.cheng@qq.com
*   created date: 2018/03/26
*   description: implemention
*
================================================================*/

#include "faceModel.h"
#include <algorithm>

float faceModel::calculateFaceScore(const cv::Mat &image, cv::Mat &output,
                                    float &areaScore, float &positionScore) {
  positionScore = 0;
  areaScore = 0;
  float finalScore;
  // std::vector<cv::Rect> faces = detectFaceOpencv(image);
  std::vector<cv::Rect> faces = detectFaceDlib(image);
  int height = image.rows;
  int width = image.cols;

  std::vector<faceBoundingBox> allFaceBoxes;

  allFaceBoxes.reserve(faces.size());
  // normalize face rect to [0,1]
  for (auto iter = faces.begin(); iter != faces.end(); ++iter) {
    allFaceBoxes.push_back(faceBoundingBox(*iter, width, height));
  }

  // remove small faces
  std::vector<faceBoundingBox> faceBoxes;
  faceBoxes.reserve(faces.size());
  for (int i = 0; i < (int)allFaceBoxes.size(); i++) {
    if (allFaceBoxes[i].height > 0.14) {
      faceBoxes.push_back(allFaceBoxes[i]);
    }
  }

  // draw faces
  output = image.clone();
  for (int i = 0; i < (int)faceBoxes.size(); i++) {
    cv::rectangle(
        output, cv::Point(faceBoxes[i].x * width, faceBoxes[i].y * height),
        cv::Point(faceBoxes[i].x * width + faceBoxes[i].width * width,
                  faceBoxes[i].y * height + faceBoxes[i].height * height),
        cv::Scalar(255, 0, 0));
  }

  if (faceBoxes.size() == 0) {
    return 0;
  }
  // calculate areaScore based on biggest face
  std::sort(faceBoxes.begin(), faceBoxes.end(), faceBoundingBox::areaGreater);
  float bestArea = 0.4;
  areaScore = 1 - std::abs(faceBoxes[0].height - bestArea);

  // calculate positionScore based on biggest face
  auto box_center = faceBoxes[0].center;
  if (faceBoxes.size() == 1) {
    positionScore =
        1 - (std::abs(box_center.x - 0.5) + std::abs(box_center.y - 0.5)) / 2;
  } else if (faceBoxes.size() >= 2) {
    float min_varience =
        (std::abs(box_center.x - 0.5) + std::abs(box_center.y - 0.5)) / 2;
    min_varience = std::min<float>(
        min_varience,
        (std::abs(box_center.x - 0.3) + std::abs(box_center.y - 0.5)) / 2);
    min_varience = std::min<float>(
        min_varience,
        (std::abs(box_center.x - 0.7) + std::abs(box_center.y - 0.5)) / 2);
    positionScore = 1 - min_varience;
  }
  finalScore = areaScore * positionScore;
  return finalScore;
}

faceModel::faceModel() {
}

std::vector<cv::Rect> faceModel::detectFaceDlib(const cv::Mat &image) {
  dlib::cv_image<dlib::bgr_pixel> cimg(image);
  std::vector<dlib::rectangle> dets = detector(cimg);
  std::vector<cv::Rect> cv_dets;
  cv_dets.reserve(dets.size());
  for (int i = 0; i < (int)dets.size(); i++) {
    int l_ = dets[i].left();
    int t_ = dets[i].top();
    int w_ = dets[i].right() - dets[i].left();
    int h_ = dets[i].bottom() - dets[i].top();
    cv::Rect rect(l_, t_, w_, h_);
    cv_dets.push_back(rect);
  }
  return cv_dets;
}
faceModel::~faceModel() {}
