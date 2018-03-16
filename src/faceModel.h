/*! \class faceModel
 *  \brief Brief class description
 *
 *  Detailed description
 */

#ifndef FACE_MODEL_H
#define FACE_MODEL_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>

class faceModel {
public:
  faceModel(std::string face_cascade_name_ =
                "../model/haarcascade_frontalface_default.xml");

  float calculateFaceScore(const cv::Mat &image, cv::Mat &output,
                           float &areaScore, float &posScore);
  virtual ~faceModel();

private:
  std::vector<cv::Rect> detectFace(const cv::Mat &image);

protected:
  std::string faceCascadeName; /*!< Member description */
  cv::CascadeClassifier faceCascade;

  struct faceBoundingBox {
    /* normalized face bounding box. each value is between [0,1]*/
    float x;
    float y;
    float width;
    float height;
    cv::Point2f center;
    faceBoundingBox(const cv::Rect &faceRect, int width_, int height_) {
      x = (float)faceRect.x / width_;
      y = (float)faceRect.y / height_;
      width = (float)faceRect.width / width_;
      height = (float)faceRect.height / height_;
      center.x = x + width / 2;
      center.y = y + height / 2;
    }
    inline float area() const { return width * height; }
    static int areaGreater(const faceBoundingBox &lhs,
                           const faceBoundingBox &rhs) {
      return lhs.area() > rhs.area();
    }
  };
};

#endif
