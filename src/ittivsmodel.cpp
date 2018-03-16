#include "ittivsmodel.h"

vector<Mat> IttiVSModel::getPyramid(Mat image, int level) {
  vector<Mat> intPyramid;
  buildPyramid(image, intPyramid, level, BORDER_DEFAULT);
  return intPyramid;
}

Mat IttiVSModel::normanization(Mat src) {
  int kernelSize = 2;
  double max_value = 255;

  double maxVal = 0;
  Mat m;
  normalize(src, m, 0, max_value, NORM_MINMAX);
  // minMaxLoc(src, NULL, &maxVal, NULL, NULL);
  // m = max_value * src / maxVal;
  vector<double> localMax;
  double mean_max = 0;

  for (int i = 0; i < src.rows - kernelSize; i += kernelSize)
    for (int j = 0; j < src.cols - kernelSize; j += kernelSize) {
      Mat m_part = m(Rect(j, i, kernelSize, kernelSize));
      minMaxLoc(m_part, NULL, &maxVal, NULL, NULL);
      localMax.push_back(maxVal);
    }
  if (src.cols % kernelSize != 0) {
    Mat m_part =
        m(Rect(src.cols - kernelSize + 1, 0, kernelSize - 1, src.rows));
    minMaxLoc(m_part, NULL, &maxVal, NULL, NULL);
    localMax.push_back(maxVal);
  }
  if (src.rows % kernelSize != 0) {
    Mat m_part =
        m(Rect(0, src.rows - kernelSize + 1, src.cols, kernelSize - 1));
    minMaxLoc(m_part, NULL, &maxVal, NULL, NULL);
    localMax.push_back(maxVal);
  }
  int iter = 0;
  for (size_t i = 0; i < localMax.size(); i++) {
    if (max_value > localMax[i]) {
      mean_max += localMax[i];
      iter++;
    }
  }
  mean_max = mean_max / iter;
  return (max_value - mean_max) * (max_value - mean_max) * m;
}

// Returns 0,45,90,135 deg rotated and filtered images.
vector<Mat> IttiVSModel::getGaborImages(Mat img, int kernelSize) {
  double sigma = 3;
  double lambda = CV_PI;
  double gamma = 1;
  double psi = 0;
  vector<Mat> gaborImg(4);
  Mat dst;

  Mat kernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, 0, lambda,
                              gamma, psi, CV_32F);
  filter2D(img, dst, -1, kernel);
  gaborImg[0] = dst;

  kernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, CV_PI / 4,
                          lambda, gamma, psi, CV_32F);
  filter2D(img, dst, -1, kernel);
  gaborImg[1] = dst;

  kernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, CV_PI / 2,
                          lambda, gamma, psi, CV_32F);
  filter2D(img, dst, -1, kernel);
  gaborImg[2] = dst;

  kernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, (3 * CV_PI) / 4,
                          lambda, gamma, psi, CV_32F);
  filter2D(img, dst, -1, kernel);
  gaborImg[3] = dst;
#ifdef ORIENTATION_DEBUG
  namedWindow("img45", CV_WINDOW_AUTOSIZE);
  imshow("img45", gaborImg[1]);
#endif // ORIENTATION_DEBUG
  return gaborImg;
}

vector<Mat> IttiVSModel::colorFeatureMap(vector<Mat> rgbyCh, int depth) {
  int c, delta, up;
  vector<Mat> colorMap(12),
      colorMapNorm(12); // first 6 -> RG maps next 6 BY maps
  vector<Mat> rPyr(depth);
  vector<Mat> gPyr(depth);
  vector<Mat> bPyr(depth);
  vector<Mat> yPyr(depth);

  // Find pyramids of color channels
  rPyr = getPyramid(rgbyCh[0], depth - 1);
  gPyr = getPyramid(rgbyCh[1], depth - 1);
  bPyr = getPyramid(rgbyCh[2], depth - 1);
  yPyr = getPyramid(rgbyCh[3], depth - 1);

  Mat tmpRG, tmpBY, dstRG, dstBY;

  for (c = 2; c <= 4; c++) {
    for (delta = 3; delta <= 4; delta++) {
      // Scale up
      // tmp = intPyr[c + delta];
      tmpRG = gPyr[c + delta] - rPyr[c + delta];
      tmpBY = yPyr[c + delta] - bPyr[c + delta];
      for (up = 1; up <= delta; up++) {
        pyrUp(tmpRG, dstRG, Size(tmpRG.cols * 2, tmpRG.rows * 2));
        tmpRG = dstRG;
        pyrUp(tmpBY, dstBY, Size(tmpBY.cols * 2, tmpBY.rows * 2));
        tmpBY = dstBY;
      }
      // Size of images should be same for subtraction
      // scuuessive scale down and scale up ulters the size if the size if not
      // power of 2
      if (rPyr[c].size() != dstRG.size()) {
        // Opencv rounds up the size to upper value if the size is odd while
        // scaling down Hence we discard the extra row/col while subtracting
        // C(2,5) is stored in colorMap[0] and so on.... hence the magic
        // expression "2*c+delta-7" !!!
        Mat mapres;
        resize(dstRG, mapres, cv::Size(rPyr[c].cols, rPyr[c].rows));
        colorMap[2 * c + delta - 7] = abs((rPyr[c] - gPyr[c]) - mapres);
        resize(dstBY, mapres, cv::Size(bPyr[c].cols, bPyr[c].rows));
        colorMap[2 * c + delta - 7 + 6] = abs((bPyr[c] - yPyr[c]) - mapres);
        // std::cout << "image size mismatch; need to throw few rows and cols\n"
        // ;
      } else {
        colorMap[2 * c + delta - 7] = abs((rPyr[c] - gPyr[c]) - dstRG);
        colorMap[2 * c + delta - 7 + 6] = abs((bPyr[c] - yPyr[c]) - dstBY);
      }
      colorMapNorm[2 * c + delta - 7] =
          normanization(colorMap[2 * c + delta - 7]);
      colorMapNorm[2 * c + delta - 7 + 6] =
          normanization(colorMap[2 * c + delta - 7 + 6]);
    }
  }
#ifdef COLOR_DEBUG
  namedWindow("cmap0", CV_WINDOW_AUTOSIZE);
  imshow("cmap0", colorMapNorm[0]);

  namedWindow("cmap6", CV_WINDOW_AUTOSIZE);
  imshow("cmap6", colorMapNorm[6]);
#endif // COLOR_DEBUG
  return colorMapNorm;
}

Mat IttiVSModel::aggregateFeatureMap(vector<Mat> featureMap) {
  Mat aggMap, tmp, dst;
  int c, delta, down;

  // add maps at level 4.. featureMap[4] and [5] stores these maps
  cv::add(featureMap[4], featureMap[5], aggMap); // saturate add

  for (c = 2; c <= 3; c++) {
    for (delta = 3; delta <= 4; delta++) {
      tmp = featureMap[2 * c + delta - 7]; // I(2,5) is stored on Map[0]
      for (down = 1; down <= 4 - c; down++) {
        pyrDown(tmp, dst);
        tmp = dst;
      }
      if (aggMap.size() == dst.size()) {
        cv::add(aggMap, dst, aggMap);
      } else {
        Mat mapres;
        resize(dst, mapres, cv::Size(aggMap.cols, aggMap.rows));
        cv::add(aggMap, dst(Range(0, aggMap.rows), Range(0, aggMap.cols)),
                aggMap);
      }
    }
  }
  Mat aggMapNorm = normanization(aggMap);
  return aggMapNorm;
}

// Find the intensity/orientation 6 feature maps
vector<Mat> IttiVSModel::intFeatureMaps(vector<Mat> intPyr) {
  int c, s, delta, up;
  vector<Mat> intMap(6), intMapNorm(6);
  Mat tmp, dst;

  for (c = 2; c <= 4; c++) {
    for (delta = 3; delta <= 4; delta++) {
      // Scale up
      tmp = intPyr[c + delta];
      for (up = 1; up <= delta; up++) {
        pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
        tmp = dst;
      }
      // Size of images should be same for subtraction
      // scuuessive scale down and scale up ulters the size if the size if not
      // power of 2
      if (intPyr[c].size() != dst.size()) {
        // Opencv rounds up the size to upper value if the size is odd while
        // scaling down Hence we discard the extra row/col while subtracting
        Mat mapres;
        resize(dst, mapres, cv::Size(intPyr[c].cols, intPyr[c].rows));
        intMap[2 * c + delta - 7] = cv::abs(intPyr[c] - mapres);
        // std::cout << "image size mismatch; need to throw few rows and cols\n"
        // ;
      } else {
        intMap[2 * c + delta - 7] = cv::abs(intPyr[c] - dst);
      }
      intMapNorm[2 * c + delta - 7] = normanization(intMap[2 * c + delta - 7]);
    }
  }
#ifdef INT_DEBUG
  namedWindow("map_2_5", CV_WINDOW_AUTOSIZE);
  imshow("map_2_5", intMapNorm[0]);

  // namedWindow( "map_3_6", CV_WINDOW_AUTOSIZE );
  // imshow( "map_3_6", intMapNorm[2]);

  // namedWindow( "map_4_7", CV_WINDOW_AUTOSIZE );
  // imshow( "map_4_7", intMapNorm[4]);
#endif // INT_DEBUG

  return intMapNorm;
}

Mat IttiVSModel::obtainAggIntMap(Mat intensity, int depth) {
  vector<Mat> intPyramid;
  vector<Mat> intMapsNorm;
  Mat aggMapNorm;

  // Find the intensity pyramid
  intPyramid = getPyramid(intensity, depth - 1);
  intMapsNorm = intFeatureMaps(intPyramid);
  // find the aggregate feature map
  aggMapNorm = aggregateFeatureMap(intMapsNorm);
#ifdef INT_DEBUG
  namedWindow("Agg int map", CV_WINDOW_AUTOSIZE);
  imshow("Agg int map", aggMap);
#endif // INT_DEBUG
  return aggMapNorm;
}

Mat IttiVSModel::obtainAggColorMap(Mat rgbImage, Mat intensity, int depth) {
  double maxIntensity;
  int row, col;
  // Find the max intensity in the intensity image.
  minMaxLoc(intensity, NULL, &maxIntensity, NULL, NULL);

  vector<Mat> colorCh(4);
  vector<Mat> chanels;
  split(rgbImage, chanels);
  // Normalization of all channels
  // Pixels are normalized only if the corresponding pixel in the intensity
  // image is more than 10% of max intensity
  for (int i = 0; i < intensity.rows; i++) {
    for (int j = 0; j < intensity.cols; j++) {
      if (intensity.at<float>(i, j) > 0.1 * maxIntensity) {
        chanels[0].at<float>(i, j) = chanels[0].at<float>(i, j) / maxIntensity;
        chanels[1].at<float>(i, j) = chanels[1].at<float>(i, j) / maxIntensity;
        chanels[2].at<float>(i, j) = chanels[2].at<float>(i, j) / maxIntensity;
      } else {
        chanels[0].at<float>(i, j) = 0;
        chanels[1].at<float>(i, j) = 0;
        chanels[2].at<float>(i, j) = 0;
      }
    }
  }

  colorCh[0] = chanels[2] - (chanels[1] + chanels[0]) / 2; // R channel
  colorCh[1] = chanels[1] - (chanels[0] + chanels[2]) / 2; // G channel
  colorCh[2] = chanels[0] - (chanels[0] + chanels[1]) / 2; // B channel
  colorCh[3] = (chanels[2] + chanels[1]) / 2 -
               cv::abs(chanels[2] - chanels[1]) / 2 - chanels[0]; // Y channel
  threshold(colorCh[3], colorCh[3], 0, 255.0, THRESH_TOZERO);
  threshold(colorCh[2], colorCh[2], 0, 255.0, THRESH_TOZERO);
  threshold(colorCh[1], colorCh[1], 0, 255.0, THRESH_TOZERO);
  threshold(colorCh[0], colorCh[0], 0, 255.0, THRESH_TOZERO);

  vector<Mat> colorMapNorm =
      colorFeatureMap(colorCh, depth); // 12 maps, 0->5 RG maps, 6->11 BY maps
  // Add RG and BY maps at respective scales
  vector<Mat> RGBYMap(6);

  for (int i = 0; i < 6; i++) {
    cv::add(colorMapNorm[i], colorMapNorm[i + 6], RGBYMap[i]);
  }

  Mat aggMapNorm = aggregateFeatureMap(RGBYMap);

#ifdef COLOR_DEBUG
  Mat normImg;
  merge(chanels, normImg);
  namedWindow("norm image", CV_WINDOW_AUTOSIZE);
  imshow("norm image", normImg);

  namedWindow("R ch", CV_WINDOW_AUTOSIZE);
  imshow("R ch", colorCh[0]);

  namedWindow("G ch", CV_WINDOW_AUTOSIZE);
  imshow("G ch", colorCh[1]);

  namedWindow("B ch", CV_WINDOW_AUTOSIZE);
  imshow("B ch", colorCh[2]);

  namedWindow("Y ch", CV_WINDOW_AUTOSIZE);
  imshow("Y ch", colorCh[3]);

  namedWindow("Agg color map", CV_WINDOW_AUTOSIZE);
  imshow("Agg color map", aggMap);
#endif // COLOR_DEBUG
  return aggMapNorm;
}

Mat IttiVSModel::obtianAggOrientationMap(Mat intensity, int depth) {
  vector<Mat> gaborImg = getGaborImages(intensity, 5);

  vector<Mat> gaborPyr0(depth);
  vector<Mat> gaborPyr45(depth);
  vector<Mat> gaborPyr90(depth);
  vector<Mat> gaborPyr135(depth);

  vector<Mat> orMap0(6);
  vector<Mat> orMap45(6);
  vector<Mat> orMap90(6);
  vector<Mat> orMap135(6);

  // First find the gabor pyramid
  gaborPyr0 = getPyramid(gaborImg[0], depth - 1);
  gaborPyr45 = getPyramid(gaborImg[1], depth - 1);
  gaborPyr90 = getPyramid(gaborImg[2], depth - 1);
  gaborPyr135 = getPyramid(gaborImg[3], depth - 1);

  orMap0 = intFeatureMaps(gaborPyr0);
  orMap45 = intFeatureMaps(gaborPyr45);
  orMap90 = intFeatureMaps(gaborPyr90);
  orMap135 = intFeatureMaps(gaborPyr135);

  Mat aggMap0 = aggregateFeatureMap(orMap0);
  Mat aggMap45 = aggregateFeatureMap(orMap45);
  Mat aggMap90 = aggregateFeatureMap(orMap90);
  Mat aggMap135 = aggregateFeatureMap(orMap135);

  Mat tmpAgg1, tmpAgg2, orAggMap;
  cv::add(aggMap0, aggMap45, tmpAgg1);
  cv::add(aggMap90, aggMap135, tmpAgg2);

  cv::add(normanization(tmpAgg1), normanization(tmpAgg2), orAggMap);
#ifdef ORIENTATION_DEBUG
  namedWindow("aggOrMap", CV_WINDOW_AUTOSIZE);
  imshow("aggOrMap", orAggMap);
#endif // ORIENTATION_DEBUG
  return normanization(orAggMap);
}

void IttiVSModel::IttiVSModelInit() {
  min_img_height = 256;
  min_img_width = 256;
  py_depth = 9;
}

Mat IttiVSModel::getIttiSaliency(Mat img) {
  /*---------- small size image ----------*/
  int height, width;
  width = img.cols;
  height = img.rows;
  if (height < min_img_height || width < min_img_width) {
    std::cout << " Image is too small to satisfy model's request ["
              << min_img_height << "*" << min_img_width << "]" << std::endl;
    exit(1);
  }
  /*---------- convert to rgb image ----------*/
  Mat rgbImage;
  if (img.channels() == 4)
    cvtColor(img, rgbImage, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1)
    cvtColor(img, rgbImage, cv::COLOR_GRAY2BGR);
  else
    rgbImage = img;
  rgbImage.convertTo(rgbImage, CV_32FC3);
  /*---------- obtain intensity images ----------*/
  Mat intensity, red, green, blue;
  vector<Mat> rgb;
  split(rgbImage, rgb);
  red = rgb[2];
  green = rgb[1];
  blue = rgb[0];

  // yellow = (red + green)/2;
  // yellow = (red + green)/2 - (red-green)/2 -blue;
  // cvtColor(rgbImage, intensity, CV_BGR2GRAY);
  intensity = (red + green + blue) / 3;

  Mat intMapNorm, colorMapNorm, oriMapNorm;
  intMapNorm = obtainAggIntMap(intensity, py_depth);
  colorMapNorm = obtainAggColorMap(rgbImage, intensity, py_depth);
  oriMapNorm = obtianAggOrientationMap(intensity, py_depth);

  Mat finalMap = (intMapNorm + colorMapNorm + oriMapNorm) / 3;
  Mat tmpFinalMap;
  for (int up = 1; up <= 4; up++) {
    pyrUp(finalMap, tmpFinalMap);
    finalMap = tmpFinalMap;

    /*pyrUp(oriMapNorm, tmpFinalMap);
    oriMapNorm = tmpFinalMap;
    pyrUp(colorMapNorm, tmpFinalMap);
    colorMapNorm = tmpFinalMap;
    pyrUp(intMapNorm, tmpFinalMap);
    intMapNorm = tmpFinalMap;*/
  }
  /*namedWindow("intMapNorm", CV_WINDOW_AUTOSIZE);
  imshow("intMapNorm", intMapNorm);
  namedWindow("colorMapNorm", CV_WINDOW_AUTOSIZE);
  imshow("colorMapNorm", colorMapNorm);
  namedWindow("oriMapNorm", CV_WINDOW_AUTOSIZE);
  imshow("oriMapNorm", oriMapNorm);*/
  Mat finalMapNorm;
  double maxIntensity, minIntensity;
  cv::minMaxLoc(finalMap, NULL, &maxIntensity, NULL, NULL);
  finalMapNorm = finalMap / maxIntensity;
  Mat finalMapRes;
  resize(finalMapNorm, finalMapRes, cv::Size(width, height));
  cv::minMaxLoc(finalMapNorm, &minIntensity, &maxIntensity, NULL, NULL);
  // std::cout << "ori image size is: " << width << "*" << height << std::endl;
  // std::cout << "final map size is: " << finalMapNorm.cols << "*" <<
  // finalMapNorm.rows << std::endl;
  // std::cout << "max value is: " << maxIntensity << ", min value is: " <<
  // minIntensity << std::endl;
  return finalMapRes;
}
