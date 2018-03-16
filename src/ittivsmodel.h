#ifndef ITTIVSMODEL_H
#define ITTIVSMODEL_H

#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdlib.h>
using namespace cv;
using namespace std;
class IttiVSModel
{
public:
    // IttiVSModel();
    void IttiVSModelInit();
    Mat getIttiSaliency(Mat img);

private:
    vector<Mat> getPyramid(Mat image, int level);
    vector<Mat> getGaborImages(Mat img, int kernelSize);
    vector<Mat> colorFeatureMap(vector<Mat> rgbyCh, int depth);
    Mat aggregateFeatureMap(vector<Mat> featureMap);
    vector<Mat> intFeatureMaps(vector<Mat> intPyr);
    Mat obtainAggIntMap(Mat intensity, int depth);
    Mat obtainAggColorMap(Mat rgbImage, Mat intensity, int depth);
    Mat obtianAggOrientationMap(Mat intensity, int depth);
    Mat normanization(Mat src);

private:
    int min_img_height;
    int min_img_width;
    int py_depth;
};

#endif // ITTIVSMODEL_H
