#ifndef NRIQA_H
#define NRIQA_H
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdlib.h>
using namespace cv;
using namespace std;

class nriqa
{
public:
    void nriqaInit();
    vector<float> nriqaQualityIndex(Mat img, Mat vsi);

private:
    Mat cannyEdgeCalc(Mat im);
    Mat sobelFilter(Mat im);
    float calcMSSIM(Mat im1, Mat im2);
    float calcImgEntropy(Mat im);
    Mat calcImgNrss(Mat im1, Mat im2);
    Mat calcImgHitRatio(Mat im);
    Mat calcRatioPower(Mat edge, Mat power);
    int findSaArea(Mat im);

private:
    int min_img_height;
    int min_img_width;
    int area_w_size;
    int area_h_size;
    Mat perception_mask;
};

#endif // NRIQA_H
