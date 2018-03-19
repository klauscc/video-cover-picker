#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "algorithms/ittivsmodel.h"
#include "algorithms/nriqa.h"

int main(int argc, char** argv) {

    std::string img_path = "../dataset/ori_imgs/2/7.jpg";
    Mat input = imread(img_path);
    if( !input.data ) {
        printf(" No data! -- Exiting the program \n");
        return -1;
    }
    //namedWindow("ori map", CV_WINDOW_AUTOSIZE);
    //imshow("ori map", input);

    IttiVSModel itti;
    itti.IttiVSModelInit();
    Mat vsiMap = itti.getIttiSaliency(input);
    //Mat finalMapImg = 255 * vsiMap;
    //finalMapImg.convertTo(finalMapImg, CV_8UC1);
    namedWindow("vsi map", CV_WINDOW_AUTOSIZE);
    imshow("vsi map", vsiMap);
    //Mat thImg;
    //inRange(vsiMap, 0.20, 1, thImg);
    //namedWindow("threshold map", CV_WINDOW_AUTOSIZE);
    //imshow("threshold map", thImg);

    /*vector<Mat> rgb;
    split(input, rgb);
    Mat merge_img;
    rgb[0] = rgb[0] & thImg;
    rgb[1] = rgb[1] & thImg;
    rgb[2] = rgb[2] & thImg;
    merge(rgb, merge_img);
    namedWindow("overlap map", CV_WINDOW_AUTOSIZE);
    imshow("overlap map", merge_img);*/
    nriqa nrIQA;
    nrIQA.nriqaInit();
    nrIQA.nriqaQualityIndex(input, vsiMap);
    waitKey(0);


}

