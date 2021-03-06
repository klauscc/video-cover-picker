#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "algorithms/ittivsmodel.h"
#include "algorithms/nriqa.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <fstream>
#include <dirent.h>

int main(int argc, char** argv) {

    std::string ori_img_dir = "../dataset/ori_imgs/";
    std::string qa_img_dir = "../dataset/qa_imgs/";
    int stop_dir = 74;
    for (int i = 1; i <= stop_dir; i++) {
        std::stringstream ss_qa_dir;
        ss_qa_dir << qa_img_dir << i;
        DIR *dp;
        struct dirent *ptr;
        if ((dp = opendir(ss_qa_dir.str().c_str())) != NULL) {
            while ((ptr = readdir(dp)) != NULL) {
                if(strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
                    continue;
                else if(ptr->d_type == 8) {
                    std::stringstream szFileName;
                    szFileName << qa_img_dir << i << "/" << ptr->d_name;
                    //std::cout << szFileName.str() << std::endl;
                    remove(szFileName.str().c_str());
                }
            }

            int isDel = remove(ss_qa_dir.str().c_str());
            if( !isDel ) {
               printf("delete dir:%s\n", ss_qa_dir.str().c_str());
            }else {
               printf("delete dir failed! error code : %s \n", isDel, ss_qa_dir.str().c_str());
            }
        }
        closedir(dp);

        int isCreate = mkdir(ss_qa_dir.str().c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        if( !isCreate ) {
           printf("create dir:%s\n", ss_qa_dir.str().c_str());
        }else {
           printf("create dir failed! error code : %s \n", isCreate, ss_qa_dir.str().c_str());
        }

        for (int j = 1; j <= 12; j++) {
            std::stringstream ss_img_path, ss_img_write;
            ss_img_path << ori_img_dir << i << "/" << j << ".jpg";
            ss_img_write << qa_img_dir << i << "/";
            //std::cout << ss_img_path.str() << std::endl;
            Mat input = imread(ss_img_path.str());
            IttiVSModel itti;
            itti.IttiVSModelInit();
            Mat vsiMap = itti.getIttiSaliency(input);
            nriqa nrIQA;
            nrIQA.nriqaInit();
            vector<float> iqa = nrIQA.nriqaQualityIndex(input, vsiMap);
            //std::cout << iqa[0] << std::endl;
            for (int k = 0; k < iqa.size(); k++)
                ss_img_write << (int) (100000*iqa[k]) << "_";
            ss_img_write << j << ".jpg";
            //std::cout << ss_img_write.str() << std::endl;
            imwrite(ss_img_write.str(), input);
        }

    }





    /*Mat input = imread(img_path);
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
    //namedWindow("vsi map", CV_WINDOW_AUTOSIZE);
    //imshow("vsi map", vsiMap);
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
    /*nriqa nrIQA;
    nrIQA.nriqaInit();
    nrIQA.nriqaQualityIndex(input, vsiMap);
    waitKey(0);*/


}

