#ifndef IQA
#define IQA 

#include <iostream>
#include "faceModel.h"
#include "brisque.h"
#include "ittivsmodel.h"
#include "nriqa.h"

class iqa
{
public:
    iqa (std::string allRangeFile = "../model/brisque/allrange", std::string allModel = "../model/brisque/allmodel");
    float computeScore(const std::string &imagePath);
    float getAllScore(const std::string imagePath, std::vector<float> &scores);
    int getBestImages(std::string imageDir, float &bestScore, std::string &bestImageName, std::string &bestImagePath); 
    virtual ~iqa ();

private:
    /* data */
    Brisque brisque;
    faceModel facemodel;
    IttiVSModel itti;
    nriqa nrIQA;
};

#endif /* ifndef IQA */
