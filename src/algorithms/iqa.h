#ifndef IQA
#define IQA 

#include <iostream>
#include "faceModel.h"
#include "brisque.h"

class iqa
{
public:
    iqa (std::string allRangeFile = "../model/brisque/allrange", std::string allModel = "../model/brisque/allmodel");
    float computeScore(const std::string &imagePath);
    int getBestImages(std::string imageDir, float &bestScore, std::string &bestImageName, std::string &bestImagePath); 
    virtual ~iqa ();

private:
    /* data */
    Brisque brisque;
    faceModel facemodel;
};

#endif /* ifndef IQA */
