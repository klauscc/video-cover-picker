/*================================================================
*   God Bless You.
*
*   file name: iqa_pool.h
*   author: klaus
*   email: klaus.cheng@qq.com
*   created date: 2018/03/27
*   description:
*
================================================================*/

#pragma once

#include "iqa.h"
#include <sys/sysinfo.h>

/**
* @brief: given a directory, get the best image name and full path.
*
* @param: std::string imageDir. image directory.
*       : float &bestScore. the score of the best image in imageDir
*       : std::string &bestImageName.
*       : std::string &bestImagePath.
*
* @return: int
*/
int getBestImage(const std::string &imageDir, float &bestScore,
                 std::string &bestImageName, std::string &bestImagePath);

/**
* @brief: init iqa 
*
* @param: std::string modelDir. 
*
* @return: void  
*/
void   initIqa(std::string modelDir);
