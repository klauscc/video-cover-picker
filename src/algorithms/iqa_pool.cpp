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

#include "iqa_pool.h"
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

static const int POOL_SIZE = max(get_nprocs() / MAX_IMAGE_NUM, 1);
static std::vector<iqa> iqaVec(POOL_SIZE) ;
static std::vector<bool> iqaFree(POOL_SIZE, true);
static std::mutex iqaGetterMu;
static std::condition_variable iqaGetterCv;
static bool newFree = false;
static int newFreeIqaId = -1;

extern int getIqa();
extern void freeIqa(int iqaId);

void initIqa(std::string modelDir) {
  iqaVec.reserve(POOL_SIZE);
  for (int i = 0; i < POOL_SIZE; ++i) {
    iqa iqa_(modelDir);
    iqaVec.push_back(iqa_);
  }
}

int getBestImage(const std::string &imageDir, float &bestScore,
                 std::string &bestImageName, std::string &bestImagePath) {
  int iqaId = getIqa();
  int res = iqaVec[iqaId].getBestImage(imageDir, bestScore, bestImageName,
                                       bestImagePath);
  freeIqa(iqaId);
  return res;
}

/**
* @brief: get Iqa
*
* @param:
*
* @return: int
*/
int getIqa() {
  int iqaId = -1;
  while (1) {
    {
      std::lock_guard<std::mutex> lkGuard(iqaGetterMu);
      for (int i = 0; i < POOL_SIZE; ++i) {
        if (iqaFree[i] == true) {
          iqaFree[i] = false;
          iqaId = i;
          return iqaId;
        }
      }
    }
    std::unique_lock<std::mutex> lk(iqaGetterMu);
    iqaGetterCv.wait(lk, [] { return newFree; });
    newFree = false;
    lk.unlock();
  }
}

/**
* @brief: freeIqa
*
* @param: int iqaId
*
* @return: void
*/
void freeIqa(int iqaId) {
  {
    std::lock_guard<std::mutex> lkGuard(iqaGetterMu);
    iqaFree[iqaId] = true;
    newFree = true;
    newFreeIqaId = iqaId;
  }
  iqaGetterCv.notify_one();
}
