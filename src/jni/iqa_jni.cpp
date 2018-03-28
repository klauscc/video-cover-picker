#include <iostream>

#include "algorithms/iqa_pool.h"
#include "com_resetta_iqa_iqa.h"

JNIEXPORT jstring JNICALL Java_com_resetta_iqa_Iqa_detectBestImageIndir(
    JNIEnv *env, jobject obj, jstring videoDir_) {
  const char *videoDir_cStr = env->GetStringUTFChars(videoDir_, 0);
  std::string videoDir(videoDir_cStr);

  std::string bestImagePath, bestImageName;
  float bestScore;
  getBestImage(videoDir, bestScore, bestImageName, bestImagePath);
  env->ReleaseStringUTFChars(videoDir_, videoDir_cStr);
  jstring jvideoDir = env->NewStringUTF(bestImagePath.c_str());
  return jvideoDir;
}

JNIEXPORT void JNICALL Java_com_resetta_iqa_Iqa_initIqa(JNIEnv *env,
                                                        jobject obj,
                                                        jstring modelDir_) {
  const char *modelDir_cStr = env->GetStringUTFChars(modelDir_, 0);
  std::string modelDir(modelDir_cStr);

  initIqa(modelDir);

  env->ReleaseStringUTFChars(modelDir_, modelDir_cStr);
}
