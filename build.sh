#!/usr/bin/sh

#================================================================
#   God Bless You. 
#   
#   file name: build.sh
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/03/26
#   description: 
#
#================================================================

mkdir -p build
cd build
cmake -DDLIB_USE_BLAS=OFF -DDLIB_USE_LAPACK=OFF -DDLIB_USE_CUDA=OFF -DDLIB_USE_MKL_FFT=OFF ../src/
make -j
make install
echo "Test java..."
java -Djava.library.path="./jni" -cp java/Demo.jar com.resetta.iqa.Iqa
