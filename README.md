### 1. Dependency

    1. Cmake.  > 3.1
    2. Opencv. 

### 2. Compile

```
mkdir build
cd build
cmake ../src
make -j
``` 
then multiple executable files will generate. in `build` dir, run `./demo_face` will assess images in `../dataset/ori_imgs` and produce result in `../dataset/qa_imgs`
