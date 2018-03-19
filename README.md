### 1. Dependencies

    1. Cmake.  > 3.1
    2. Opencv. 

### 2. Compile

```
mkdir build
cd build
cmake ../src
make -j
``` 
### 3. Run

make sure data in `./dataset`

#### run `./demo_face` in dir `./build`. 
this  command will assess images in `./dataset/ori_imgs` and produce result in `./dataset/qa_imgs`
the images named with `{finalScore}_{faceAreaScore}_{facePositionScore}_{brisqueScore}_idx.jpg`. The values are convert to int by multiply 1000, e.g. 0.23 converted to 230

```
# note the brisque is normalized to [0,1]. So higher score means better quality
brisqueScore = 1 - stantardBrisqueScore / 100
finalScore = faceAreaScore * facePositionScore * 2 + brisqueScore.
```
