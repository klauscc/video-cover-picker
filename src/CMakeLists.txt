cmake_minimum_required (VERSION 3.1)
project (iqa_shr)
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set (CMAKE_CXX_FLAGS "-fPIC")
add_subdirectory(third_party/dlib)

include_directories(${OpenCV_INCLUDE_DIRS} ./third_party/dlib)
message(STATUS "Opencv include dir:" ${OpenCV_INCLUDE_DIRS} )
message(STATUS "Opencv libs dir:" ${OpenCV_LIBS} )

add_library(iqa_algorithm SHARED ittivsmodel.cpp nriqa.cpp faceModel.cpp)
target_link_libraries(iqa_algorithm ${OpenCV_LIBS} dlib)
 
add_executable(demo_bak demo_bak20180301.cpp )
target_link_libraries(demo_bak iqa_algorithm)
 
add_executable(demo demo.cpp )
target_link_libraries(demo iqa_algorithm)

add_executable(demo_face demo_faceModel.cpp )
target_link_libraries(demo_face iqa_algorithm)
