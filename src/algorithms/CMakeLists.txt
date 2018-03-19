include_directories(${OpenCV_INCLUDE_DIRS} ./third_party/dlib)
message(STATUS "Opencv include dir:" ${OpenCV_INCLUDE_DIRS} )
message(STATUS "Opencv libs dir:" ${OpenCV_LIBS} )

set (CMAKE_CXX_STANDARD 11)

set (CMAKE_CXX_FLAGS "-fPIC")
add_subdirectory(../third_party/dlib ../third_party/dlib)

add_library(iqa_algorithm SHARED ittivsmodel.cpp nriqa.cpp faceModel.cpp)
target_link_libraries(iqa_algorithm ${OpenCV_LIBS} dlib)
