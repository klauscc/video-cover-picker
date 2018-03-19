cmake_minimum_required(VERSION 2.8)
project(brisquefeat )

find_package( OpenCV REQUIRED )
include_directories ( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "Opencv include dir:" ${OpenCV_INCLUDE_DIRS} )
message(STATUS "Opencv libs dir:" ${OpenCV_LIBS} )
link_directories ("/usr/lib")
add_definitions(-DGCC_COMPILER=1)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_CXX_FLAGS -fpermissive)

add_executable( brisquequality main.cpp trainModel.cpp  computescore.cpp brisque.cpp libsvm/svm.cpp)

target_link_libraries( brisquequality  ${OpenCV_LIBS} opencv_videostab)
