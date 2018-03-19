cmake_minimum_required (VERSION 3.1)
project (iqa)

set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_subdirectory(algorithms)

add_executable(demo_bak demo_bak20180301.cpp )
target_link_libraries(demo_bak iqa_algorithm)
 
add_executable(demo demo.cpp )
target_link_libraries(demo iqa_algorithm)

add_executable(demo_face demo_faceModel.cpp )
target_link_libraries(demo_face iqa_algorithm)
