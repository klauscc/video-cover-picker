cmake_minimum_required (VERSION 3.1)
project (iqa)

set (CMAKE_CXX_STANDARD 11)
#set(Boost_USE_STATIC_LIBS OFF) #set(Boost_USE_MULTITHREADED OFF) #set(Boost_USE_STATIC_RUNTIME OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem)

message(STATUS "Boost include dir:" ${Boost_INCLUDE_DIRS} )
message(STATUS "Boost libs dir:" ${Boost_LIBRARIES} )

include_directories(${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
    ./third_party/dlib
    ./third_party/)

add_subdirectory(algorithms)

#add_executable(demo_bak demo_bak20180301.cpp )
#target_link_libraries(demo_bak iqa_algorithm)
 
add_executable(demo demo.cpp )
target_link_libraries(demo iqa_algorithm)

add_executable(demo_face demo_faceModel.cpp )
target_link_libraries(demo_face iqa_algorithm)

add_executable(demo_singleColor demo_singleColorImage.cpp)
target_link_libraries(demo_singleColor iqa_algorithm ${Boost_LIBRARIES})

add_executable(demo_video1000 demo_video1000.cpp)
target_link_libraries(demo_video1000 iqa_algorithm ${Boost_LIBRARIES})
