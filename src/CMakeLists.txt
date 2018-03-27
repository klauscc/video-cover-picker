cmake_minimum_required (VERSION 3.1)
project (iqa)

set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED filesystem)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARE_LINKER_FLAGS "${CMAKE_SHARE_LINKER_FLAGS} ${OpenMP_SHARE_LINKER_FLAGS}")
endif()

message(STATUS "Boost include dir:" ${Boost_INCLUDE_DIRS} )
message(STATUS "Boost libs dir:" ${Boost_LIBRARIES} )

include_directories(${OpenCV_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
    ./third_party/dlib
    ./third_party/)

add_subdirectory(algorithms)

add_executable(demo_video1000_003 demo_video1000_003.cpp)
target_link_libraries(demo_video1000_003 iqa_algorithm ${Boost_LIBRARIES})
