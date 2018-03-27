set (CMAKE_CXX_FLAGS "-fPIC")

add_subdirectory(../third_party/dlib ../third_party/dlib)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARE_LINKER_FLAGS "${CMAKE_SHARE_LINKER_FLAGS} ${OpenMP_SHARE_LINKER_FLAGS}")
endif()

add_library(iqa_algorithm STATIC 
    iqa.cpp
    ittivsmodel.cpp 
    nriqa.cpp 
    faceModel.cpp
    brisque.cpp
    ../third_party/brisque/brisque.cpp
    ../third_party/brisque/libsvm/svm.cpp
    )
target_link_libraries(iqa_algorithm ${OpenCV_LIBS} dlib ${Boost_LIBRARIES})
