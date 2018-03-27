find_package(JNI REQUIRED)

include_directories(${JNI_INCLUDE_DIRS}

add_library(iqa_jni SHARED
        
        )    

    target_link_libraries(iqa_jni iqa_algorithm ${JNI_LIBRARIES})    
