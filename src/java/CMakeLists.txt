find_package(Java REQUIRED)
include(UseJava)

set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8")
add_jar(Demo ./src/com/resetta/iqa/Iqa.java 
        )

get_target_property(_jarFile Demo JAR_FILE)
get_target_property(_classDir Demo CLASSDIR)

message(STATUS "Jar file ${_jarFile}")
message(STATUS "Class compiled to ${_classDir}")    

