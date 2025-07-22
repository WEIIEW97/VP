## for ethzasl_apriltag2
add_definitions("-fPIC -O3")
#library
include_directories(ethzasl_apriltag2/include)
file(GLOB SOURCE_FILES "ethzasl_apriltag2/src/*.cc")
add_library(ethzasl_apriltag2  ${SOURCE_FILES})
target_link_libraries(ethzasl_apriltag2 ${OpenCV_LIBS})
