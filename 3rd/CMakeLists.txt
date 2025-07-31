## for ethzasl_apriltag2
add_definitions("-fPIC -O3")
#library
include_directories(ethzasl_apriltag2/include)
find_package(Eigen3)
find_package(OpenCV)
file(GLOB SOURCE_FILES "ethzasl_apriltag2/src/*.cc")
add_library(ethzasl_apriltag2  ${SOURCE_FILES})
target_include_directories(
        ethzasl_apriltag2
        PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ethzasl_apriltag2 ${OpenCV_LIBS})
