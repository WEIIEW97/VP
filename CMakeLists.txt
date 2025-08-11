cmake_minimum_required(VERSION 3.22)
project(vp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

set(VP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ethzasl_apriltag2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd/ethzasl_apriltag2/include)

set(PROJECT_HOME ${CMAKE_SOURCE_DIR})
file(TO_CMAKE_PATH "${PROJECT_HOME}" PROJECT_HOME_ESCAPED)
add_compile_definitions(PROJECT_HOME="${PROJECT_HOME_ESCAPED}")

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_subdirectory(3rd)

add_executable(
        ${PROJECT_NAME}
        ${sources}
#         main.cpp
        test/test_est_roll.cpp
)

if(MSVC)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)
        set_source_files_properties(${sources} PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${VP_INCLUDE_DIR}
        ${fmt_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${ethzasl_apriltag2_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        nlohmann_json::nlohmann_json
        fmt::fmt
        ${OpenCV_LIBS}
        ethzasl_apriltag2
        ${CERES_LIBRARIES}
)

add_executable(
        recalib_test
        src/recalib.cpp
        test/test_recalib.cpp
)

target_link_libraries(
        recalib_test
        PRIVATE
        ${OpenCV_LIBS}
        )

target_include_directories(
        recalib_test
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${VP_INCLUDE_DIR}
)