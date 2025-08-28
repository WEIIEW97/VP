cmake_minimum_required(VERSION 3.22)
project(vp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
set(GFLAGS_USE_TARGET_NAMESPACE ON) # for unknown reason have to add this shit
endif()

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

set(VP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ethzasl_apriltag2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd/ethzasl_apriltag2/include)

set(PROJECT_HOME ${CMAKE_SOURCE_DIR})
file(TO_CMAKE_PATH "${PROJECT_HOME}" PROJECT_HOME_ESCAPED)
add_compile_definitions(PROJECT_HOME="${PROJECT_HOME_ESCAPED}")

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp) 

# do not include aprilgrid2.cpp, stitch.cpp and aprilgrid2.h, stitch.h
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/stitch.cpp)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/stitch.h)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/aprilgrid2.cpp)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/aprilgrid2.h)


# add_subdirectory(3rd)
add_subdirectory(lib)

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
        # ${ethzasl_apriltag2_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        nlohmann_json::nlohmann_json
        fmt::fmt
        ${OpenCV_LIBS}
        # ethzasl_apriltag2
        ${CERES_LIBRARIES}
)

add_executable(
        test_recalib
        ${sources}
        test/test_recalib.cpp
)

target_link_libraries(
        test_recalib
        PRIVATE
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        nlohmann_json::nlohmann_json
        fmt::fmt
        # ethzasl_apriltang2
        )

target_include_directories(
        test_recalib
        PRIVATE
        ${VP_INCLUDE_DIR}
        ${fmt_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
        ${OpenCV_INCLUDE_DIRS}
        # ${ethzasl_apriltag2_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
)

add_executable(
        test_lib_recalib
        test/test_lib_recalib.cpp
)

target_link_libraries(
        test_lib_recalib
        PRIVATE
        recalib
        )

target_include_directories(
        test_lib_recalib
        PRIVATE
        ${CMAKE_SOURCE_DIR}/lib
)


set(recalib_release_name "recalib_vp")

add_executable(
        ${recalib_release_name}
        src/recalib.cpp
        tool_recalib.cpp
)

target_link_libraries(
        ${recalib_release_name}
        PRIVATE
        ${OpenCV_LIBS}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        nlohmann_json::nlohmann_json
        # ethzasl_apriltag2
        )

target_include_directories(
        ${recalib_release_name}
        PRIVATE
        ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${VP_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
)