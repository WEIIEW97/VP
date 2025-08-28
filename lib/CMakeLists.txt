# CMakeLists.txt for lib directory
# Builds the e_recalib library

# Create e_recalib library
add_library(recalib
    e_recalib.cpp
    ${CMAKE_SOURCE_DIR}/src/recalib.cpp
)

# Set include directories for e_recalib library
target_include_directories(recalib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(recalib
    PUBLIC
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

# Set compile definitions
target_compile_definitions(recalib
    PRIVATE
        PROJECT_HOME="${PROJECT_HOME_ESCAPED}"
        BUILDING_E_RECALIB
)

# Set C++ standard
set_target_properties(recalib PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Export the library for use by parent CMakeLists.txt
set_target_properties(recalib PROPERTIES
    EXPORT_NAME recalib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

