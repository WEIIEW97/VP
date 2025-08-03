# VP: A Computer Vision Perception Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-orange)
[![Eigen](https://img.shields.io/badge/Eigen-3.4.0%2B-green)](https://eigen.tuxfamily.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue)](https://opencv.org)

This project provides a dual C++ and Python framework for advanced computer vision tasks, focusing on camera pose estimation, image transformation, and stitching. It is designed for applications in robotics and autonomous systems, featuring robust algorithms for AprilTag-based localization, Inverse Perspective Mapping (IPM), and vanishing point analysis.

## Core Features

- **AprilTag Detection**: Utilizes the `ethzasl_apriltag2` library for detecting fiducial markers and `aprilgrid` for grid-based detection, enabling precise localization.
- **Inverse Perspective Mapping (IPM)**: Transforms images from a perspective view to a top-down, bird's-eye view.
- **Image Stitching**: Merges multiple BEV (Bird's-Eye View) images to create a single panoramic view of an area.
- **Pose Estimation**: Estimates camera orientation (yaw, pitch, roll) using vanishing points and AprilTag grids.
- **Dual Language Implementation**: Offers high-performance C++ for core operations and flexible Python scripts for prototyping, calibration, and visualization.
- **Camera Calibration**: Includes tools and examples for determining camera intrinsic and extrinsic parameters.

## Dependencies

### C++
- **CMake** (>= 3.22)
- **C++ Compiler** (C++20 standard)
- **Eigen3**
- **OpenCV**
- **nlohmann_json**
- **fmt**

### Python
- **Python** (>= 3.8)
- **numpy**
- **opencv-python**
- **matplotlib**
- **scipy**

You can install the Python dependencies using pip:
```bash
pip install numpy opencv-python matplotlib scipy
```

## Build Instructions (C++)

To build the C++ application, use the following standard CMake workflow:

```bash
# Create a build directory
mkdir build
cd build

# Configure the project
cmake ..

# Compile the project
make -j$(nproc)
```
The main executable, `vp`, will be located in the `build` directory.

## Usage

### C++ Executable
After building, you can run the main C++ application from the `build` directory:
```bash
./vp [arguments]
```
Refer to the `main.cpp` file for specific command-line arguments and functionality.

### Python Scripts
The Python scripts in the `vp/` directory are designed for individual tasks such as stitching, calibration, and IPM transformation.

**Example: Running the BEV stitching script:**
```bash
python3 vp/stitch_fwd.py
```
This script will process sample images from the `data/zed_360` directory, generate BEV images, stitch them together, and display the result using `matplotlib`.

## Project Structure

```
.
├── 3rd/                # Third-party libraries (ethzasl_apriltag2)
├── CMakeLists.txt      # Main CMake build script
├── data/               # Sample images, calibration files, and test data
├── include/            # C++ header files
├── main.cpp            # Main entry point for the C++ application
├── misc/               # Miscellaneous scripts for experiments (e.g., recalibration)
├── src/                # C++ source code implementation
└── vp/                 # Python scripts and modules
    ├── aprilgrid/      # Python-based AprilTag grid detection
    ├── ipm.py          # Inverse Perspective Mapping implementation
    ├── stitch_fwd.py   # Forward stitching for BEV images
    └── ...             # Other Python utility and experiment scripts
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.