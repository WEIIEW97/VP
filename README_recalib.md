# Recalibration Tool Manual

## Overview

The Recalibration Tool (`recalib_vp`) is a C++ application designed for camera calibration and image recalibration using chessboard patterns. It automatically detects chessboard corners, calculates rotation angles (pitch, yaw, roll), and provides recalibration capabilities for camera systems.

## Features

- **Automatic Chessboard Detection**: Detects chessboard patterns in images or YUV files
- **Camera Calibration**: Calculates intrinsic parameters and distortion coefficients
- **Rotation Angle Estimation**: Determines pitch, yaw, and roll angles from detected patterns
- **Image Recalibration**: Applies rotation corrections to images
- **Multiple Input Formats**: Supports YUV420 (I420) and common image formats (PNG, JPG)
- **Flexible Pattern Sizes**: Configurable chessboard pattern dimensions
- **Command-Line Interface**: Easy-to-use CLI with comprehensive options

## Prerequisites

### System Requirements

- Linux/Unix operating system
- C++20 compatible compiler (GCC 10+ or Clang 12+)
- CMake 3.22 or higher

### Dependencies

- **OpenCV 4.x**: Computer vision library
- **Eigen3**: Linear algebra library
- **nlohmann/json**: JSON parsing library
- **Boost**: Program options library
- **Ceres**: Optimization library

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vp
```

### 2. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libopencv-dev libeigen3-dev nlohmann-json3-dev libboost-program-options-dev libceres-dev

# Or use conda
conda install opencv eigen nlohmann-json boost
```

### 3. Build the Project

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The recalibration tool will be built as `recalib_vp` in the build directory.

## Usage

### Basic Command Structure

```bash
./recalib_vp [options]
```

### Required Parameters

- `--input, -i`: Path to input image file (.yuv, .png, .jpg, etc.)
- `--intrinsic, -k`: Path to intrinsic calibration JSON file

### Optional Parameters

- `--height, -h`: Image height (default: 1080)
- `--width, -w`: Image width (default: 1920)
- `--pattern_rows, -r`: Number of inner corners per chessboard row (default: 3)
- `--pattern_cols, -c`: Number of inner corners per chessboard column (default: 6)
- `--square_size, -s`: Size of chessboard square in meters (default: 0.025)
- `--help`: Display help message

### Examples

#### 1. Basic Usage with YUV File

```bash
./recalib_vp \
  --input /path/to/image.yuv \
  --intrinsic /path/to/intrinsics.json \
  --height 1080 \
  --width 1920
```

#### 2. Custom Chessboard Pattern

```bash
./recalib_vp \
  --input /path/to/image.png \
  --intrinsic /path/to/intrinsics.json \
  --pattern_rows 5 \
  --pattern_cols 8 \
  --square_size 0.03
```

#### 3. Different Image Dimensions

```bash
./recalib_vp \
  --input /path/to/image.jpg \
  --intrinsic /path/to/intrinsics.json \
  --height 720 \
  --width 1280
```

## Input File Formats

### Image Files

- **Supported formats**: PNG, JPG, JPEG, BMP, TIFF
- **Color spaces**: RGB, BGR, Grayscale
- **Bit depth**: 8-bit, 16-bit

### YUV Files

- **Format**: YUV420 (I420) planar format
- **Layout**: Y plane followed by U and V planes
- **File size**: Must be exactly `width × height × 3/2` bytes

### Intrinsic JSON Format

The intrinsic calibration file should contain:

```json
{
  "cam_intrinsic": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "cam_distcoeffs": [k1, k2, p1, p2, k3, k4, k5, k6]
}
```

Where:

- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point coordinates
- `k1-k6`: Distortion coefficients
- `p1, p2`: Tangential distortion coefficients

## Output

### Success Case

```
Rotation angle (degrees): -6.71674
```

### Error Cases

```
Failed to detect rotation angles, please check your conditions manually!
Failed to run calibration!
```

## Chessboard Pattern Requirements

### Pattern Specifications

- **Material**: High contrast (black/white) pattern
- **Shape**: Regular grid of squares
- **Size**: Configurable square dimensions
- **Corners**: Inner corners must be clearly visible

### Detection Tips

- Ensure good lighting conditions
- Avoid shadows and reflections
- Keep the pattern flat and undistorted
- Ensure the entire pattern is visible in the image

## Advanced Usage

### Batch Processing

For processing multiple images, you can create a shell script:

```bash
#!/bin/bash
for image in /path/to/images/*.png; do
  ./recalib_vp \
    --input "$image" \
    --intrinsic /path/to/intrinsics.json \
    --pattern_rows 3 \
    --pattern_cols 6
done
```

### Integration with Other Tools

The tool can be integrated into larger calibration pipelines:

```bash
# Example pipeline
./recalib_vp --input image.yuv --intrinsic calib.json > angle.txt
rotation_angle=$(cat angle.txt | grep "Rotation angle" | awk '{print $4}')
# Use rotation_angle in subsequent processing
```

## Troubleshooting

### Common Issues

#### 1. "Failed to detect rotation angles"

- **Cause**: Chessboard pattern not detected
- **Solution**:
  - Check pattern visibility and lighting
  - Verify pattern size parameters
  - Ensure image quality and contrast

#### 2. "Failed to run calibration!"

- **Cause**: Internal calibration error
- **Solution**:
  - Verify intrinsic parameters
  - Check image dimensions
  - Ensure sufficient pattern coverage

#### 3. YUV File Size Mismatch

- **Cause**: Incorrect image dimensions
- **Solution**: Verify height and width parameters match the actual YUV file

### Debug Mode

For debugging, you can modify the source code to add verbose output:

```cpp
// Add debug prints in the main function
std::cout << "Processing: " << input_path << std::endl;
std::cout << "Pattern size: " << pattern_size << std::endl;
```

## API Reference

### ChessboardCalibrator Class

#### Constructor

```cpp
ChessboardCalibrator(const cv::Matx33d& K, const cv::Vec<double, 8>& dist)
```

#### Methods

- `detect()`: Detect chessboard and calculate angles
- `get_warped_image()`: Get corrected image
- `get_rgb_image()`: Get original RGB image

#### CalibResult Structure

```cpp
struct CalibResult {
    bool success;
    cv::Vec3d angle_degrees;  // [pitch, yaw, roll]
    std::vector<cv::Point2f> corners;
    cv::Size pattern_size;
};
```

## Performance Considerations

### Optimization Tips

- Use appropriate image resolution for your needs
- Consider downsampling large images for faster processing
- Optimize chessboard pattern size for your use case

### Memory Usage

- YUV files require `width × height × 3/2` bytes of memory
- Processing large images may require significant RAM
- Consider processing images in batches for memory-constrained systems

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow the existing C++ coding conventions
- Use meaningful variable names
- Add comments for complex logic
- Ensure proper error handling

## License

Copyright (c) 2022-2025, William Wei. All rights reserved.

Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the source code
3. Create an issue in the repository
4. Contact the development team

## Version History

- **v1.0**: Initial release with basic calibration functionality
- **v1.1**: Added YUV file support and improved error handling
- **v1.2**: Enhanced pattern detection and rotation calculation

---

*This manual covers the recalibration tool based on `tool_recalib.cpp`. For the latest updates and additional features, refer to the source code and project documentation.*

