# IMU Video Processing Solution

## Overview

This solution addresses the complex task of:
1. **Parsing IMU data** from a .txt file containing JSON records
2. **Extracting frames** from an H.265 video file using timestamp indices
3. **Matching timestamps** between video frames and IMU data
4. **Interpolating IMU values** (yaw, pitch, roll) for frames without exact matches
5. **Generating a comprehensive mapping** between frame indices and IMU data

## Problem Statement

The original challenge was to:
- Parse `20250731_135519_imu.txt` containing IMU sensor data
- Extract frames from `20250731_135519_main.h265` using timestamps from `20250731_135519_main.h265.index`
- Match video frame timestamps with IMU data timestamps
- Handle cases where timestamps don't exactly match (interpolation)
- Generate output with frame index as key and values: `[image_file, yaw, pitch, roll]`

## Solution Components

### 1. IMU Data Parser (`parse_imu_file`)
- **Input**: Text file with JSON records per line
- **Output**: Dictionary with timestamp as key
- **Data extracted**: yaw, pitch, roll, acceleration, gyroscope, quaternion
- **Error handling**: Skips invalid JSON lines gracefully

### 2. H.265 Index Parser (`parse_h265_index`)
- **Input**: Index file with one timestamp per line
- **Output**: List of frame timestamps
- **Format**: Each line contains a timestamp for video frame synchronization

### 3. Timestamp Matching & Interpolation (`find_closest_imu_data`)
- **Algorithm**: Finds closest IMU timestamp to frame timestamp
- **Interpolation**: Linear interpolation between two closest IMU readings
- **Fallback**: Uses closest match if interpolation not possible
- **Configurable**: Maximum time difference threshold for interpolation

### 4. Frame Extraction (`extract_frames_by_number`)
- **Method**: Uses ffmpeg with frame number selection instead of timestamp seeking
- **Reason**: H.265 files have seeking issues with large timestamp values
- **Output**: PNG images named `frame_XXXXXX.png`
- **Configuration**: Skips first 300 frames, extracts only 5 frames

### 5. Frame-IMU Mapping (`generate_frame_imu_mapping`)
- **Input**: Frame timestamps, IMU data, extracted frame paths
- **Output**: Dictionary mapping frame index to frame data
- **Structure**: `{frame_index: {image_file, yaw, pitch, roll, timestamp}}`

## File Structure

```
vp/
├── imu_align.py              # Single integrated solution
└── README_IMU_Video_Processing.md
```

## Usage

### Run the Complete Solution
```bash
python vp/imu_align.py
```

## Configuration

The script is configured with the following parameters:
- **Start frame**: 300 (skip first 300 frames)
- **Number of frames**: 5 (only process 5 frames)
- **Input files**: Automatically configured for the specific dataset
- **Output directory**: `/home/william/extdisk/data/boximu-rgb/imu-box/extracted_frames`

## Output Format

### Frame-IMU Mapping JSON
```json
{
  "0": {
    "image_file": "frame_00300.png",
    "yaw": -104.76,
    "pitch": 7.38,
    "roll": 1.19,
    "timestamp": 19728945
  },
  "1": {
    "image_file": "frame_00301.png",
    "yaw": -104.76,
    "pitch": 7.38,
    "roll": 1.20,
    "timestamp": 19762231
  }
}
```

### Extracted Images
- **Naming convention**: `frame_XXXXXX.png` (6-digit zero-padded)
- **Format**: PNG with high quality (q:v 2)
- **Resolution**: 1920x1080 (original video resolution)
- **Location**: Specified output directory
- **Quantity**: Only 5 frames (frames 300-304)

## Key Features

✅ **Robust parsing**: Handles malformed JSON and missing data  
✅ **Smart interpolation**: Linear interpolation between IMU readings  
✅ **Efficient extraction**: Frame-by-frame extraction using ffmpeg  
✅ **Specific configuration**: Skips first 300 frames, processes only 5 frames  
✅ **Clean output**: Image file values contain only filenames, not full paths  
✅ **Error handling**: Graceful fallbacks and detailed error reporting  

## Technical Details

### Timestamp Handling
- **IMU timestamps**: Large integer values (e.g., 19728945)
- **Video timestamps**: Same scale as IMU timestamps
- **Interpolation threshold**: Configurable (default: 100,000 units)
- **Time base**: Not in seconds - using raw timestamp units

### Video Processing
- **Codec**: H.265/HEVC
- **Frame rate**: 25 fps
- **Resolution**: 1920x1080
- **Extraction method**: Frame number selection (`-vf select=eq(n\,X)`)
- **Quality**: High quality PNG output
- **Frame selection**: Frames 300-304 (skipping first 300)

### Interpolation Algorithm
```python
# Linear interpolation between two IMU readings
alpha = (frame_timestamp - closest_timestamp) / (next_timestamp - closest_timestamp)
yaw = closest_data['yaw'] + alpha * (next_data['yaw'] - closest_data['yaw'])
pitch = closest_data['pitch'] + alpha * (next_data['pitch'] - closest_data['pitch'])
roll = closest_data['roll'] + alpha * (next_data['roll'] - closest_data['roll'])
```

## Performance Considerations

### Processing Scope
- **IMU data**: Loads all 29,683+ records for comprehensive matching
- **Video frames**: Processes only 5 frames (300-304)
- **Output**: Minimal file generation for focused analysis

### Memory Usage
- **IMU data**: Loaded entirely into memory for fast lookup
- **Frame paths**: Stored in dictionary for quick access
- **Output**: Streamed to disk to avoid memory buildup

## Troubleshooting

### Common Issues

#### 1. ffmpeg Seeking Errors
- **Symptom**: "could not seek to position" errors
- **Solution**: Script uses frame number extraction instead of timestamp seeking
- **Status**: Already implemented in the solution

#### 2. Frame Range Issues
- **Symptom**: "Requested frames exceed available frames"
- **Solution**: Script automatically adjusts frame count if needed
- **Alternative**: Modify `start_frame` and `num_frames` variables

#### 3. File Path Issues
- **Symptom**: Missing input files
- **Solution**: Verify file paths in the configuration section
- **Files required**: H.265 video, index file, and IMU data file

### Debug Mode
```python
# Enable detailed error reporting
import traceback
try:
    # Your processing code
    pass
except Exception as e:
    traceback.print_exc()
```

## Customization

### Modify Frame Selection
```python
# In the main() function, change these values:
start_frame = 300  # Change to skip different number of frames
num_frames = 5     # Change to process different number of frames
```

### Change Output Directory
```python
# Modify the output directory path:
output_dir = "/path/to/your/output/directory"
```

### Adjust Interpolation Threshold
```python
# In find_closest_imu_data function call:
yaw, pitch, roll = find_closest_imu_data(timestamp, imu_data, max_time_diff=50000)
```

## Conclusion

This solution successfully addresses the specific requirements:
- **Parses** IMU data from text files
- **Skips** first 300 frames as requested
- **Extracts** only 5 frames for focused analysis
- **Generates** clean output with filenames only (no full paths)
- **Provides** comprehensive frame-IMU mapping with interpolation

The single integrated file (`imu_align.py`) contains all necessary functionality without redundancy, making it easy to maintain and modify for future needs.

---

*For questions or issues, refer to the script documentation or contact the development team.*
