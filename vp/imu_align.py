"""
Multi-Group IMU Data Parser with Video Frame Extraction

This script automatically detects and processes all available data groups in the directory,
parsing IMU data and extracting frames from H.265 videos with timestamp matching.
"""

import json
import os
import subprocess
import numpy as np
import glob
from typing import Dict, List, Any, Tuple
from pathlib import Path


def parse_imu_file(file_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Parse IMU data from a .txt file.
    
    Args:
        file_path: Path to the IMU data file
        
    Returns:
        Dictionary with timestamp as key and parsed data as value
    """
    imu_data = {}
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON from each line
                data = json.loads(line)
                
                # Extract timestamp
                timestamp = data.get('timestamp')
                if timestamp is None:
                    continue
                
                # Extract required fields
                parsed_data = {
                    'timestamp': timestamp,
                    'yaw': data.get('yaw'),
                    'pitch': data.get('pitch'),
                    'roll': data.get('roll'),
                    'raw_data': {
                        'acceleration': {
                            'x': data.get('Acce', {}).get('x'),
                            'y': data.get('Acce', {}).get('y'),
                            'z': data.get('Acce', {}).get('z')
                        },
                        'gyroscope': {
                            'x': data.get('Gyro', {}).get('x'),
                            'y': data.get('Gyro', {}).get('y'),
                            'z': data.get('Gyro', {}).get('z')
                        },
                        'gyro_bias': {
                            'x': data.get('Gbias', {}).get('x'),
                            'y': data.get('Gbias', {}).get('y'),
                            'z': data.get('Gbias', {}).get('z')
                        }
                    },
                    'quaternion': data.get('quaternion', [])
                }
                
                imu_data[timestamp] = parsed_data
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    return imu_data


def parse_h265_index(index_file_path: str) -> List[int]:
    """
    Parse H.265 index file to extract frame timestamps.
    
    Args:
        index_file_path: Path to the .h265.index file
        
    Returns:
        List of timestamps for each frame
    """
    timestamps = []
    
    with open(index_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    timestamp = int(line)
                    timestamps.append(timestamp)
                except ValueError:
                    continue
    
    return timestamps


def find_closest_imu_data(frame_timestamp: int, imu_data: Dict[int, Dict[str, Any]], 
                          max_time_diff: int = 100000) -> Tuple[float, float, float]:
    """
    Find the closest IMU data for a given frame timestamp and interpolate if needed.
    
    Args:
        frame_timestamp: Frame timestamp
        imu_data: Dictionary of IMU data indexed by timestamp
        max_time_diff: Maximum time difference for interpolation (in timestamp units)
        
    Returns:
        Tuple of (yaw, pitch, roll) - interpolated or copied from closest match
    """
    imu_timestamps = sorted(imu_data.keys())
    
    if not imu_timestamps:
        return 0.0, 0.0, 0.0
    
    # Find the closest timestamp
    closest_idx = np.argmin(np.abs(np.array(imu_timestamps) - frame_timestamp))
    closest_timestamp = imu_timestamps[closest_idx]
    time_diff = abs(closest_timestamp - frame_timestamp)
    
    if time_diff <= max_time_diff:
        # Use the closest match directly
        closest_data = imu_data[closest_timestamp]
        return closest_data['yaw'], closest_data['pitch'], closest_data['roll']
    
    # Need to interpolate between two closest timestamps
    if closest_timestamp < frame_timestamp:
        # Frame timestamp is after closest IMU timestamp
        if closest_idx + 1 < len(imu_timestamps):
            next_timestamp = imu_timestamps[closest_idx + 1]
            next_data = imu_data[next_timestamp]
            closest_data = imu_data[closest_timestamp]
            
            # Linear interpolation
            alpha = (frame_timestamp - closest_timestamp) / (next_timestamp - closest_timestamp)
            yaw = closest_data['yaw'] + alpha * (next_data['yaw'] - closest_data['yaw'])
            pitch = closest_data['pitch'] + alpha * (next_data['pitch'] - closest_data['pitch'])
            roll = closest_data['roll'] + alpha * (next_data['roll'] - closest_data['roll'])
            
            return yaw, pitch, roll
        else:
            # Use the closest match if no next timestamp available
            closest_data = imu_data[closest_timestamp]
            return closest_data['yaw'], closest_data['pitch'], closest_data['roll']
    else:
        # Frame timestamp is before closest IMU timestamp
        if closest_idx > 0:
            prev_timestamp = imu_timestamps[closest_idx - 1]
            prev_data = imu_data[prev_timestamp]
            closest_data = imu_data[closest_timestamp]
            
            # Linear interpolation
            alpha = (frame_timestamp - prev_timestamp) / (closest_timestamp - prev_timestamp)
            yaw = prev_data['yaw'] + alpha * (closest_data['yaw'] - prev_data['yaw'])
            pitch = prev_data['pitch'] + alpha * (closest_data['pitch'] - prev_data['pitch'])
            roll = prev_data['roll'] + alpha * (closest_data['roll'] - prev_data['roll'])
            
            return yaw, pitch, roll
        else:
            # Use the closest match if no previous timestamp available
            closest_data = imu_data[closest_timestamp]
            return closest_data['yaw'], closest_data['pitch'], closest_data['roll']


def extract_frames_by_number(video_path: str, output_dir: str, start_frame: int, num_frames: int, sep_frame: int) -> Dict[int, str]:
    """
    Extract frames from H.265 video file by frame number.
    
    Args:
        video_path: Path to H.265 video file
        output_dir: Output directory for extracted frames
        start_frame: Starting frame number (skip first N frames)
        num_frames: Number of frames to extract
        
    Returns:
        Dictionary mapping frame index to image file path
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = {}
    
    # Use ffmpeg to extract frames by frame number
    for i in range(num_frames):
        frame_number = start_frame + i * sep_frame
        output_filename = f"frame_{frame_number:06d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Extract frame by frame number using -vf select
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f'select=eq(n\\,{frame_number})',
            '-vframes', '1',
            '-q:v', '2',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and os.path.exists(output_path):
                frame_paths[i] = output_filename  # Only store filename, not full path
                print(f"  Extracted frame {i+1}/{num_frames}: {output_filename}")
            else:
                print(f"  Failed to extract frame {i+1}: {result.stderr}")
        except Exception as e:
            print(f"  Error extracting frame {i+1}: {e}")
    
    return frame_paths


def generate_frame_imu_mapping(frame_timestamps: List[int], imu_data: Dict[int, Dict[str, Any]], 
                              frame_paths: Dict[int, str], start_frame: int, num_frames: int, sep_frame: int) -> Dict[int, Dict[str, Any]]:
    """
    Generate mapping between frame indices and IMU data.
    
    Args:
        frame_timestamps: List of frame timestamps
        imu_data: Dictionary of IMU data
        frame_paths: Dictionary mapping frame index to image filename
        start_frame: Starting frame number
        num_frames: Number of frames to process
        
    Returns:
        Dictionary mapping frame index to frame data
    """
    frame_mapping = {}
    
    for i in range(num_frames):
        frame_idx = start_frame + i * sep_frame
        if i in frame_paths and frame_idx < len(frame_timestamps):
            timestamp = frame_timestamps[frame_idx]
            yaw, pitch, roll = find_closest_imu_data(timestamp, imu_data)
            
            frame_mapping[i] = {
                'image_file': frame_paths[i],  # Only filename, not full path
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'timestamp': timestamp
            }
    
    return frame_mapping


def save_frame_mapping(frame_mapping: Dict[int, Dict[str, Any]], output_path: str):
    """
    Save frame mapping to JSON file.
    
    Args:
        frame_mapping: Dictionary mapping frame index to frame data
        output_path: Path to save the output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(frame_mapping, f, indent=2)
    print(f"  Frame mapping saved to: {output_path}")


def find_data_groups(base_dir: str) -> List[Dict[str, str]]:
    """
    Automatically find all available data groups in the directory.
    
    Args:
        base_dir: Base directory to search for data groups
        
    Returns:
        List of dictionaries containing group information
    """
    data_groups = []
    
    # Find all IMU files
    imu_files = glob.glob(os.path.join(base_dir, "*_imu.txt"))
    
    for imu_file in imu_files:
        # Extract group name from IMU filename
        group_name = os.path.basename(imu_file).replace("_imu.txt", "")
        
        # Check if corresponding video and index files exist
        video_file = os.path.join(base_dir, f"{group_name}_main.h265")
        index_file = os.path.join(base_dir, f"{group_name}_main.h265.index")
        
        if os.path.exists(video_file) and os.path.exists(index_file):
            data_groups.append({
                'name': group_name,
                'imu_file': imu_file,
                'video_file': video_file,
                'index_file': index_file
            })
            print(f"Found data group: {group_name}")
    
    return data_groups


def process_data_group(group_info: Dict[str, str], base_dir: str, start_frame: int = 300, num_frames: int = 5, sep_frame: int = 1):
    """
    Process a single data group.
    
    Args:
        group_info: Dictionary containing group file paths
        base_dir: Base directory for output
        start_frame: Starting frame number
        num_frames: Number of frames to extract
        sep_frame: Frame separation interval
    """
    group_name = group_info['name']
    print(f"\n=== Processing Group: {group_name} ===")
    
    # Create output directory
    output_dir = os.path.join(base_dir, f"{group_name}_extracted_frames")
    
    try:
        # Parse IMU data
        print(f"  Parsing IMU data...")
        imu_data = parse_imu_file(group_info['imu_file'])
        print(f"  Parsed {len(imu_data)} IMU records")
        
        # Parse H.265 index
        print(f"  Parsing H.265 index...")
        frame_timestamps = parse_h265_index(group_info['index_file'])
        print(f"  Found {len(frame_timestamps)} frame timestamps")
        
        # Check if we have enough frames
        max_frame_needed = start_frame + (num_frames - 1) * sep_frame
        if max_frame_needed >= len(frame_timestamps):
            print(f"  Warning: Requested frames exceed available frames. Adjusting parameters.")
            num_frames = max(0, (len(frame_timestamps) - start_frame) // sep_frame + 1)
            if num_frames == 0:
                print(f"  No frames available after skipping.")
                return
        
        # Extract frames
        print(f"  Extracting frames starting from {start_frame} with separation {sep_frame}...")
        frame_paths = extract_frames_by_number(group_info['video_file'], output_dir, start_frame, num_frames, sep_frame)
        print(f"  Extracted {len(frame_paths)} frames")
        
        # Generate frame-IMU mapping
        print(f"  Generating frame-IMU mapping...")
        frame_mapping = generate_frame_imu_mapping(frame_timestamps, imu_data, frame_paths, start_frame, num_frames, sep_frame)
        print(f"  Generated mapping for {len(frame_mapping)} frames")
        
        # Save mapping
        mapping_path = os.path.join(output_dir, "frame_imu_mapping.json")
        save_frame_mapping(frame_mapping, mapping_path)
        
        # Show results summary
        print(f"  Successfully processed {len(frame_mapping)} frames!")
        print(f"  Output directory: {output_dir}")
        
        return frame_mapping
        
    except Exception as e:
        print(f"  Error processing group {group_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_align(video_path: str, index_path: str, imu_path: str, output_dir: str, start_frame: int = 300, num_frames: int = 5, sep_frame: int = 1):
    if not all(os.path.exists(path) for path in [video_path, index_path, imu_path]):
        print("Some required files are missing!")
        return
    
    try:
        # Parse IMU data
        print("Parsing IMU data...")
        imu_data = parse_imu_file(imu_path)
        print(f"Parsed {len(imu_data)} IMU records")
        
        # Parse H.265 index
        print("Parsing H.265 index...")
        frame_timestamps = parse_h265_index(index_path)
        print(f"Found {len(frame_timestamps)} frame timestamps")
        
        # Check if we have enough frames
        max_frame_needed = start_frame + (num_frames - 1) * sep_frame
        if max_frame_needed >= len(frame_timestamps):
            print(f"Warning: Requested frames exceed available frames. Adjusting parameters.")
            num_frames = max(0, (len(frame_timestamps) - start_frame) // sep_frame + 1)
            if num_frames == 0:
                print("No frames available after skipping.")
                return
        
        # Extract frames
        print(f"Extracting frames starting from {start_frame} with separation {sep_frame}...")
        frame_paths = extract_frames_by_number(video_path, output_dir, start_frame, num_frames, sep_frame)
        print(f"Extracted {len(frame_paths)} frames")
        
        # Generate frame-IMU mapping
        print("Genering frame-IMU mapping...")
        frame_mapping = generate_frame_imu_mapping(frame_timestamps, imu_data, frame_paths, start_frame, num_frames, sep_frame)
        print(f"Generated mapping for {len(frame_mapping)} frames")
        
        # Save mapping
        mapping_path = os.path.join(output_dir, "frame_imu_mapping.json")
        save_frame_mapping(frame_mapping, mapping_path)
        
        # Show results
        print("\n=== Frame-IMU Mapping Results ===")
        for idx in sorted(frame_mapping.keys()):
            data = frame_mapping[idx]
            actual_frame = start_frame + idx * sep_frame
            print(f"Frame {actual_frame} (index {idx}):")
            print(f"  Image: {data['image_file']}")
            print(f"  Yaw: {data['yaw']:.2f}°")
            print(f"  Pitch: {data['pitch']:.2f}°")
            print(f"  Roll: {data['roll']:.2f}°")
            print(f"  Timestamp: {data['timestamp']}")
            print()
        
        print(f"Successfully processed {len(frame_mapping)} frames!")
        print(f"Output directory: {output_dir}")
        print(f"Mapping file: {mapping_path}")
        
    except Exception as e:
        print(f"Error processing video and IMU data: {e}")
        import traceback
        traceback.print_exc()

def main_group():
    """Main function to process all available data groups."""
    print("=== Multi-Group IMU Video Processing ===")
    
    # Configuration
    base_dir = "/home/william/extdisk/data/boximu-rgb/imu-box"
    
    # Processing parameters
    start_frame = 200  # Skip first 300 frames
    num_frames = 5    # Only save 2 frames results
    sep_frame = 200   # Extract every 100th frame
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return
    
    try:
        # Find all available data groups
        print("Scanning for data groups...")
        data_groups = find_data_groups(base_dir)
        
        if not data_groups:
            print("No data groups found!")
            return
        
        print(f"Found {len(data_groups)} data groups to process")
        
        # Process each group
        results = {}
        for group_info in data_groups:
            result = process_data_group(group_info, base_dir, start_frame, num_frames, sep_frame)
            if result:
                results[group_info['name']] = result
        
        # Summary
        print(f"\n=== Processing Summary ===")
        print(f"Total groups processed: {len(results)}")
        for group_name, frame_mapping in results.items():
            print(f"  {group_name}: {len(frame_mapping)} frames")
        
        print(f"\nAll data groups processed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Example usage of the frame extraction functionality."""
    print("=== Example Frame Extraction ===")
    
    # Example parameters
    video_path = "/path/to/your/video.h265"
    index_path = "/path/to/your/video.h265.index"
    imu_path = "/path/to/your/imu.txt"
    output_dir = "/path/to/output"
    
    # Frame extraction parameters
    start_frame = 1000    # Start from frame 1000
    num_frames = 10       # Extract 10 frames
    sep_frame = 50        # Extract every 50th frame
    
    # This will extract frames: 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450
    print(f"Will extract frames: {[start_frame + i * sep_frame for i in range(num_frames)]}")
    
    # Uncomment the line below to run the extraction
    process_align(video_path, index_path, imu_path, output_dir, start_frame, num_frames, sep_frame)


if __name__ == "__main__":
    # index_path = "/home/william/extdisk/data/boximu-rgb/imu-box/20250731_140008_main.h265.index"
    # print("Parsing H.265 index...")
    # frame_timestamps = parse_h265_index(index_path)
    # print(f"Found {len(frame_timestamps)} frame timestamps")
    main_group()
