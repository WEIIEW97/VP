import cv2
import numpy as np
import os
from pipeline import VelocityEstimationPipeline


def demo_with_video(video_path, camera_intrinsics=None):
    """
    Demo velocity estimation with video file
    
    Args:
        video_path: path to video file
        camera_intrinsics: camera intrinsics dict (optional)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    pipeline = VelocityEstimationPipeline(camera_intrinsics)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_count / fps
        
        # Process frame
        result = pipeline.process(frame, timestamp)
        
        # Display results
        velocity = result['velocity']
        range_str = result['range_string']
        
        # Draw text on frame
        cv2.putText(frame, f"Velocity: {velocity:.1f} km/h", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Range: {range_str}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        if result['optical_velocity'] is not None:
            cv2.putText(frame, f"Optical: {result['optical_velocity']:.1f} km/h", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Velocity Estimation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()


def demo_with_images_and_imu(image_folder, imu_data_path=None):
    """
    Demo velocity estimation with image sequence and IMU data
    
    Args:
        image_folder: folder containing sequential images
        imu_data_path: path to IMU data file (optional)
    """
    # Load images
    image_files = sorted([f for f in os.listdir(image_folder) 
                         if f.endswith(('.jpg', '.png'))])
    
    if len(image_files) < 2:
        print("Error: need at least 2 images")
        return
        
    print(f"Found {len(image_files)} images")
    
    # Load IMU data if available
    imu_data = None
    if imu_data_path and os.path.exists(imu_data_path):
        imu_data = np.loadtxt(imu_data_path)
        print(f"Loaded IMU data with shape {imu_data.shape}")
        
    pipeline = VelocityEstimationPipeline()
    
    # Process images
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
            
        # Timestamp (assume 30 fps)
        timestamp = i / 30.0
        
        # Get IMU data if available
        imu_accel = None
        if imu_data is not None and i < len(imu_data):
            imu_accel = imu_data[i, :3]
            
        # Process
        result = pipeline.process(frame, timestamp, imu_accel)
        
        # Display
        velocity = result['velocity']
        range_str = result['range_string']
        
        print(f"Frame {i}: velocity={velocity:.1f} km/h, range={range_str}")
        
        # Visualize
        cv2.putText(frame, f"Velocity: {velocity:.1f} km/h", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Range: {range_str}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.imshow('Velocity Estimation', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()


def demo_synthetic():
    """Demo with synthetic data to verify functionality"""
    print("Running synthetic data demo...")
    
    pipeline = VelocityEstimationPipeline()
    
    # Generate synthetic frames with known motion
    width, height = 640, 480
    num_frames = 50
    
    # Simulate different speeds
    speed_phases = [
        (0, 10, 2.0),    # 0-10: ~2 km/h (very low)
        (10, 20, 15.0),  # 10-20: ~15 km/h (low)
        (20, 35, 45.0),  # 20-35: ~45 km/h (medium)
        (35, 50, 80.0),  # 35-50: ~80 km/h (high)
    ]
    
    results = []
    
    for i in range(num_frames):
        # Determine current speed
        true_speed = 0.0
        for start, end, speed in speed_phases:
            if start <= i < end:
                true_speed = speed
                break
                
        # Generate synthetic frame with dots moving
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add random dots (features)
        num_dots = 50
        for _ in range(num_dots):
            x = np.random.randint(0, width)
            y = np.random.randint(height//2, height)  # bottom half
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
            
        # Simulate motion blur effect proportional to speed
        motion_pixels = int(true_speed / 10.0)
        if motion_pixels > 0:
            kernel = np.zeros((motion_pixels, motion_pixels))
            kernel[:, motion_pixels//2] = 1.0
            kernel = kernel / motion_pixels
            frame = cv2.filter2D(frame, -1, kernel)
            
        timestamp = i / 30.0
        
        # Generate synthetic IMU data
        base_accel = true_speed / 20.0  # rough acceleration
        imu_accel = np.array([base_accel, 0.0, 9.81]) + np.random.randn(3) * 0.1
        
        # Process
        result = pipeline.process(frame, timestamp, imu_accel)
        results.append(result)
        
        velocity = result['velocity']
        range_str = result['range_string']
        
        print(f"Frame {i:2d}: true={true_speed:5.1f} km/h, "
              f"estimated={velocity:5.1f} km/h, range={range_str}")
              
    print("\nDemo completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "synthetic":
            demo_synthetic()
        elif sys.argv[1] == "video" and len(sys.argv) > 2:
            demo_with_video(sys.argv[2])
        elif sys.argv[1] == "images" and len(sys.argv) > 2:
            imu_path = sys.argv[3] if len(sys.argv) > 3 else None
            demo_with_images_and_imu(sys.argv[2], imu_path)
        else:
            print("Usage:")
            print("  python demo.py synthetic")
            print("  python demo.py video <video_path>")
            print("  python demo.py images <image_folder> [imu_data_path]")
    else:
        demo_synthetic()
