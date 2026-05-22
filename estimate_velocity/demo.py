import cv2
import numpy as np
import os
import pandas as pd
from pipeline import VelocityEstimationPipeline


def create_ipm_birdseye_view(vis_data, camera_intrinsics, camera_height, canvas_size=(800, 600)):
    """
    Create bird's-eye view (IPM) visualization of ground features and velocity
    
    Args:
        vis_data: visualization data from optical flow estimator
        camera_intrinsics: dict with 'fx', 'fy', 'cx', 'cy'
        camera_height: camera height above ground in meters
        canvas_size: (width, height) of output canvas in pixels
        
    Returns:
        IPM canvas showing bird's-eye view
    """
    if vis_data is None:
        return None
    
    canvas_w, canvas_h = canvas_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Define bird's-eye view parameters (meters per pixel)
    meters_per_pixel_x = 0.05  # 5cm per pixel in x
    meters_per_pixel_y = 0.08  # 8cm per pixel in y (forward)
    max_forward_distance = canvas_h * meters_per_pixel_y  # ~48m
    max_lateral_distance = canvas_w * meters_per_pixel_x / 2  # ~20m each side
    
    # Camera position is at bottom center
    camera_x = canvas_w // 2
    camera_y = canvas_h - 50
    
    # Draw distance circles
    for dist in [5, 10, 15, 20, 30, 40]:
        if dist < max_forward_distance:
            radius_px = int(dist / meters_per_pixel_y)
            cv2.circle(canvas, (camera_x, camera_y), radius_px, (40, 40, 40), 1)
            # Label
            label_y = camera_y - radius_px
            if label_y > 20:
                cv2.putText(canvas, f"{dist}m", (camera_x + 5, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Draw lateral lines (left/right boundaries)
    for lateral in [-10, -5, 0, 5, 10]:
        x_px = camera_x + int(lateral / meters_per_pixel_x)
        if 0 <= x_px < canvas_w:
            cv2.line(canvas, (x_px, 0), (x_px, canvas_h), (40, 40, 40), 1)
    
    # Draw camera position
    cv2.circle(canvas, (camera_x, camera_y), 8, (0, 255, 255), -1)
    cv2.putText(canvas, "Camera", (camera_x + 12, camera_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Get feature data
    p0 = vis_data['p0']
    bottom_mask = vis_data['bottom_mask']
    feature_distances = vis_data.get('feature_distances', [])
    flow_vectors = vis_data['flow_vectors']
    
    if len(feature_distances) == 0:
        return canvas
    
    # Get camera parameters
    fy = camera_intrinsics.get('fy', 1080)
    cy = camera_intrinsics.get('cy', 540)
    fx = camera_intrinsics.get('fx', fy)
    cx = camera_intrinsics.get('cx', 960)
    
    # Project ground features to bird's-eye view
    ground_features = p0[bottom_mask]
    
    for i, (feature_pt, distance) in enumerate(zip(ground_features, feature_distances)):
        x_img, y_img = feature_pt.ravel()
        
        # Calculate lateral position (x in ground plane)
        # For small angles: x_ground ≈ distance * (x_img - cx) / fx
        x_ground = distance * (x_img - cx) / fx
        
        # Forward distance is just the distance we calculated
        z_ground = distance
        
        # Convert to canvas coordinates
        canvas_x = camera_x + int(x_ground / meters_per_pixel_x)
        canvas_y = camera_y - int(z_ground / meters_per_pixel_y)
        
        # Check if in canvas bounds
        if 0 <= canvas_x < canvas_w and 0 <= canvas_y < canvas_h:
            # Draw feature point (color by distance: near=green, far=blue)
            color_ratio = min(1.0, distance / 20.0)
            color = (0, int(255 * (1 - color_ratio)), int(255 * color_ratio))
            cv2.circle(canvas, (canvas_x, canvas_y), 4, color, -1)
            
            # Draw velocity vector (flow direction)
            if i < len(flow_vectors):
                flow_y = flow_vectors[i, 1]  # Vertical flow (forward motion)
                # Flow magnitude represents velocity, scale for visualization
                arrow_length = abs(flow_y) * 2  # Scale factor for visibility
                arrow_end_y = canvas_y - int(arrow_length)
                if arrow_end_y >= 0:
                    cv2.arrowedLine(canvas, (canvas_x, canvas_y), 
                                   (canvas_x, arrow_end_y), 
                                   (0, 255, 0), 2, tipLength=0.3)
    
    # Add legend
    cv2.putText(canvas, "Bird's-Eye View (Ground Plane)", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"Features: {len(feature_distances)}", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Distance color legend
    cv2.putText(canvas, "Near", (10, canvas_h - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.circle(canvas, (55, canvas_h - 55), 4, (0, 255, 0), -1)
    cv2.putText(canvas, "Far", (10, canvas_h - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.circle(canvas, (45, canvas_h - 35), 4, (255, 0, 0), -1)
    
    return canvas


def draw_optical_flow(frame, vis_data):
    """Draw optical flow feature points and vectors on frame"""
    if vis_data is None:
        return frame
    
    vis_frame = frame.copy()
    p0 = vis_data['p0']
    p1 = vis_data['p1']
    bottom_mask = vis_data['bottom_mask']
    
    # Draw all tracked points in blue
    for pt in p0:
        x, y = pt.ravel()
        cv2.circle(vis_frame, (int(x), int(y)), 3, (255, 0, 0), -1)
    
    # Draw bottom region points and flow vectors in green
    bottom_p0 = p0[bottom_mask]
    bottom_p1 = p1[bottom_mask]
    
    for pt0, pt1 in zip(bottom_p0, bottom_p1):
        x0, y0 = pt0.ravel()
        x1, y1 = pt1.ravel()
        # Draw flow vector
        cv2.arrowedLine(vis_frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2, tipLength=0.3)
        # Draw endpoint
        cv2.circle(vis_frame, (int(x1), int(y1)), 5, (0, 255, 0), -1)
    
    # Draw ground region boundaries
    h = frame.shape[0]
    ground_start = int(h * 0.65)
    ground_focus = int(h * 0.70)
    cv2.line(vis_frame, (0, ground_start), (frame.shape[1], ground_start), (100, 100, 100), 1)
    cv2.line(vis_frame, (0, ground_focus), (frame.shape[1], ground_focus), (0, 255, 255), 2)
    
    # Add labels
    cv2.putText(vis_frame, "Detection Region (65%)", (10, ground_start - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(vis_frame, "Ground Focus (70%)", (10, ground_focus - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis_frame


def demo_motorev_data(data_folder, camera_height=1.2, show_flow=True, 
                      ground_truth_speed=None, auto_calibrate=True):
    """
    Demo with motorEV dataset structure
    
    Args:
        data_folder: folder containing color/, color.txt, imu.txt
        camera_height: camera height above ground in meters
        show_flow: visualize optical flow feature points
        ground_truth_speed: ground truth speed for evaluation only (not used in estimation)
        auto_calibrate: enable automatic online calibration using IMU (default: True)
    """
    color_txt = os.path.join(data_folder, "color.txt")
    imu_txt = os.path.join(data_folder, "imu.txt")
    
    if not os.path.exists(color_txt):
        print(f"Error: {color_txt} not found")
        return
        
    # Load color timestamps and paths
    color_data = []
    with open(color_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # Timestamp appears to be in microseconds (based on magnitude ~1e12)
                timestamp = float(parts[0]) / 1e6  # Convert microseconds to seconds
                img_path = os.path.join(data_folder, parts[1])
                color_data.append((timestamp, img_path))
    
    print(f"Loaded {len(color_data)} image timestamps")
    if color_data:
        print(f"Image time range: [{color_data[0][0]:.3f}, {color_data[-1][0]:.3f}] seconds")
    
    # Load IMU data if available
    imu_data = None
    if os.path.exists(imu_txt):
        # Format: #Time Gx Gy Gz Ax Ay Az
        # Units: Time in microseconds, Gyro in mdps (millidegrees/sec), Accel in mg (milligravity)
        imu_df = pd.read_csv(imu_txt, sep=r'\s+', comment='#', 
                            names=['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        
        # Unit conversions (following imugyrtraj convention):
        # Timestamp: microseconds -> seconds
        imu_df['timestamp'] = imu_df['timestamp'] / 1e6
        
        # Gyro: mdps (millidegrees per second) -> rad/s
        imu_df['gx'] = imu_df['gx'] * np.pi / (180.0 * 1000)
        imu_df['gy'] = imu_df['gy'] * np.pi / (180.0 * 1000)
        imu_df['gz'] = imu_df['gz'] * np.pi / (180.0 * 1000)
        
        # Accel: mg (milligravity) -> m/s²
        imu_df['ax'] = imu_df['ax'] * 9.81 / 1000.0
        imu_df['ay'] = imu_df['ay'] * 9.81 / 1000.0
        imu_df['az'] = imu_df['az'] * 9.81 / 1000.0
        
        print(f"Loaded {len(imu_df)} IMU samples")
        print(f"IMU accel range: ax=[{imu_df['ax'].min():.2f}, {imu_df['ax'].max():.2f}] m/s²")
        print(f"IMU gyro range: gx=[{imu_df['gx'].min():.4f}, {imu_df['gx'].max():.4f}] rad/s")
        imu_data = imu_df
    
    # Create pipeline with calibrated parameters
    camera_intrinsics = {
        'fx': 1036.558812,  # Approximate focal length for typical camera
        'fy': 1037.823922,
        'cx': 980.342506,
        'cy': 578.285398
    }
    
    # Create pipeline
    # Note: velocity_scale=1.0 because we use corrected distance estimation (30m)
    initial_scale = 1.0  # With corrected physics model, scale should be ~1.0
    pipeline = VelocityEstimationPipeline(camera_intrinsics, camera_height, 
                                          initial_scale, auto_calibrate)
    
    velocities = []
    imu_velocities = []
    
    print(f"\nEstimation Method:")
    print(f"  Optical Flow: Physics-based model (estimated ground distance: 30m)")
    print(f"  IMU: Statistical heuristic (for reference only)")
    print(f"  Camera height: {camera_height} m")
    if ground_truth_speed:
        print(f"  Ground truth (for evaluation): {ground_truth_speed} km/h")
    print()
    
    # Process images
    for i, (img_timestamp, img_path) in enumerate(color_data):
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue
            
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: failed to read {img_path}, skipping")
            continue
        
        # Find closest IMU sample
        imu_accel = None
        if imu_data is not None:
            idx = (imu_data['timestamp'] - img_timestamp).abs().idxmin()
            imu_row = imu_data.iloc[idx]
            # Extract acceleration (ax, ay, az) - already converted to m/s²
            imu_accel = np.array([imu_row['ax'], imu_row['ay'], imu_row['az']])
        
        # Debug for first 20 frames to understand velocity estimation
        debug_mode = (i < 20)
        if debug_mode:
            print(f"\n=== Frame {i}, t={img_timestamp:.3f}s ===")
            if imu_accel is not None:
                accel_mag = np.linalg.norm(imu_accel)
                print(f"  [IMU Accel] ax={imu_accel[0]:.3f}, ay={imu_accel[1]:.3f}, az={imu_accel[2]:.3f} m/s²")
                print(f"  [IMU Mag] |a|={accel_mag:.3f} m/s² (expected ~9.81 when stationary)")
        
        # Process with visualization
        result = pipeline.process(frame, img_timestamp, imu_accel, 
                                 debug=debug_mode, return_visualization=show_flow)
        
        velocity = result['velocity']
        optical_vel = result['optical_velocity']
        imu_vel = result['imu_velocity']
        velocities.append(velocity)
        if imu_vel is not None:
            imu_velocities.append(imu_vel)
        
        # Print status
        opt_str = f"{optical_vel:.1f}" if optical_vel is not None else "None"
        imu_str = f"{imu_vel:.1f}" if imu_vel is not None else "None"
        
        # Print status
        gt_str = f", GT:{ground_truth_speed:.1f}" if ground_truth_speed else ""
        
        if i % 20 == 0 or i < 20:
            # Detailed output every 20 frames
            print(f"Frame {i}: Fused={velocity:.1f} km/h (Opt:{opt_str}, IMU:{imu_str}{gt_str})")
        else:
            # Simpler output
            print(f"Frame {i}: {velocity:.1f} km/h (opt:{opt_str}, imu:{imu_str}{gt_str})")
        
        # Draw optical flow visualization and IPM bird's-eye view
        flow_info = ""
        if show_flow and 'vis_data' in result and result['vis_data'] is not None:
            # Camera view with optical flow
            camera_view = draw_optical_flow(frame, result['vis_data'])
            flow_info = f" flow:{result['vis_data']['flow']:.3f}px"
            
            # Bird's-eye view (IPM)
            ipm_view = create_ipm_birdseye_view(result['vis_data'], camera_intrinsics, camera_height)
            
            # Combine views side by side
            if ipm_view is not None:
                # Resize camera view to match height of IPM view
                h_ipm, w_ipm = ipm_view.shape[:2]
                h_cam, w_cam = camera_view.shape[:2]
                scale = h_ipm / h_cam
                w_cam_scaled = int(w_cam * scale)
                camera_view_scaled = cv2.resize(camera_view, (w_cam_scaled, h_ipm))
                
                # Combine horizontally
                frame = np.hstack([camera_view_scaled, ipm_view])
            else:
                frame = camera_view
        
        # Overlay velocity info with background for better visibility
        vel_text = f"Velocity: {velocity:.1f} km/h"
        if ground_truth_speed:
            vel_text += f" (GT: {ground_truth_speed:.1f})"
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (600, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        cv2.putText(frame, vel_text, 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Opt:{opt_str} IMU:{imu_str}{flow_info}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Velocity Estimation', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Print statistics
    if velocities:
        velocities = np.array(velocities)
        final_scale = pipeline.get_velocity_scale()
        is_calibrated = pipeline.is_calibrated()
        
        print(f"\n{'='*60}")
        print(f"Velocity Estimation Results:")
        print(f"\nEstimation Method:")
        print(f"  Physics-based optical flow (effective ground distance: 30m)")
        
        print(f"\nFused Speed Statistics:")
        print(f"  Mean: {np.mean(velocities):.2f} km/h")
        print(f"  Median: {np.median(velocities):.2f} km/h")
        print(f"  Std: {np.std(velocities):.2f} km/h")
        print(f"  Min: {np.min(velocities):.2f} km/h")
        print(f"  Max: {np.max(velocities):.2f} km/h")
        
        if imu_velocities:
            imu_velocities = np.array(imu_velocities)
            print(f"\nIMU Speed Statistics:")
            print(f"  Mean: {np.mean(imu_velocities):.2f} km/h")
            print(f"  Median: {np.median(imu_velocities):.2f} km/h")
            print(f"  Std: {np.std(imu_velocities):.2f} km/h")
        
        # Compare with ground truth if provided
        if ground_truth_speed:
            print(f"\nGround Truth Comparison:")
            print(f"  Expected: {ground_truth_speed:.2f} km/h")
            print(f"  Estimated: {np.mean(velocities):.2f} km/h")
            error = abs(np.mean(velocities) - ground_truth_speed)
            error_pct = (error / ground_truth_speed) * 100
            print(f"  Error: {error:.2f} km/h ({error_pct:.1f}%)")
        
        print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    # Default parameters
    # data_folder = "/home/william/extdisk/data/motorEV/20260116/3-4kmh-slow-speed/"
    # ground_truth = 3.5
    data_folder = "/home/william/extdisk/data/motorEV/20260116/70kmh-high-speed/"
    ground_truth = 70 # Ground truth for evaluation
    camera_height = 1.4
    auto_calibrate = True  # Enable IMU-based auto-calibration with improved coefficients
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    if len(sys.argv) > 2:
        camera_height = float(sys.argv[2])
    if len(sys.argv) > 3:
        ground_truth = float(sys.argv[3])
    
    print(f"\n{'='*60}")
    print(f"Velocity Estimation - Auto-Calibrating")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Data folder: {data_folder}")
    print(f"  Camera height: {camera_height} m")
    print(f"  Auto-calibration: {'Enabled (IMU-assisted)' if auto_calibrate else 'Disabled (physics-based only)'}")
    if ground_truth:
        print(f"  Ground truth (for evaluation): {ground_truth} km/h")
    print(f"{'='*60}\n")
    
    if len(sys.argv) <= 1:
        print("Usage: python demo.py [data_folder] [camera_height] [ground_truth]")
        print("  camera_height: camera height above ground (default: 1.2)")
        print("  ground_truth: optional, for evaluation only\n")
    
    demo_motorev_data(data_folder, 
                     camera_height=camera_height, 
                     show_flow=True,
                     ground_truth_speed=ground_truth,
                     auto_calibrate=auto_calibrate)
