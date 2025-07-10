import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from stitch.apriltag import AprilGridDetector, ZED_DIST, ZED_K

def undistort_images(images, camera_matrix, dist_coeffs):
    """Undistort a list of images using camera calibration parameters"""
    undistorted = []
    h, w = images[0].shape[:2]
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    for img in images:
        undist = cv2.undistort(
            img, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        # Crop the image (optional)
        x, y, w, h = roi
        undist = undist[y:y+h, x:x+w]
        undistorted.append(undist)
    
    return undistorted, new_camera_matrix

def create_avm(images, rvecs, tvecs, camera_matrix, dist_coeffs, output_size=(1000, 1000)):
    """
    Create Around View Monitoring from 3 images and their poses
    """
    # Validate input
    assert len(images) == 3, "Exactly 3 images required"
    assert len(rvecs) == 3 and len(tvecs) == 3, "Exactly 3 poses required"
    
    # 1. Undistort images first
    images, new_camera_matrix = undistort_images(images, camera_matrix, dist_coeffs)
    
    # 2. Convert all poses to be relative to the SECOND image's coordinate system
    R_ref = cv2.Rodrigues(rvecs[1])[0]  # Second image's rotation
    t_ref = tvecs[1]                     # Second image's translation
    
    relative_poses = []
    for i in range(3):
        Ri = cv2.Rodrigues(rvecs[i])[0]
        ti = tvecs[i]
        
        # Transform to second camera's coordinate system
        R_rel = R_ref.T @ Ri
        t_rel = R_ref.T @ (ti - t_ref)
        
        relative_poses.append({
            'R': R_rel if i != 1 else np.eye(3),  # Second image is identity
            't': t_rel if i != 1 else np.zeros(3), # Second image is origin
            'rvec': cv2.Rodrigues(R_rel)[0] if i != 1 else np.zeros(3)
        })
    
    # 3. Define ground plane in second image's coordinate system
    ground_width = 10.0  # meters (x-direction)
    ground_length = 10.0 # meters (y-direction)
    
    ground_pts = np.array([
        [-ground_width/2, -ground_length/2, 0],
        [ground_width/2, -ground_length/2, 0],
        [ground_width/2, ground_length/2, 0],
        [-ground_width/2, ground_length/2, 0]
    ], dtype=np.float32)
    
    # 4. Project ground points to each image and compute homographies
    homographies = []
    warped_images = []
    
    for i in range(3):
        pose = relative_poses[i]
        
        # Project ground points to image
        img_pts, _ = cv2.projectPoints(
            ground_pts,
            pose['rvec'],
            pose['t'],
            new_camera_matrix,
            None
        )
        img_pts = img_pts.reshape(-1, 2)
        
        # Define output points in bird's-eye view
        output_pts = np.array([
            [0, 0],
            [output_size[0]-1, 0],
            [output_size[0]-1, output_size[1]-1],
            [0, output_size[1]-1]
        ], dtype=np.float32)
        
        # Compute homography
        H, status = cv2.findHomography(img_pts, output_pts, cv2.RANSAC, 5.0)
        if H is None or np.sum(status) < 2:
            raise ValueError(f"Failed to compute homography for image {i+1}")
        homographies.append(H)
        
        # Warp the image
        warped = cv2.warpPerspective(
            images[i], H, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        warped_images.append(warped)
    
    # 5. Improved blending with distance-based weights
    masks = []
    for i, warped in enumerate(warped_images):
        mask = (warped > 0).any(axis=2).astype(np.float32)
        
        # Create weights based on distance from image center
        y, x = np.indices(mask.shape)
        if i == 0:  # Center weight for reference image
            center = np.array([output_size[0]/2, output_size[1]/2])
        else:  # Offset weight for other images
            offset_x = 100 if i == 1 else -100  # Adjust based on camera position
            center = np.array([output_size[0]/2 + offset_x, output_size[1]/2])
            
        dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        mask = mask * (1 - dist/max_dist)
        masks.append(mask)
    
    # Normalize and blend
    mask_sum = np.stack(masks).sum(axis=0)
    mask_sum[mask_sum == 0] = 1  # Avoid division by zero
    
    birdseye_view = np.zeros((output_size[1], output_size[0], 3), dtype=np.float32)
    for warped, mask in zip(warped_images, masks):
        birdseye_view += warped.astype(np.float32) * (mask/mask_sum)[..., np.newaxis]
    
    return birdseye_view.astype(np.uint8), warped_images

if __name__ == "__main__":
    project_dir = Path.cwd().parent
    data_dir = project_dir / Path('data/zed_360')
    images = []
    rvecs = []
    tvecs = []
    at_detector = AprilGridDetector()
    
    # Collect all valid frames first
    valid_frames = []
    for data_path in sorted(data_dir.glob('*.png')):
        try:
            im_raw = cv2.imread(str(data_path))
            if im_raw is None:
                continue
                
            im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
            detections = at_detector.detect(im_gray)
            
            if len(detections) < 4:  # Require minimum 4 tags
                continue
                
            rvec, tvec = at_detector.estimate_pose(detections, ZED_K, ZED_DIST)
            if rvec is None or tvec is None:
                continue
                
            # Convert to camera pose in grid coordinates
            R, _ = cv2.Rodrigues(rvec)
            cam_pose = -R.T @ tvec
            cam_rot = cv2.Rodrigues(R.T)[0]
            
            valid_frames.append({
                'image': im_raw,
                'rvec': cam_rot,
                'tvec': cam_pose
            })
        except Exception as e:
            print(f"Error processing {data_path.name}: {str(e)}")
            continue
    
    # Select the best 3 frames (most tags detected)
    if len(valid_frames) < 3:
        raise ValueError(f"Need at least 3 valid frames, only got {len(valid_frames)}")
    
    # Sort by number of tags detected (descending)
    valid_frames = sorted(valid_frames, key=lambda x: -len(x.get('detections', [])))[:3]
    
    # Prepare inputs for AVM
    images = [frame['image'] for frame in valid_frames]
    rvecs = [frame['rvec'] for frame in valid_frames]
    tvecs = [frame['tvec'] for frame in valid_frames]
    
    try:
        avm, warped_imgs = create_avm(images, rvecs, tvecs, ZED_K, ZED_DIST)
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        for i, ax in enumerate(axes.flat[:3]):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Input Image {i+1}")
            ax.axis('off')
        
        axes.flat[3].imshow(cv2.cvtColor(avm, cv2.COLOR_BGR2RGB))
        axes.flat[3].set_title("Around View Monitoring")
        axes.flat[3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the result
        cv2.imwrite('avm_result.jpg', avm)
        print("AVM created successfully!")
    except Exception as e:
        print(f"Error creating AVM: {str(e)}")