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
    
    # 2. Convert rotation vectors to rotation matrices
    R_mats = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    
    # 3. Define ground plane points (in world coordinates)
    # Adjust these values based on your scene scale
    ground_width = 2.0  # meters (x-direction)
    ground_length = 2.0  # meters (y-direction)
    
    # Create 3D points (must be Nx3 array)
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
        # Project ground points to image
        img_pts, _ = cv2.projectPoints(
            ground_pts,
            rvecs[i],
            tvecs[i],
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
        
        # Compute homography (using RANSAC for robustness)
        H, status = cv2.findHomography(img_pts, output_pts, cv2.RANSAC, 5.0)
        if H is None or np.sum(status) < 2:  # At least 2 good points
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
    
    # 5. Improved blending with alpha masks
    masks = []
    for warped in warped_images:
        mask = (warped > 0).any(axis=2).astype(np.float32)
        # Create distance-based weights
        y, x = np.indices(mask.shape)
        center = np.array([output_size[0]/2, output_size[1]/2])
        dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        mask = mask * (1 - dist/max_dist)
        masks.append(mask)
    
    # Normalize masks
    mask_sum = np.stack(masks).sum(axis=0)
    mask_sum[mask_sum == 0] = 1  # Avoid division by zero
    
    # Blend images
    birdseye_view = np.zeros((output_size[1], output_size[0], 3), dtype=np.float32)
    for warped, mask in zip(warped_images, masks):
        birdseye_view += warped.astype(np.float32) * (mask/mask_sum)[..., np.newaxis]
    
    return birdseye_view.astype(np.uint8), warped_images

if __name__ == "__main__":
    data_dir = Path('/home/william/Codes/vp/data/zed_360')
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