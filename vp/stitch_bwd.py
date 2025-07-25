import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def bev_stitching_optimized(
    image_paths: List[str],
    camera_params: List[dict],  # List of camera intrinsics/extrinsics (K, dist, R, t)
    bev_range: Tuple[float, float, float, float] = (-10, 10, -10, 10),  # X_min, X_max, Y_min, Y_max
    bev_resolution: Tuple[int, int] = (1000, 1000),  # Width, height of output BEV
    show_result: bool = True,
) -> np.ndarray:
    """
    Memory-efficient BEV stitching by reverse mapping from virtual BEV plane to source images.
    
    Args:
        image_paths: Paths to input images (front/left/right/back views).
        camera_params: Camera parameters for each image:
            - 'K': 3x3 intrinsic matrix
            - 'dist': Distortion coefficients (k1,k2,p1,p2,k3)
            - 'R': 3x3 rotation matrix
            - 't': 3x1 translation vector
        bev_range: Physical range (meters) covered by BEV in X/Y axes.
        bev_resolution: Output BEV image dimensions.
        show_result: Whether to display the result.
    
    Returns:
        Stitched BEV image (uint8 numpy array).
    """
    # Load all source images
    images = [cv2.imread(path) for path in image_paths]
    assert all(img is not None for img in images), "Image loading failed."
    
    # Initialize virtual BEV plane
    bev_width, bev_height = bev_resolution
    X_min, X_max, Y_min, Y_max = bev_range
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.float32)
    bev_weights = np.zeros((bev_height, bev_width), dtype=np.float32)
    
    # Iterate over every pixel in the BEV plane
    for v in range(bev_height):
        for u in range(bev_width):
            # Step 1: BEV pixel → World coordinates (Z=0 ground plane)
            X = X_min + u * (X_max - X_min) / bev_width
            Y = Y_min + v * (Y_max - Y_min) / bev_height
            Z = 0.0
            
            # Step 2: Check all cameras for contributions
            for i, (img, cam) in enumerate(zip(images, camera_params)):
                # World → Camera coordinates
                X_cam = cam['R'] @ np.array([X, Y, Z]) + cam['t']
                X_c, Y_c, Z_c = X_cam.ravel()
                
                # Skip if point is behind the camera
                if Z_c <= 0:
                    continue
                
                # Camera coordinates → Image pixels (with distortion)
                x = X_c / Z_c
                y = Y_c / Z_c
                r2 = x**2 + y**2
                radial = 1 + cam['dist'][0]*r2 + cam['dist'][1]*r2**2 + cam['dist'][4]*r2**3
                x_distorted = x*radial + 2*cam['dist'][2]*x*y + cam['dist'][3]*(r2 + 2*x**2)
                y_distorted = y*radial + cam['dist'][2]*(r2 + 2*y**2) + 2*cam['dist'][3]*x*y
                u_img = cam['K'][0, 0]*x_distorted + cam['K'][0, 2]
                v_img = cam['K'][1, 1]*y_distorted + cam['K'][1, 2]
                
                # Bilinear sampling if within image bounds
                if 0 <= u_img < img.shape[1] and 0 <= v_img < img.shape[0]:
                    pixel = bilinear_sample(img, u_img, v_img)
                    # Weight by distance to image border (simple heuristic)
                    weight = min(u_img, img.shape[1]-u_img, v_img, img.shape[0]-v_img)
                    bev_image[v, u] += pixel * weight
                    bev_weights[v, u] += weight
    
    # Normalize by accumulated weights
    bev_image = np.divide(
        bev_image, 
        bev_weights[..., np.newaxis], 
        out=np.zeros_like(bev_image), 
        where=bev_weights[..., np.newaxis] > 0
    )
    
    # Visualization
    if show_result:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        plt.title("Source Camera View")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(bev_image.astype(np.uint8))
        plt.title("Stitched BEV (Optimized)")
        plt.axis('off')
        plt.show()
    
    return bev_image.astype(np.uint8)

def bilinear_sample(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear interpolation for sub-pixel sampling."""
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
    
    # Get four neighboring pixels
    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]
    
    # Weights
    wx = x - x0
    wy = y - y0
    
    # Interpolation
    top = (1 - wx) * p00 + wx * p01
    bottom = (1 - wx) * p10 + wx * p11
    return (1 - wy) * top + wy * bottom

# Example usage
if __name__ == "__main__":
    # Mock camera parameters (replace with actual calibration)
    camera_params = [
        {
            'K': np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]),
            'dist': np.array([-0.1, 0.01, 0, 0, 0]),
            'R': np.eye(3),
            't': np.array([0, 0, 2])
        },
        # Add more cameras...
    ]
    
    # Run stitching
    bev_result = bev_stitching_optimized(
        image_paths=["front.jpg", "left.jpg"],
        camera_params=camera_params,
        bev_range=(-5, 5, -5, 5),  # 5m x 5m area
        bev_resolution=(800, 800)
    )