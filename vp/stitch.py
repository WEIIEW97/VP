from typing import List, Optional
import cv2
from cv2.xphoto import createSimpleWB
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from apriltag import AprilGridDetector, ZED_DIST, ZED_K
from ipm import IPM, IPMInfo
import scipy.ndimage


def undistort_images(images, camera_matrix, dist_coeffs):
    """Undistort a list of images using camera calibration parameters"""
    undistorted = []
    h, w = images[0].shape[:2]

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    for img in images:
        undist = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        # Crop the image (optional)
        x, y, w, h = roi
        undist = undist[y : y + h, x : x + w]
        undistorted.append(undist)

    return undistorted, new_camera_matrix


def get_homogeneous_transform(rvec: np.ndarray, tvec: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def process_single_bev(
    im_rgb: np.ndarray, detector: AprilGridDetector, ipm_info: IPMInfo
):
    im = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
    ret = detector.detect(im)
    rvec, tvec = detector.estimate_pose(ret, ZED_K, ZED_DIST, id_end=23)

    T_c_b = get_homogeneous_transform(rvec, tvec)
    R_c_b = cv2.Rodrigues(rvec)[0]

    ipm = IPM(ZED_K, ZED_DIST, (1280, 720), R_c_b, tvec)
    bev_image = ipm.GetIPMImage(im_rgb, ipm_info, R_c_g=R_c_b)
    return bev_image, T_c_b


def stitch(
    image_paths: List[str],
    ipm_x_scale: float = 500,
    ipm_y_scale: float = 500,
    base_image_index: int = 1,
    show_result: bool = True,
) -> Optional[np.ndarray]:
    """
    Stitch multiple bird's-eye view (BEV) images together and show debug view with all BEVs.

    Args:
        image_paths: List of paths to input images
        ipm_x_scale: Scale factor for IPM transformation in x direction
        ipm_y_scale: Scale factor for IPM transformation in y direction
        base_image_index: Index of image to use as reference (0-based)
        show_result: Whether to display the result using matplotlib

    Returns:
        The stitched image if successful, None otherwise
    """

    # Initialize detector and IPM info
    at_detector = AprilGridDetector()
    ipm_info = IPMInfo()
    ipm_info.x_scale = ipm_x_scale
    ipm_info.y_scale = ipm_y_scale

    # Load and process images
    original_images = []
    bevs = []
    transforms = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        original_images.append(img)
        bev, T_c_b = process_single_bev(img, at_detector, ipm_info)
        bevs.append(bev)
        transforms.append(T_c_b)

    total_bev = np.zeros_like(bevs[0], dtype=np.float32)
    total_weights = np.zeros(bevs[0].shape, dtype=np.float32)
    weighted_total_bev = np.zeros(bevs[0].shape, dtype=np.float32)
    count_map = np.zeros(bevs[0].shape[:2], dtype=np.float32)
    for bev in bevs:
        bev_float = bev.astype(np.float32)
        total_bev += bev
        if bev.ndim == 3:
            mask = bev.sum(axis=-1) > 0
        else:
            mask = bev > 0

        count_map[mask] += 1

        weight_map = scipy.ndimage.distance_transform_edt(mask.astype(np.uint8))
        if bev.ndim == 3:
            weight_map = weight_map[..., np.newaxis]

        weighted_total_bev += bev_float * weight_map
        total_weights += weight_map

    if total_bev.ndim == 3:
        count_map = count_map[..., np.newaxis]

    final_bev = np.divide(
        total_bev, count_map, out=np.zeros_like(total_bev), where=count_map != 0
    )
    final_bev_edt = np.divide(
        weighted_total_bev,
        total_weights,
        out=np.zeros_like(weighted_total_bev),
        where=total_weights != 0,
    )
    stitched_image = final_bev.astype(np.uint8)
    sticthed_image_edt = final_bev_edt.astype(np.uint8)
    # refine_sticthed = process_averaged_stitching(bevs)

    if show_result:
        # Create debug figure with all BEVs and stitched result
        plt.figure(figsize=(20, 12))

        # Show original images
        for i, img in enumerate(original_images):
            plt.subplot(3, len(image_paths) + 1, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Original {i}", fontsize=10)
            plt.axis("off")

        # Show BEV images
        for i, bev in enumerate(bevs):
            plt.subplot(3, len(image_paths) + 1, len(image_paths) + i + 1)
            plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
            plt.title(f"BEV {i}", fontsize=10)
            plt.axis("off")

        # Show stitched result
        plt.subplot(3, 1, 3)
        plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
        plt.title(
            f"Final Stitched Result (Reference: Image {base_image_index})", fontsize=12
        )
        plt.axis("off")

        # plt.subplot(3, 1, 2)
        # plt.imshow(sticthed_image_edt)
        # plt.title(f"Refined Stitched Result", fontsize=12)
        # plt.axis('off')

        plt.tight_layout()
        plt.show()

    return stitched_image


def distance_weights(mask: np.ndarray, blend_width=100) -> np.ndarray:
    """Generate smooth weights based on distance from image edges"""
    binary_mask = (mask > 0).astype(np.uint8) * 255
    # compute distance from nearest edge
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    weights = np.clip(dist_transform / blend_width, 0, 1)
    return weights


def gradient_weights(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Generate weights considering image content"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gray_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient_img = np.sqrt(gray_x**2 + gray_y**2)

    dist_weights = distance_weights(mask)

    edge_weights = 1 / (1 + gradient_img)
    edge_weights = cv2.GaussianBlur(edge_weights, (101, 101), 0)

    combined = dist_weights * edge_weights
    return combined / combined.max()


def pyramid_weights(mask: np.ndarray, num_levels=5) -> List[np.ndarray]:
    weights = distance_weights(mask)

    pyramid = [weights]
    for _ in range(num_levels - 1):
        weights = cv2.pyrDown(weights)
        pyramid.append(weights)
    return pyramid


def find_optimal_seam(
    img1: np.ndarray, img2: np.ndarray, overlap_mask: np.ndarray
) -> np.ndarray:
    """Compute optimal seam using graph cuts"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute difference
    diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))

    # Create graph cut weights
    weights = np.zeros_like(diff)
    weights[overlap_mask] = diff[overlap_mask]

    # Find minimum cut path (simplified)
    seam_mask = np.zeros_like(overlap_mask)
    for y in range(weights.shape[0]):
        x_min = np.argmin(weights[y])
        seam_mask[y, :x_min] = 1

    return seam_mask


def apply_weights(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    weights = []
    for img, mask in zip(images, masks):
        w = gradient_weights(img, mask)
        weights.append(w)

    sum_weights = np.sum(weights, axis=0)
    sum_weights[sum_weights < 1e-7] = 1
    norm_weights = [w / sum_weights for w in weights]

    res = np.zeros_like(images[0], dtype=np.float32)
    for img, w in zip(images, norm_weights):
        res += img.astype(np.float32) * cv2.merge([w, w, w])

    return res.astype(np.uint8)


def multi_band_blend(images: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    """Advanced blending with pyramid decomposition"""
    blender = cv2.detail.MultiBandBlender()
    h, w = images[0].shape[:2]
    blender.prepare((0, 0, w, h))

    for img, mask in zip(images, masks):
        blender.feed(img.astype(np.float32), mask.astype(np.uint8))

    blended, _ = blender.blend()
    return np.clip(blended, 0, 255).astype(np.uint8)


def post_process_stitched(
    stitched_image: np.ndarray, bevs: List[np.ndarray]
) -> np.ndarray:
    """
    Enhanced post-processing for averaged BEV images
    Args:
        stitched_image: Raw averaged output from your code
    Returns:
        Enhanced 8-bit BGR image
    """
    # Convert from accumulated sum to proper average
    stitched_image = (stitched_image / len(bevs)).astype(np.uint8)

    # 1. White Balancing (Critical for averaged images)
    wb = createSimpleWB()
    img_wb = wb.balanceWhite(stitched_image.astype(np.float32) / 255)
    img_wb = (img_wb * 255).astype(np.uint8)

    # 2. Noise Reduction (Important for overlapping areas)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_wb,
        h=7,  # Filter strength
        hColor=7,  # Color component strength
        templateWindowSize=7,  # Odd size (3-21)
        searchWindowSize=21,  # Odd size (>template)
    )

    # 3. Edge-Preserving Contrast Enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 4. Sharpening (Unsharp Mask)
    blurred = cv2.GaussianBlur(l, (0, 0), 3)
    l = cv2.addWeighted(l, 1.5, blurred, -0.5, 0)

    # 5. Color Vibrancy Boost
    a = cv2.addWeighted(a, 1.1, np.zeros_like(a), 0, 10)
    b = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 10)

    # Merge channels
    enhanced_lab = cv2.merge([l, a, b])
    final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 6. Border Cleanup (Remove stitching artifacts)
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    final = cv2.bitwise_and(final, final, mask=clean_mask)

    return final


def process_averaged_stitching(bevs: list) -> np.ndarray:
    """
    Complete processing for averaged BEV stitching
    Args:
        bevs: List of input BEV images
    Returns:
        Enhanced final image
    """
    # 1. Create weighted average (better than simple sum)
    weights = [create_weight_map(bev.shape[:2]) for bev in bevs]
    weighted_sum = np.zeros_like(bevs[0], dtype=np.float32)

    for bev, weight in zip(bevs, weights):
        weighted_sum += bev.astype(np.float32) * weight

    # Normalize
    stitched = (weighted_sum / np.sum(weights, axis=0)).astype(np.uint8)

    # 2. Post-processing
    return post_process_stitched(stitched, bevs)


def create_weight_map(shape: tuple) -> np.ndarray:
    """
    Creates feathering weights with 3 channels for direct multiplication
    """
    h, w = shape[:2]  # Get height and width (ignore channels if present)
    weights = np.ones((h, w), dtype=np.float32)

    # Feather edges (50px gradient)
    feather_size = 50
    ramp = np.linspace(0, 1, feather_size)

    # Left/right edges
    weights[:, :feather_size] *= ramp
    weights[:, -feather_size:] *= ramp[::-1]

    # Top/bottom edges
    weights[:feather_size, :] *= ramp.reshape(-1, 1)
    weights[-feather_size:, :] *= ramp[::-1].reshape(-1, 1)

    # Convert to 3-channel by broadcasting
    return np.dstack([weights] * 3)  # Shape becomes (h,w,3)


def main():
    im2 = cv2.imread(
        "/home/william/Codes/vp/data/zed_360/1752046415086918221.png",
        cv2.IMREAD_GRAYSCALE,
    )
    im2_raw = cv2.imread("/home/william/Codes/vp/data/zed_360/1752046415086918221.png")

    at_detector = AprilGridDetector()
    ret = at_detector.detect(im2)
    rvec, tvec = at_detector.estimate_pose(ret, ZED_K, ZED_DIST, id_end=23)

    R_c_b, _ = cv2.Rodrigues(rvec)

    ipm = IPM(ZED_K, ZED_DIST, (1280, 720), R_c_b=R_c_b, t_c_b=tvec)
    ipm_info = IPMInfo()
    ipm_info.x_scale = 1000
    ipm_info.y_scale = 1000

    ipm_img = ipm.GetIPMImage(im2_raw, ipm_info, R_c_b)
    cv2.imshow("ipm_img", ipm_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "zed_360"
    image_names = [
        "1752046398086179221.png",
        "1752046409920110221.png",
        "1752046415086918221.png",
    ]
    image_paths = [str(data_dir / name) for name in image_names]

    result = stitch(image_paths, base_image_index=1)
