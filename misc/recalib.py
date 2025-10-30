import json
import math

import cv2
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def ypr2r(ypr: np.ndarray | list):
    y, p, r = np.deg2rad(ypr)
    Rz = [math.cos(y), -math.sin(y), 0, math.sin(y), math.cos(y), 0, 0, 0, 1]
    Ry = [math.cos(p), 0, math.sin(p), 0, 1, 0, -math.sin(p), 0, math.cos(p)]
    Rx = [1, 0, 0, 0, math.cos(r), -math.sin(r), 0, math.sin(r), math.cos(r)]

    Rz = np.array(Rz).reshape((3, 3))
    Ry = np.array(Ry).reshape((3, 3))
    Rx = np.array(Rx).reshape((3, 3))

    return Rz @ Ry @ Rx


def r2ypr(R: np.ndarray):
    n = R[:, 0]
    o = R[:, 1]
    a = R[:, 2]

    y = np.arctan2(n[1], n[0])
    p = np.arctan2(-n[2], n[0] * np.cos(y) + n[1] * np.sin(y))
    r = np.arctan2(
        a[0] * np.sin(y) - a[1] * np.cos(y), -o[0] * np.sin(y) + o[1] * np.cos(y)
    )
    ypr = np.array([y, p, r])
    return np.rad2deg(ypr)


def load_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        lines = f.readlines()

    # Drop the first line if it starts with "%YAML"
    if lines[0].startswith("%YAML"):
        lines = lines[1:]

    data = yaml.safe_load("".join(lines))
    return data


def load_yaml(yaml_path: str):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    model_type = fs.getNode("model_type").string()
    K = None
    dist_coef = None
    is_fisheye = None
    if model_type == "PINHOLE_FULL":
        fx = fs.getNode("projection_parameters").getNode("fx").real()
        fy = fs.getNode("projection_parameters").getNode("fy").real()
        cx = fs.getNode("projection_parameters").getNode("cx").real()
        cy = fs.getNode("projection_parameters").getNode("cy").real()

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        dist = []
        for key in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]:
            dist.append(fs.getNode("distortion_parameters").getNode(key).real())

        dist_coef = np.array(dist, dtype=np.float64).reshape(-1, 1)
        is_fisheye = False

    elif model_type == "KANNALA_BRANDT":
        mu = fs.getNode("projection_parameters").getNode("mu").real()
        mv = fs.getNode("projection_parameters").getNode("mv").real()
        u0 = fs.getNode("projection_parameters").getNode("u0").real()
        v0 = fs.getNode("projection_parameters").getNode("v0").real()

        K = np.array([[mu, 0, u0], [0, mv, v0], [0, 0, 1]], dtype=np.float64)

        # Distortion (OpenCV fisheye expects [k1, k2, k3, k4])
        dist = []
        for key in ["k2", "k3", "k4", "k5"]:
            dist.append(fs.getNode("projection_parameters").getNode(key).real())

        dist_coef = np.array(dist, dtype=np.float64).reshape(-1, 1)
        is_fisheye = True

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    fs.release()
    return K, dist_coef, is_fisheye


def convert_yuv_to_rgb(yuv_path: str, h: int, w: int):
    yuv_data = np.fromfile(yuv_path, dtype=np.uint8)
    yuv = yuv_data.reshape((h * 3 // 2, w))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb


def mask_aruco(im: np.ndarray, rois: np.ndarray):
    """
    Extract a bounding box from ArUco markers positioned at the edges of an image.

    Takes ROI bounding boxes and finds the innermost edges to create a rectangular mask.
    Assumes 4 markers are positioned at left, right, top, and bottom edges.

    Args:
        im: Input image (H, W) or (H, W, C)
        rois: ROI corners with shape (N, 4, 2), where each ROI has 4 corner points

    Returns:
        mask: Binary mask with the extracted region set to image values
    """
    mask = np.zeros_like(im, dtype=np.uint8)
    # Compute centers for all ROIs (average of 4 corner points)
    centers = np.mean(rois, axis=1)

    # Identify ROI indices by position using centers
    left_idx = np.argmin(centers[:, 0])  # leftmost center
    right_idx = np.argmax(centers[:, 0])  # rightmost center
    top_idx = np.argmin(centers[:, 1])  # topmost center
    bottom_idx = np.argmax(centers[:, 1])  # bottommost center

    # Extract corner points from each marker
    # Left marker: take rightmost x coordinate (innermost edge)
    left_roi = rois[left_idx]
    rightmost_x = np.max(left_roi[:, 0])
    left_corner = np.array([rightmost_x, centers[left_idx, 1]])

    # Right marker: take leftmost x coordinate (innermost edge)
    right_roi = rois[right_idx]
    leftmost_x = np.min(right_roi[:, 0])
    right_corner = np.array([leftmost_x, centers[right_idx, 1]])

    # Top marker: take bottommost y coordinate (innermost edge)
    top_roi = rois[top_idx]
    bottommost_y = np.max(top_roi[:, 1])
    top_corner = np.array([centers[top_idx, 0], bottommost_y])

    # Bottom marker: take topmost y coordinate (innermost edge)
    bottom_roi = rois[bottom_idx]
    topmost_y = np.min(bottom_roi[:, 1])
    bottom_corner = np.array([centers[bottom_idx, 0], topmost_y])

    # Compute bounding box from the four corner points
    corners = np.array(
        [left_corner, right_corner, top_corner, bottom_corner], dtype=np.int32
    )
    x_min, y_min = np.min(corners, axis=0)
    x_max, y_max = np.max(corners, axis=0)

    # Copy the region to the mask
    mask[y_min:y_max, x_min:x_max] = im[y_min:y_max, x_min:x_max]
    region = (x_min, y_min, x_max, y_max)
    return mask, region


def margin_marching(region: tuple, stride: int = 4):
    x_min, y_min, x_max, y_max = region
    # shrink region inward by stride while keeping order (x_min, y_min, x_max, y_max)
    return (x_min + stride, y_min + stride, x_max - stride, y_max - stride)


def chessboard_detect(
    rgb, K, dist_coef, pattern_size=(8, 5), square_size=0.08, is_fisheye=False
):
    """
    Detect chessboard and estimate its 3D pose using solvePnP

    Args:
        image_path: Path to the input image
        pattern_size: Tuple of (inner corners per row, inner corners per column)
        square_size: Size of one chessboard square in meters (for real-world scale)

    Returns:
        angles: Rotation angles in degrees (yaw, pitch, roll)
        annotated_img: Image with detected corners and axis visualization
    """
    # Read image
    img = np.copy(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prepare 3D object points (0,0,0), (1,0,0), ..., (6,6,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size
    )

    # Find chessboard corners
    corners_found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )

    if not corners_found:
        print("Chessboard corners not found.")
        return None, img

    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Solve PnP to get rotation and translation vectors
    if is_fisheye:
        success, rvec, tvec = cv2.fisheye.solvePnP(objp, corners, K, dist_coef)
    else:
        success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist_coef)

    if not success:
        print("solvePnP failed.")
        return None, img

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles (yaw, pitch, roll)
    ypr = r2ypr(rmat)

    # Draw detected corners and 3D axes
    cv2.drawChessboardCorners(img, pattern_size, corners, corners_found)

    # Draw coordinate axes (3D axes projected to 2D)
    axis_length = 2 * square_size
    axis_points = np.float32(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]
    ).reshape(-1, 3)

    if is_fisheye:
        axis_points = axis_points.astype(np.float32)
        axis_points = axis_points.reshape(-1, 1, 3)
        img_points, _ = cv2.fisheye.projectPoints(axis_points, rvec, tvec, K, dist_coef)
    else:
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coef)
    img_points = np.int32(img_points).reshape(-1, 2)

    # Draw axes (X: red, Y: green, Z: blue)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 3)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 3)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 3)

    # Annotate image with angles
    cv2.putText(
        img,
        f"Yaw: {ypr[0]:.2f} degrees",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"Pitch: {ypr[1]:.2f} degrees",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"Roll: {ypr[2]:.2f} degrees",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return ypr, img


def adpative_chessboard_detect(
    rgb: np.ndarray,
    K: np.ndarray,
    dist_coef: np.ndarray,
    region: tuple,
    pattern_size: tuple = (8, 5),
    square_size: float = 0.08,
    is_fisheye: bool = False,
    max_attempts: int = 12,
):
    img = np.copy(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize ROI and validate bounds
    region = (
        max(0, region[0]),
        max(0, region[1]),
        min(gray.shape[1], region[2]),
        min(gray.shape[0], region[3]),
    )

    valid_gray = gray[region[1] : region[3], region[0] : region[2]]

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size
    )

    corners_found = False
    corners = None

    # Guard against infinite expansion attempts
    attempts = 0
    while not corners_found and attempts < max_attempts:

        _corners_found, _corners = cv2.findChessboardCorners(
            valid_gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        if _corners_found:
            corners = _corners
            corners_found = True
            break
        # Expand search region
        prev_region = region
        region = margin_marching(region)
        print(f"shrinkage region is: {region}")
        if region == prev_region:
            break
        if region[0] >= region[2] or region[1] >= region[3]:
            break
        valid_gray = gray[region[1] : region[3], region[0] : region[2]]
        attempts += 1

    if corners is None:
        print("Chessboard corners not found.")
        return None, img

    # Refine corner locations in ROI, then convert to full-image coordinates
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(valid_gray, corners, (11, 11), (-1, -1), criteria)
    # # Offset corners by ROI origin to match full image coordinates
    offset = np.array([[[region[0], region[1]]]], dtype=corners.dtype)
    corners_full = corners + offset

    if is_fisheye:
        objp_f = objp.reshape(-1, 1, 3).astype(np.float64)
        corners_f = corners_full.astype(np.float64)
        # corners = corners.astype(np.float64)
        success, rvec, tvec = cv2.fisheye.solvePnP(objp_f, corners_f, K, dist_coef)
        # success, rvec, tvec = cv2.fisheye.solvePnP(objp_f, corners, K, dist_coef)

    else:
        success, rvec, tvec = cv2.solvePnP(objp, corners_full, K, dist_coef)
        # success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist_coef)

    if not success:
        print("solvePnP failed.")
        return None, img

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles (yaw, pitch, roll)
    ypr = r2ypr(rmat)

    # Draw detected corners and 3D axes (use full-image coordinates)
    cv2.drawChessboardCorners(img, pattern_size, corners_full, corners_found)
    cv2.drawChessboardCorners(img, pattern_size, corners, corners_found)
    # Draw coordinate axes (3D axes projected to 2D)
    axis_length = 2 * square_size
    axis_points = np.float32(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]
    ).reshape(-1, 3)

    if is_fisheye:
        axis_points = axis_points.astype(np.float32)
        axis_points = axis_points.reshape(-1, 1, 3)
        img_points, _ = cv2.fisheye.projectPoints(axis_points, rvec, tvec, K, dist_coef)
    else:
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coef)
    img_points = np.int32(img_points).reshape(-1, 2)

    # Draw axes (X: red, Y: green, Z: blue)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 3)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 3)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 3)

    # Annotate image with angles
    cv2.putText(
        img,
        f"Yaw: {ypr[0]:.2f} degrees",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"Pitch: {ypr[1]:.2f} degrees",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"Roll: {ypr[2]:.2f} degrees",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return ypr, img


def recalib(im: np.ndarray, K: np.ndarray, rotation_angles: tuple, crop: bool = False):
    """
    Apply 3D rotation (pitch, yaw, roll) to image using perspective transformation.

    Args:
        im: Input image
        intri: Camera intrinsic parameters dictionary
        rotation_angles: Tuple of (pitch, yaw, roll) angles in degrees
        crop: Whether to crop black borders after rotation

    Returns:
        Rotated image
    """
    ypr = rotation_angles

    h, w = im.shape[:2]

    R = ypr2r(ypr)

    # Create homography matrix for perspective transformation
    # H = K * R^(-1) * K^(-1)  # Correct formula for perspective transformation
    H = K @ np.linalg.inv(R) @ np.linalg.inv(K)

    # Apply perspective transformation
    corrected_im = cv2.warpPerspective(
        im,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return corrected_im


rois = np.array(
    [
        [
            [1400.041, 739.371],
            [1401.901, 432.779],
            [1787.924, 435.121],
            [1786.064, 741.713],
        ],
        [
            [732.485, 61.231],
            [1196.261, 60.705],
            [1196.500, 270.821],
            [732.723, 271.347],
        ],
        [
            [689.009, 868.880],
            [1196.861, 867.400],
            [1197.337, 1030.711],
            [689.484, 1032.191],
        ],
        [
            [107.398, 457.852],
            [470.870, 447.373],
            [478.764, 721.162],
            [115.292, 731.641],
        ],
    ]
)


def test_patch():
    root_dir = "/home/william/extdisk/data/calib/failed/20251030"
    dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
    failed = 0
    total = 0
    success_files = []

    for dir in dirs:
        intri_path = Path(root_dir) / dir.name / "result" / "RGB.yaml"
        K, dist_coef, flag_is_fisheye = load_yaml(str(intri_path))
        # intri_path = Path(root_dir).parent / "intrinsics" / f"intrinsics-{dir.name.replace("#", "")}.json"
        # intri_data = load_json(str(intri_path))["camera_para"]
        # K = np.array(intri_data["cam_intrinsic"], dtype=np.float32).reshape((3, 3))
        # dist_coef = np.array(intri_data["cam_distcoeffs"], dtype=np.float32)
        # flag_is_fisheye = False

        # KK = np.copy(K)
        K[:2, :] = K[:2, :] * 0.5 

        img_path = dir / "RGB"
        # find the .png file in img_path
        img_files = [
            f for f in img_path.iterdir() if f.is_file() and f.suffix == ".png"
        ]
        for img_file in img_files:
            print(f"Processing image: {img_file}, fish eye mode {flag_is_fisheye}")
            # recalib_file = str(img_file).replace(".png", "_recalib.jpg")
            rgb = cv2.imread(str(img_file))
            rgb = cv2.resize(rgb, fx=0.5, fy=0.5, dsize=None)
            scale = 0.5
            rois_scaled = (rois * scale).astype(np.float32)
            mask, region = mask_aruco(rgb, rois_scaled)
            print(f"initial region is: {region}")
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # binarize this mask
            # mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            # rgb = cv2.resize(rgb, fx=0.5, fy=0.5, dsize=None)
            # plt.figure()
            # plt.imshow(mask)
            # plt.axis("off")
            # plt.tight_layout()
            # plt.show()
            # cv2.imwrite(str(img_file).replace(".png", "_mask.jpg"), mask)
            angles, img = adpative_chessboard_detect(
                rgb,
                K,
                dist_coef,
                region,
                pattern_size=(6, 3),
                square_size=0.08,
                is_fisheye=flag_is_fisheye,
                max_attempts=20,
            )
            print(f"detect angle offset is: {angles}")
            if angles is None:
                failed += 1
                total += 1
                continue
            if np.any(np.abs(angles) >= 1.5):
                failed += 1
                total += 1
                print(f"Failed to detect chessboard in image: {img_file}")
                print("Due to the threshold of 1.5 degrees")
                continue
            total += 1
            success_files.append(str(img_file))
            corrected_im = recalib(img, K, angles, False)
            # cv2.imwrite(str(recalib_file), corrected_im)
            # plt.figure()
            # plt.imshow(corrected_im)
            # plt.savefig(str(recalib_file))
            # plt.axis("off")
            # plt.tight_layout()
            # plt.show()

    plt.close()
    print(f"failed rate is {failed/total*100:.4f}%")
    print("done")

    print(f"Passed files are: ")
    for file in success_files:
        print(file)


def test_single():
    img_path = "/home/william/extdisk/data/calib/failed/20251030/Z0CABLB25IRA0066#BA.04.00.0069.01-nodetection/RGB/rgbvi-2025-10-30-10-50-42.png"
    intri_path = "/home/william/extdisk/data/calib/failed/20251030/Z0CABLB25IRA0066#BA.04.00.0069.01-nodetection/result/RGB.yaml"
    K, dist_coef, flag_is_fisheye = load_yaml(str(intri_path))
    print(f"K is: {K}")
    print(f"dist_coef is: {dist_coef}")
    print(f"flag_is_fisheye is: {flag_is_fisheye}")
    rgb = cv2.imread(str(img_path))
    rgb = cv2.resize(rgb, fx=0.5, fy=0.5, dsize=None)
    # mask = mask_aruco(rgb, rois)
    K[:2, :] = K[:2, :] * 0.5

    angles, img = chessboard_detect(
        rgb,
        K,
        dist_coef,
        pattern_size=(6, 3),
        square_size=0.08,
        is_fisheye=flag_is_fisheye,
    )
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    if angles is None:
        print("Failed to detect chessboard in image.")
        return
    if np.any(np.abs(angles) >= 1.5):
        print("Exceed the offset threshold of 1.5 degrees.")
        return
    print(f"detect angle offset is: {angles}")
    corrected_im = recalib(img, K, angles, False)
    plt.figure()
    plt.imshow(corrected_im)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_patch()
    # test_single()
