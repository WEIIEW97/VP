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


def main():
    root_dir = "/home/william/extdisk/data/calib/failed/20251028/"
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

        img_path = dir / "RGB"
        # find the .png file in img_path
        img_files = [
            f for f in img_path.iterdir() if f.is_file() and f.suffix == ".png"
        ]
        for img_file in img_files:
            print(f"Processing image: {img_file}, fish eye mode {flag_is_fisheye}")
            recalib_file = str(img_file).replace(".png", "_recalib.jpg")
            rgb = cv2.imread(str(img_file))
            mask = np.zeros_like(rgb, dtype=np.uint8)
            mask[314:932, 488:1366] = rgb[314:932, 488:1366]
            angles, img = chessboard_detect(
                mask,
                K,
                dist_coef,
                pattern_size=(6, 3),
                square_size=0.025,
                is_fisheye=flag_is_fisheye,
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
            plt.figure()
            plt.imshow(corrected_im)
            plt.savefig(str(recalib_file))
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    plt.close()
    print(f"failed rate is {failed/total*100:.4f}%")
    print("done")

    print(f"Passed files are: ")
    for file in success_files:
        print(file)


def test_single():
    img_path = "/home/william/extdisk/data/calib/failed/20251028/Z0CABLB25IRA0061#BA.04.00.0069.01-nodetection/RGB/recheck_image.png"
    intri_path = "/home/william/extdisk/data/calib/failed/20251028/Z0CABLB25IRA0061#BA.04.00.0069.01-nodetection/result/RGB.yaml"
    K, dist_coef, flag_is_fisheye = load_yaml(str(intri_path))
    print(f"K is: {K}")
    print(f"dist_coef is: {dist_coef}")
    print(f"flag_is_fisheye is: {flag_is_fisheye}")
    rgb = cv2.imread(str(img_path))
    mask = np.zeros_like(rgb, dtype=np.uint8)
    mask[314:832, 488:1366] = rgb[314:832, 488:1366]
    angles, img = chessboard_detect(
        mask,
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
    print(f"detect angle offset is: {angles}")
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
    # main()
    test_single()