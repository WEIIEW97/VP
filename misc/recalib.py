import cv2
import numpy as np
import math
import json
import matplotlib.pyplot as plt

from pathlib import Path


def load_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def convert_yuv_to_rgb(yuv_path: str, h: int, w: int):
    yuv_data = np.fromfile(yuv_path, dtype=np.uint8)
    yuv = yuv_data.reshape((h * 3 // 2, w))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb


def chessboard_detect(rgb, K, dist_coef, pattern_size=(8, 5), square_size=0.025):
    """
    Detect chessboard and estimate its 3D pose using solvePnP

    Args:
        image_path: Path to the input image
        pattern_size: Tuple of (inner corners per row, inner corners per column)
        square_size: Size of one chessboard square in meters (for real-world scale)

    Returns:
        angles: Rotation angles in degrees (pitch, yaw, roll)
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
    ret, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not ret:
        return None, img

    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Solve PnP to get rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist_coef)

    if not success:
        return None, img

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles (pitch, yaw, roll)
    pitch = math.degrees(
        math.atan2(-rmat[2, 0], math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
    )
    yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))

    # Draw detected corners and 3D axes
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)

    # Draw coordinate axes (3D axes projected to 2D)
    axis_length = 2 * square_size
    axis_points = np.float32(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]
    ).reshape(-1, 3)

    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coef)
    img_points = np.int32(img_points).reshape(-1, 2)

    # Draw axes (X: red, Y: green, Z: blue)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 3)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 3)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 3)

    # Annotate image with angles
    cv2.putText(
        img,
        f"Pitch: {pitch:.2f} degrees",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"Yaw: {yaw:.2f} degrees",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"Roll: {roll:.2f} degrees",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return (pitch, yaw, roll), img


def recalib(im: np.ndarray, intri: dict, rotate_angle: float, crop: bool = False):
    K = np.array(intri["cam_intrinsic"]).reshape((3, 3))
    cx = K[0, 2]
    cy = K[1, 2]
    roll = rotate_angle

    h, w = im.shape[:2]
    # undistort_im = cv2.undistort(im, K, dist_coefs)

    center = (cx, cy)
    rotation_matrix = cv2.getRotationMatrix2D(center, roll, 1.0)
    corrected_im = cv2.warpAffine(
        im,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # add an option to crop the image with the maximum inner rectangle with the valid pixels(avoid black border after rotation)
    if crop:
        gray = cv2.cvtColor(corrected_im, cv2.COLOR_BGR2GRAY)
        # max_rect = find_max_inscribed_rect(gray)
        top, bottom, left, right = find_max_inner_rect(corrected_im)
        # Crop the image to remove black borders
        # [x, y, width, height]
        cropped_im = corrected_im[
           top:bottom+1, left:right+1
        ]
        # Resize back to original dimensions
        final_im = cv2.resize(cropped_im, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return final_im

    return corrected_im


def find_max_inscribed_rect1(gray):
    """Find the maximum inscribed rectangle containing only valid (non-black) pixels"""
    # Create mask of valid pixels and convert to uint8
    mask = (gray > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [0, 0, gray.shape[1], gray.shape[0]]  # Return full image if no contours

    # Find the largest contour
    max_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle [x, y, width, height]
    x, y, w, h = cv2.boundingRect(max_contour)

    return [x, y, w, h]


def max_rect_in_hist(hist):
    # 实现直方图最大矩形算法（返回left, right, height, area）
    stack = []
    max_area = 0
    max_rect = (0, 0, 0, 0)
    n = len(hist)
    for idx in range(n):
        start = idx
        while stack and stack[-1][1] > hist[idx]:
            i, h = stack.pop()
            area = h * (idx - i)
            if area > max_area:
                max_area = area
                max_rect = (i, idx, h, area)
            start = i
        stack.append((start, hist[idx]))
    while stack:
        i, h = stack.pop()
        area = h * (n - i)
        if area > max_area:
            max_area = area
            max_rect = (i, n, h, area)
    return max_rect


def find_max_inner_rect(R):
    # 二值化：非黑色区域为1
    binary = np.all(R != [0, 0, 0], axis=-1).astype(int)
    H, W = binary.shape
    dp = np.zeros((H, W), dtype=int)
    # 计算dp表
    for i in range(H):
        for j in range(W):
            if binary[i][j] == 1:
                dp[i][j] = dp[i - 1][j] + 1 if i > 0 else 1
            else:
                dp[i][j] = 0
    max_rect = (0, 0, 0, 0, 0)  # top, bottom, left, right, area
    for i in range(H):
        hist = dp[i]
        left, right, height, area = max_rect_in_hist(hist)
        if area > max_rect[4]:
            top = i - height + 1
            bottom = i
            left = left
            right = right - 1  # 因为right是开区间
            max_rect = (top, bottom, left, right, area)
    return max_rect[:4]  # 返回 (top, bottom, left, right)


def main():
    root_dir = "/home/william/extdisk/data/calib/image_save"
    # retrieve all directories in root_dir if begin with "abonr"
    dirs = [
        d for d in Path(root_dir).iterdir() if d.is_dir() and d.name.startswith("abnor")
    ]
    for dir in dirs:
        intri_path = (
            Path(root_dir)
            / "calibration_intrinsic"
            / dir.name
            / "result/intrinsics_colin.json"
        )
        intri = load_json(intri_path)
        img_path = dir / "RGB"
        # find the .png file in img_path
        img_files = [
            f for f in img_path.iterdir() if f.is_file() and f.suffix == ".png"
        ]
        for img_file in img_files:
            recalib_file = str(img_file).replace(".png", "_recalib.png")
            img = cv2.imread(str(img_file))
            angles, img = chessboard_detect(
                img,
                np.array(intri["cam_intrinsic"]).reshape((3, 3)),
                np.array(intri["cam_distcoeffs"]),
                pattern_size=(6, 3),
                square_size=0.025,
            )
            print(f"detect angle offset is: {angles}")
            corrected_im = recalib(img, intri, angles[1], True)
            # cv2.imwrite(str(recalib_file), corrected_im)
            plt.figure()
            plt.imshow(corrected_im)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    print("done")


if __name__ == "__main__":
    # yuv_path = "/home/william/Codes/vp/data/recalib/calib.yuv"
    # intri_path = "/home/william/Codes/vp/data/recalib/intrinsics.json"
    # extri_path = "/home/william/Codes/vp/data/recalib/extrinsics.json"

    # intri = load_json(intri_path)
    # extri = load_json(extri_path)

    # rgb = convert_yuv_to_rgb(yuv_path, 1080, 1920)
    # plt.figure()
    # plt.imshow(rgb)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    # angles, corners = chessboard_detect(
    #     rgb,
    #     np.array(intri["cam_intrinsic"]).reshape((3, 3)),
    #     np.array(intri["cam_distcoeffs"]),
    # )
    # print(f"detect angle offset is: {angles}")
    # plt.figure()
    # plt.imshow(corners)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    # corrected_im = recalib(rgb, intri, angles[1])
    # plt.figure()
    # plt.imshow(corrected_im)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
    main()
