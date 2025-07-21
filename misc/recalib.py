import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def load_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def convert_yuv_to_rgb(yuv_path: str, h: int, w: int):
    yuv_data = np.fromfile(yuv_path, dtype=np.uint8)
    yuv = yuv_data.reshape((h * 3 // 2, w))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb


def recalib(im: np.ndarray, intri: dict, extri: dict):
    K = np.array(intri["cam_intrinsic"]).reshape((3, 3))
    cx = K[0, 2]
    cy = K[1, 2]
    dist_coefs = np.array(intri["cam_distcoeffs"])
    roll = extri["cam_pose"][2]

    h, w = im.shape[:2]
    # undistort_im = cv2.undistort(im, K, dist_coefs)

    center = (cx, cy)
    rotation_matrix = cv2.getRotationMatrix2D(center, -roll, 1.0)
    corrected_im = cv2.warpAffine(
        im,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return corrected_im


if __name__ == "__main__":
    yuv_path = "/home/william/Codes/vp/data/recalib/calib.yuv"
    intri_path = "/home/william/Codes/vp/data/recalib/intrinsics.json"
    extri_path = "/home/william/Codes/vp/data/recalib/extrinsics.json"

    intri = load_json(intri_path)
    extri = load_json(extri_path)

    rgb = convert_yuv_to_rgb(yuv_path, 1080, 1920)
    plt.figure()
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    corrected_im = recalib(rgb, intri, extri)
    plt.figure()
    plt.imshow(corrected_im)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
