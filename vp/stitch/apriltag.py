import aprilgrid.detection
import numpy as np
import cv2
import aprilgrid
from pathlib import Path
from typing import List

ZED_K = np.array([700.0400, 0, 630.6500, 0, 700.0400, 331.5925, 0, 0, 1]).reshape(
    (3, 3)
)

ZED_DIST = np.array([-0.1724, 0.0270, 0.0024, 0.0003, 0, 0, 0, 0])


class AprilGridDetector:
    def __init__(
        self,
        tag_family_name="t36h11",
        grid_size=(4, 6),
        tag_size=0.04,
        tag_spacing=0.012,
    ):
        # tag_family_name can be "t36h11", "t36h11b1"
        self.detector = aprilgrid.Detector(tag_family_name=tag_family_name)
        self.grid_size = grid_size
        self.tag_size = tag_size
        self.tag_spacing = tag_spacing

    def detect(self, image: np.ndarray):
        return self.detector.detect(image)

    def estimate_pose(
        self,
        detections: List[aprilgrid.detection.Detection],
        K: np.ndarray,
        dist_coeff: np.ndarray,
    ):
        """
        Estimate camera pose relative to AprilGrid
        :param corners: Detected marker corners
        :param ids: Detected marker IDs
        :param camera_matrix: Camera intrinsic matrix
        :param dist_coeffs: Camera distortion coefficients
        :return: rvec, tvec (rotation and translation vectors)
        """
        obj_points = []
        img_points = []

        for i in range(len(detections)):
            marker_id = detections[i].tag_id
            corners = detections[i].corners
            row = marker_id // self.grid_size[1]
            col = marker_id % self.grid_size[1]

            tag_half = self.tag_size / 2.0
            spacing = self.tag_spacing

            x_start = col * (self.tag_size + spacing)
            y_start = row * (self.tag_size + spacing)

            marker_corners = np.array(
                [
                    [x_start - tag_half, y_start - tag_half, 0],
                    [x_start + tag_half, y_start - tag_half, 0],
                    [x_start + tag_half, y_start + tag_half, 0],
                    [x_start - tag_half, y_start + tag_half, 0],
                ],
                dtype=np.float32,
            )

            obj_points.append(marker_corners)
            img_points.append(corners)

        ret, rvec, tvec = cv2.solvePnP(
            np.concatenate(obj_points, axis=0),
            np.concatenate(img_points, axis=0),
            K,
            dist_coeff,
        )
        return rvec, tvec


def main():
    detector = AprilGridDetector()
    
    data_path = Path("/home/william/Codes/vp/data/zed_360")
    image_paths = sorted(data_path.glob("*.png"))
    
    # First pass: collect all poses
    poses = []
    for image_path in image_paths:
        print(f"Processing {image_path.name}")
        im = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        im_raw = cv2.imread(str(image_path))
        ret = detector.detect(im)
        
        if len(ret) < 2:
            print(f"Skipping {image_path.name} - not enough markers detected")
            continue
            
        rvec, tvec = detector.estimate_pose(ret, ZED_K, ZED_DIST)
        poses.append((im_raw, rvec, tvec))
    

if __name__ == "__main__":
    main()