import os, cv2, shutil, psutil
import torch # need to import torch before onnxruntime to make sure the cuda and cudnn are initialized correctly
import onnxruntime as ort
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import json

from dataclasses import dataclass, field
from typing import List


psutil.Process(os.getpid()).cpu_affinity([0, 1, 2, 3, 4, 5])  # Bind to large cores

NUM_OF_GRID = 100
NUM_OF_ROW = 56
NUM_OF_COL = 41
NUM_OF_LANE = 2


@dataclass
class LaneParams:
    """Configuration parameters for lane detection system"""

    onnx_model: str = (
        "/home/william/Codes/vp/onnx/Tusimple_fdmobilenet_20250715_qat_int8_base.onnx"  # Model path
    )
    windows_size: int = 10  # Sliding window size
    min_roll: float = 87.0  # Minimum roll value
    max_roll: float = 93.0  # Maximum roll value
    step_max: int = 20  # Maximum iteration step size
    kalman_std_weight_pos: float = 0.1  # Kalman filter coefficient
    kalman_std_weight_vel: float = 0.01  # Kalman filter coefficient
    kalman_R: float = 10.0  # Kalman filter coefficient
    std_threshold: float = 0.4  # Standard deviation threshold
    FPS: int = 10  # Valid frame selection range
    max_step_deg: float = 0.05  # Maximum change per frame (0.05 degrees)
    lane_detector_confidence: float = 0.1  # Lane detector confidence threshold
    interval_lane: int = (
        14  # Lane detection interval (5 frames at 30fps: 30/(14+1) = 2 fps)
    )
    crop_small_mode: int = 0
    Ufldv2_Exist_Col: bool = (
        False  # Default is False (no train track detection) - options: False/True
    )
    crop_tgt: List[int] = field(
        default_factory=lambda: [0, 0, 1920, 1080]
    )  # Crop target area
    resize_tgt: List[int] = field(
        default_factory=lambda: [672, 384]
    )  # Resize target dimensions
    ori_size: List[int] = field(
        default_factory=lambda: [1920, 1080]
    )  # Original image size

    def __post_init__(self):
        """Type conversion and validation"""
        self.Ufldv2_Exist_Col = bool(self.Ufldv2_Exist_Col)

        # Validate list parameters
        assert len(self.crop_tgt) == 4, "crop_tgt must have exactly 4 elements"
        assert len(self.resize_tgt) == 2, "resize_tgt must have exactly 2 elements"
        assert len(self.ori_size) == 2, "ori_size must have exactly 2 elements"


def softmax(x, axis=0):
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class UFLDv2:
    def __init__(self, lane_params: LaneParams):
        self.lane_params = lane_params
        self.model_path = self.lane_params.onnx_model
        self.Ufldv2_Exist_Col = self.lane_params.Ufldv2_Exist_Col
        self.session = ort.InferenceSession(
            self.model_path,
            providers=[
                "DmlExecutionProvider",
                # "CPUExecutionProvider",
                "CUDAExecutionProvider",
            ],
        )
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.num_row = 56
        self.num_col = 41

        self.row_anchor = np.linspace(160, 710, self.num_row) / 720
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def __call__(
        self,
        img,
        confidence: float = 0.5,
        original_image_width=672,
        original_image_height=384,
    ):
        return self.predict(
            image=img,
            confidence=confidence,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
        )

    def preprocess(self, image):
        # Color YUV_NV12 to RGB
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV12)
        image = (image.astype(np.float32) / 255.0 - (0.485, 0.456, 0.406)) / (
            0.229,
            0.224,
            0.225,
        )

        return image.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

    def predict(
        self,
        image,
        confidence: float = 0.5,
    ):
        original_image_width, original_image_height = (
            image.shape[1],
            image.shape[0] * 2 / 3,
        )
        # Preprocess
        tensor = self.preprocess(image)

        # Run onnxruntime session
        output = self.session.run(None, {self.input_name: tensor})
        if self.Ufldv2_Exist_Col:  # Train track exists
            loc_row = output[0][:, : NUM_OF_GRID * NUM_OF_ROW * NUM_OF_LANE].reshape(
                1, NUM_OF_GRID, NUM_OF_ROW, NUM_OF_LANE
            )
            exist_row = output[0][:, NUM_OF_GRID * NUM_OF_ROW * NUM_OF_LANE :].reshape(
                1, 2, NUM_OF_ROW, NUM_OF_LANE
            )
            loc_col = output[1][:, : NUM_OF_GRID * NUM_OF_COL * NUM_OF_LANE].reshape(
                1, NUM_OF_GRID, NUM_OF_COL, NUM_OF_LANE
            )
            exist_col = output[1][:, NUM_OF_GRID * NUM_OF_COL * NUM_OF_LANE :].reshape(
                1, 2, NUM_OF_COL, NUM_OF_LANE
            )
            # Postprocess
            points_row = self.postprocess_points(
                loc_row,
                exist_row,
                confidence=confidence,
                original_image_width=original_image_width,
                original_image_height=original_image_height,
                is_row=True,
            )
            points_col = self.postprocess_points(
                loc_col,
                exist_col,
                confidence=confidence,
                original_image_width=original_image_width,
                original_image_height=original_image_height,
                is_row=False,
            )
            # Safe merging method
            points = []
            if len(points_row) >= 2 and len(points_col) >= 2:
                points = [points_row[0] + points_col[0], points_row[1] + points_col[1]]
            return points

        else:  # Train track doesn't exist
            loc_row = output[0][:, : NUM_OF_GRID * NUM_OF_ROW * NUM_OF_LANE].reshape(
                1, NUM_OF_GRID, NUM_OF_ROW, NUM_OF_LANE
            )
            exist_row = output[0][:, NUM_OF_GRID * NUM_OF_ROW * NUM_OF_LANE :].reshape(
                1, 2, NUM_OF_ROW, NUM_OF_LANE
            )
            # Postprocess
            points = self.postprocess_points(
                loc_row,
                exist_row,
                confidence=confidence,
                original_image_width=original_image_width,
                original_image_height=original_image_height,
                is_row=True,
            )
            return points

    def postprocess_points(
        self,
        loc,
        exist,
        confidence: float = 0.5,
        row_anchor=np.linspace(160, 710, 56) / 720,
        local_width=1,
        original_image_width=1640,
        original_image_height=590,
        is_row=False,  # Parameter to choose between row lane or train track
    ):
        batch_size, num_grid, num_cls, num_lane = loc.shape

        max_indices = np.argmax(loc, axis=1)
        valid = np.argmax(softmax(exist, axis=1) > confidence, axis=1)

        lane_idx = [0, 1]
        lanes = []

        for i in lane_idx:
            tmp = []
            for k in range(valid.shape[1]):
                if valid[0, k, i]:
                    all_ind = np.array(
                        list(
                            range(
                                max(0, max_indices[0, k, i] - local_width),
                                min(
                                    num_grid - 1,
                                    max_indices[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        np.sum(
                            softmax(loc[0, all_ind, k, i], axis=0)
                            * all_ind.astype(np.float32)
                        )
                        + 0.5
                    )

                    if is_row:
                        # Row direction processing
                        x = out_tmp / (num_grid - 1) * original_image_width
                        y = row_anchor[k] * original_image_height
                        tmp.append([int(x), int(y)])
                    else:
                        # Column direction processing
                        y = out_tmp / (num_grid - 1) * original_image_height
                        x = k / (num_cls - 1) * original_image_width
                        tmp.append([int(x), int(y)])

            lanes.append(tmp)

        return lanes


class ResolutionInterface:
    def __init__(self, crop_tgt: list, resize_tgt: tuple, ori_size: tuple):
        self.crop_tgt = crop_tgt
        self.x_scale = (crop_tgt[2] - crop_tgt[0]) / resize_tgt[0]
        self.y_scale = (crop_tgt[3] - crop_tgt[1]) / resize_tgt[1]
        self.resize_tgt = resize_tgt
        self.ori_size = ori_size

    def preprocess_img(self, img: np.ndarray):
        x0, y0, x1, y1 = self.crop_tgt
        bgr = cv2.resize(img[y0:y1, x0:x1, :], self.resize_tgt)
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        u = cv2.resize(u, (int(u.shape[1] / 2), int(u.shape[0] / 2)))
        v = cv2.resize(v, (int(v.shape[1] / 2), int(v.shape[0] / 2)))
        uv = np.column_stack((u.flatten(), v.flatten())).reshape(
            int(y.shape[0] / 2), -1
        )
        yuv_nv12 = np.concatenate((y, uv), axis=0)
        return yuv_nv12

    def postprocess_lane(self, lanes):
        return [
            (
                [
                    (
                        np.clip(
                            pt[0] * self.x_scale + self.crop_tgt[0],
                            self.crop_tgt[0],
                            self.crop_tgt[2],
                        ),
                        np.clip(
                            pt[1] * self.y_scale + self.crop_tgt[1],
                            self.crop_tgt[1],
                            self.crop_tgt[3],
                        ),
                    )
                    for pt in lane
                ]
                if len(lane) > 0
                else []
            )
            for lane in lanes
        ]


def predict_single():
    pass


def predict_video():
    pass


if __name__ == "__main__":
    lane_params = LaneParams()
    resolution = ResolutionInterface(
        crop_tgt=lane_params.crop_tgt,
        resize_tgt=lane_params.resize_tgt,
        ori_size=lane_params.ori_size,
    )
    lane_detector = UFLDv2(lane_params)
    json_file = {}
    frame_count = 0

    video_path = "/home/william/extdisk/data/boximu-rgb/dataFromYF/data0731/zhuizi/20250731_135519_main.h265"
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Open Video Failed")

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        yuv_nv12 = resolution.preprocess_img(frame)
        lane_results = lane_detector.predict(
            image=yuv_nv12,
            confidence=lane_params.lane_detector_confidence,
        )
        lane_results = resolution.postprocess_lane(lane_results)
        frame_count += 1
        json_file[f"{frame_count}"] = {"lane": lane_results}
    capture.release()

    json_path = "/home/william/extdisk/data/boximu-rgb/dataFromYF/data0731/zhuizi/lane_results.json"
    with open(json_path, "w") as f:
        json.dump(json_file, f, indent=4)
    print(f"Processed {frame_count} frames and saved results to {json_path}")
