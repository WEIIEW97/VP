import numpy as np
import os
import json
import cv2
import torch

from geocalib import GeoCalib
from typing import Dict


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees."""
    return rad / torch.pi * 180


def print_calibration(results: Dict[str, torch.Tensor]) -> None:
    """Print the calibration results."""
    camera, gravity = results["camera"], results["gravity"]
    vfov = rad2deg(camera.vfov)
    roll, pitch = rad2deg(gravity.rp).unbind(-1)

    print("\nEstimated parameters (Pred):")
    print(
        f"Roll:  {roll.item():.2f}° (± {rad2deg(results['roll_uncertainty']).item():.2f})°"
    )
    print(
        f"Pitch: {pitch.item():.2f}° (± {rad2deg(results['pitch_uncertainty']).item():.2f})°"
    )
    print(
        f"vFoV:  {vfov.item():.2f}° (± {rad2deg(results['vfov_uncertainty']).item():.2f})°"
    )
    print(
        f"Focal: {camera.f[0, 1].item():.2f} px (± {results['focal_uncertainty'].item():.2f} px)"
    )

    if hasattr(camera, "k1"):
        print(f"K1:    {camera.k1.item():.1f}")


class GeoEstimator:
    def __init__(self, distort=True, prior_focal=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if distort:
            self.model = GeoCalib(weights="distorted").to(self.device)
        else:
            self.model = GeoCalib(weights="pinhole").to(self.device)
        self.prior_focal = prior_focal

    def load_image(self, image_path):
        self.im = self.model.load_image(image_path).to(self.device)

    def predict(self, verbose=True):
        if self.prior_focal is not None:
            result = self.model.calibrate(
                self.im,
                priors={"focal": torch.tensor(self.prior_focal).to(self.device)},
            )
        else:
            result = self.model.calibrate(self.im)

        if verbose:
            print_calibration(result)

        gravity = result["gravity"]
        roll, pitch = rad2deg(gravity.rp).unbind(-1)
        roll_uncertain = rad2deg(result["roll_uncertainty"])
        pitch_uncertain = rad2deg(result["pitch_uncertainty"])
        return (
            (roll - roll_uncertain).item(),
            (roll + roll_uncertain).item(),
            (pitch - pitch_uncertain).item(),
            (pitch + pitch_uncertain).item(),
        )


def read_json(path):
    return json.loads(path)


def retrieve_info(info_path):
    with open(info_path, "r") as f:
        frames_struct = f.readlines()
    return frames_struct


def retrieve_pack_info_by_frame(frames_struct, frame_id, key="lanes"):
    struct = json.loads(frames_struct[frame_id - 1])[key]
    return struct


def judge_valid(struct):
    return sum(1 for lst in struct if lst)


def judge_num_points(lst, thr=10):
    return False if len(lst) < thr else True


class VP:
    def __init__(self, K, verbose=False):
        self.K = K
        self.fx, self.cx = self.K[0, 0], self.K[0, 2]
        self.fy, self.cy = self.K[1, 1], self.K[1, 2]
        self.param_lst = []
        self.homo_lst = []
        self.vp = []
        self.line_fit_flag = True
        self.verbose = verbose

    def line_fit(self, frame_pts):
        for pts in frame_pts:
            if judge_num_points(pts):
                x = np.array([p[0] for p in pts])
                y = np.array([p[1] for p in pts])
                m, c = np.polyfit(x, y, 1)
                self.param_lst.append(np.array((m, c)))
                if self.verbose:
                    print(f"Fitted line: y = {m:.2f}x + {c:.2f}")
                self.homo_lst.append(
                    np.array((m, -1, c))
                )  # add -1 for b in homo coords

        if len(self.param_lst) < 2:
            self.line_fit_flag = False

    def compute_vp(self):
        sz = len(self.homo_lst)
        for i in range(sz - 1):
            for j in range(i + 1, sz):
                z = np.cross(self.homo_lst[i], self.homo_lst[j])
                self.vp.append(z / z[2])  # normalize along z-axis
        if self.verbose:
            print(f"VP candidates are: {self.vp}")

    def filter_candidates(self, strategy="mean"):
        vp_array = np.array(self.vp)[:, 0:2]
        if strategy == "mean":
            x_hat = np.mean(vp_array[:, 0])
            y_hat = np.mean(vp_array[:, 1])
        elif strategy == "close":
            vv = vp_array[:, 0]
            diff = np.abs(vv[:, None] - vv)
            np.fill_diagonal(diff, np.inf)
            min_diff_idx = np.unravel_index(np.argmin(diff), diff.shape)
            closest_vv = vp_array[list(min_diff_idx)]
            x_hat = np.mean(closest_vv[:, 0])
            y_hat = np.mean(closest_vv[:, 1])
        else:
            raise ValueError(f"not implemented strategy {strategy}.")
        return np.array((x_hat, y_hat))

    def estimate_yp(self, vp):
        x, y = vp  # vp shape: [1, 2]
        yaw = np.arctan((x - self.cx) / self.fx) * 180 / np.pi
        pitch = np.arctan((self.cy - y) / self.fy) * 180 / np.pi
        return yaw, pitch


def plot_res(im: np.ndarray, frame_lines, params, vanishing_point, yaw, pitch, **kwargs):

    for line in frame_lines:
        for point in line:
            x, y = point
            cv2.circle(im, (int(x), int(y)), 5, (0, 0, 255), -1)

    for m, b in params:
        x1 = 0
        y1 = int(m * x1 + b)
        x2 = int(vanishing_point[0])
        y2 = int(m * x2 + b)
        x3 = im.shape[1]
        y3 = int(m * x3 + b)
        if y3 > y2:
            y = y3
            x = x3
        else:
            y = y1
            x = x1
        cv2.line(im, (x, y), (x2, y2), (255, 0, 0), 2)

    if vanishing_point is not None:
        vx, vy = vanishing_point
        cv2.circle(im, (int(vx), int(vy)), 10, (0, 255, 0), -1)

    if yaw is not None and pitch is not None:
        text_yaw_pitch = f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)
        font_thickness = 4
        text_position = (10, 30)

        cv2.putText(
            im,
            text_yaw_pitch,
            text_position,
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    if kwargs:
        text_lines = []
        if "min_roll" in kwargs:
            text_lines.append(f"Min Roll: {kwargs['min_roll']:.2f}")
        if "max_roll" in kwargs:
            text_lines.append(f"Max Roll: {kwargs['max_roll']:.2f}")
        if "min_pitch" in kwargs:
            text_lines.append(f"Min Pitch: {kwargs['min_pitch']:.2f}")
        if "max_pitch" in kwargs:
            text_lines.append(f"Max Pitch: {kwargs['max_pitch']:.2f}")

        if text_lines:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 0, 0)
            font_thickness = 2
            text_position = (10, 70)  # Below the yaw and pitch text

            for i, text in enumerate(text_lines):
                cv2.putText(
                    im,
                    text,
                    (text_position[0], text_position[1] + i * 30),  # Move down for each line
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )

    return im


def ground2im(xy, K, pitch, yaw, h):
    """formula adpated from https://sites.google.com/site/yorkyuhuang/home/research/computer-vision-augmented-reality/ipm"""
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]
    sample_pts2 = xy
    sample_ptsr3 = np.ones((1, len(xy[1]))) * (-h)
    sample_pts3 = np.concatenate((sample_pts2, sample_ptsr3), axis=0)
    c1 = np.cos(pitch * np.pi / 180)
    s1 = np.sin(pitch * np.pi / 180)
    c2 = np.cos(yaw * np.pi / 180)
    s2 = np.sin(yaw * np.pi / 180)

    ipm = np.array(
        [
            [fx * c2 + cx * c1 + s2, cx * c1 * c2 - s2 * fx, -cx * s1],
            [s2 * (cy * c1 - fy * s1), c2 * (cy * c1 - fy * s1), -fy * c1 - cy * s1],
            [c1 * s2, c1 * c2, -s1],
        ]
    )
    sample_pts3 = ipm.dot(sample_pts3)
    sample_ptsr3 = sample_pts3[2, :]
    div = sample_ptsr3
    sample_pts3[0, :] /= div
    sample_pts3[1, :] /= div
    sample_pts2 = sample_pts3[0:2, :]
    return sample_pts2


def plot_navi_grid(im: np.ndarray, uv_grid: np.ndarray, vp=None):
    x_grid = uv_grid[0, ...]
    y_grid = uv_grid[1, ...]
    for j in range(uv_grid.shape[2]):
        cv2.line(
            im,
            (int(x_grid[0, j]), int(y_grid[0, j])),
            (int(x_grid[-1, j]), int(y_grid[-1, j])),
            (0, 0, 0),
            4,
        )
        if (j == 0 or j == uv_grid.shape[2] - 1) and vp is not None:
            cv2.line(
                im,
                (int(x_grid[0, j]), int(y_grid[0, j])),
                (int(vp[0]), int(vp[1])),
                (127, 127, 177),
                4,
                lineType=cv2.LINE_4,
            )

    for i in range(uv_grid.shape[1]):
        cv2.line(
            im,
            (int(x_grid[i, 0]), int(y_grid[i, 0])),
            (int(x_grid[i, -1]), int(y_grid[i, -1])),
            (0, 0, 0),
            4,
        )

    return im


def plot_navi_grid_fix(im: np.ndarray, uv_grid: np.ndarray, vp=None):
    x_grid = uv_grid[0, ...]
    y_grid = uv_grid[1, ...]
    for j in range(uv_grid.shape[2]):
        cv2.line(
            im,
            (int(x_grid[0, j]), int(y_grid[0, j])),
            (int(x_grid[-1, j]), int(y_grid[-1, j])),
            (177, 177, 177),
            4,
        )
        if (j == 0 or j == uv_grid.shape[2] - 1) and vp is not None:
            cv2.line(
                im,
                (int(x_grid[0, j]), int(y_grid[0, j])),
                (int(vp[0]), int(vp[1])),
                (127, 127, 177),
                4,
                lineType=cv2.LINE_4,
            )

    for i in range(uv_grid.shape[1]):
        cv2.line(
            im,
            (int(x_grid[i, 0]), int(y_grid[i, 0])),
            (int(x_grid[i, -1]), int(y_grid[i, -1])),
            (177, 177, 127),
            4,
        )

    return im


if __name__ == "__main__":
    info_path = "/home/william/extdisk/data/Lane_Detection_Result/20250219/19700101_002523_main.json"
    image_path = (
        "/home/william/extdisk/data/Lane_Detection_Result/frames/19700101_002021_main"
    )
    vis_path = "/home/william/extdisk/data/Lane_Detection_Result/frames/vis_navi_close+geocalib"
    os.makedirs(vis_path, exist_ok=True)

    K = np.array(
        [1033.788708, 0, 916.010200, 0, 1033.780937, 522.486183, 0, 0, 1]
    ).reshape((3, 3))

    sample_num = 100
    total_info = retrieve_info(info_path)

    x_sample = np.arange(-1, 1.25, 0.25)
    y_sample = np.arange(2, 15, 2)

    x_coord, y_coord = np.meshgrid(x_sample, y_sample)
    xyN_grid = np.array((x_coord, y_coord))
    xy_grid = xyN_grid.reshape(2, -1)

    # camera height
    cam_h = 0.73357

    geo_predictor = GeoEstimator(prior_focal=K[0, 0])

    for frame_cnt in range(1, sample_num + 1):
        im_name = f"{frame_cnt:04d}.jpg"
        frame_pack = retrieve_pack_info_by_frame(total_info, frame_cnt)
        im_path = os.path.join(image_path, im_name)
        out_path = os.path.join(vis_path, im_name)
        im = cv2.imread(im_path)
        geo_predictor.load_image(im_path)
        if judge_valid(frame_pack) < 2:
            print(f"unable to collect sufficient labels of lane points")
            cv2.imwrite(out_path, im)
        else:
            vp = VP(K, True)
            vp.line_fit(frame_pack)
            if vp.line_fit_flag:
                vp.compute_vp()
                vanishing_point = vp.filter_candidates("close")
                yaw, pitch = vp.estimate_yp(vanishing_point)
                min_roll, max_roll, min_pitch, max_pitch = geo_predictor.predict()
                est = {
                    "min_roll": min_roll,
                    "max_roll": max_roll,
                    "min_pitch": min_pitch,
                    "max_pitch": max_pitch
                }
                im1 = plot_res(
                    im,
                    frame_pack,
                    vp.param_lst,
                    vanishing_point,
                    yaw,
                    pitch,
                    **est
                )
                
                uv_grid = ground2im(xy_grid, K, pitch, -yaw, cam_h)
                uv_grid_fix = ground2im(xy_grid, K, 0, 0, cam_h)
                uvN_grid = uv_grid.reshape(xyN_grid.shape)
                uvN_grid_fix = uv_grid_fix.reshape(xyN_grid.shape)
                im2 = plot_navi_grid(im1, uvN_grid, vanishing_point)
                im3 = plot_navi_grid_fix(im2, uvN_grid_fix)
                cv2.imwrite(out_path, im3)
            else:
                print(f"unable to collect sufficient number of lane points")
                cv2.imwrite(out_path, im)

    print("done!")
