import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import CameraCalib, load_intrinsics


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_bev_intrinsics_report import (
    ErrorAnalysisConfig,
    build_xy_reprojection_plot_data,
    evaluate_geom_mle,
    grid_shape_for_points,
    print_summary,
    summarize_error_report,
    write_error_stats_csv,
    write_error_stats_html,
    write_error_stats_png,
    write_xy_error_html,
    write_xy_error_png,
)
from vp.ipm import IPM

STATS_XLSX = REPO_ROOT / "data/stats/stats.xlsx"
EXTRINSICS_DIR = REPO_ROOT / "data/stats/extrinsics"

MEAN_EXTRINSIC_JSON = REPO_ROOT / "data/stats/mean_extrinsics.json"
GEOM_MLE_INTRINSIC_JSON = REPO_ROOT / "data/stats/geom_mle_intrinsics.json"
XY_ERROR_HTML = REPO_ROOT / "data/stats/xy_error_plot.html"
XY_ERROR_PNG = REPO_ROOT / "data/stats/xy_error_plot.png"
XY_ERROR_SAMPLED_HTML = REPO_ROOT / "data/stats/xy_error_sampled_plot.html"
XY_ERROR_SAMPLED_PNG = REPO_ROOT / "data/stats/xy_error_sampled_plot.png"
ERROR_STATS_HTML = REPO_ROOT / "data/stats/error_stats_plot.html"
ERROR_STATS_PNG = REPO_ROOT / "data/stats/error_stats_plot.png"
ERROR_STATS_CSV = REPO_ROOT / "data/stats/error_stats_p95.csv"

BODY_ORIGIN_X_M = 0.0
BODY_ORIGIN_Y_M = 0.0
USE_TRANSLATION_XY_OFFSET = True

X_MIN_M = -4.5
X_MAX_M = 4.5
X_STEP_M = 1.5
Y_MIN_M = 0.0
Y_MAX_M = 40.0
Y_STEP_M = 5.0

ANALYSIS_Y_MIN_EPS_M = 1e-9
XY_PLOT_INTRINSIC_SAMPLE_COUNT = 5
MLE_MAX_NFEV = 80
INVALID_RESIDUAL_M = 50.0
MAX_REPROJECT_ERROR_PX = 1.0

ERROR_ANALYSIS_CONFIG = ErrorAnalysisConfig(
    x_min_m=X_MIN_M,
    x_max_m=X_MAX_M,
    x_step_m=X_STEP_M,
    y_min_m=Y_MIN_M,
    y_max_m=Y_MAX_M,
    y_step_m=Y_STEP_M,
    max_reproject_error_px=MAX_REPROJECT_ERROR_PX,
    xy_plot_intrinsic_sample_count=XY_PLOT_INTRINSIC_SAMPLE_COUNT,
)


@dataclass
class ExtrinsicStats:
    mean_pose: np.ndarray  # [height, pitch_deg, roll_deg, yaw_deg]
    std_pose: np.ndarray
    mean_translation: np.ndarray  # [x_offset, y_offset, z_offset]
    std_translation: np.ndarray
    count: int


@dataclass
class MLEObservation:
    xy: np.ndarray
    uv: np.ndarray


def circular_mean_deg(values: np.ndarray) -> float:
    rad = np.deg2rad(values)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def load_extrinsic_stats(root: Path) -> ExtrinsicStats:
    poses = []
    translations = []
    for path in sorted(root.glob("*/extrinsics_r.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        poses.append(np.asarray(data["cam_pose"], dtype=np.float64))
        translations.append(
            np.asarray(data["cam_translation_vector"], dtype=np.float64)
        )

    if not poses:
        raise ValueError(f"No extrinsics_r.json files found under {root}")

    poses_arr = np.stack(poses)
    trans_arr = np.stack(translations)
    mean_pose = np.array(
        [
            np.mean(poses_arr[:, 0]),
            circular_mean_deg(poses_arr[:, 1]),
            circular_mean_deg(poses_arr[:, 2]),
            circular_mean_deg(poses_arr[:, 3]),
        ],
        dtype=np.float64,
    )
    return ExtrinsicStats(
        mean_pose=mean_pose,
        std_pose=np.std(poses_arr, axis=0, ddof=1),
        mean_translation=np.mean(trans_arr, axis=0),
        std_translation=np.std(trans_arr, axis=0, ddof=1),
        count=len(poses),
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def rotation_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def R2ypr(R: np.ndarray) -> np.ndarray:
    """
    Common VINS-style R2ypr implementation.
    Returns [yaw, pitch, roll] in degrees.

    yaw   around z
    pitch around y-ish equivalent extracted from rotation matrix
    roll  around x-ish equivalent extracted from rotation matrix
    """
    n = R[:, 0]
    o = R[:, 1]
    a = R[:, 2]

    yaw = np.arctan2(n[1], n[0])
    pitch = np.arctan2(-n[2], n[0] * np.cos(yaw) + n[1] * np.sin(yaw))
    roll = np.arctan2(
        a[0] * np.sin(yaw) - a[1] * np.cos(yaw),
        -o[0] * np.sin(yaw) + o[1] * np.cos(yaw),
    )

    return np.rad2deg(np.array([yaw, pitch, roll], dtype=np.float64))


def get_pose_imu_rotation(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """
    Directly returns R_c_g from the original C++ logic.
    """

    R_tb_gtb = (
        rotation_z(np.deg2rad(yaw_deg))
        @ rotation_x(np.deg2rad(pitch_deg))
        @ rotation_y(np.deg2rad(roll_deg))
    )

    R_cnvp_gtb = rotation_x(np.deg2rad(90.0))

    R_c_g = R_cnvp_gtb @ R_tb_gtb.T

    return R_c_g


def make_pose(
    camera_xyz: np.ndarray,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, pitch_deg, roll_deg, yaw_deg = pose

    R_cw = get_pose_imu_rotation(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    t_cw = -R_cw @ camera_xyz.reshape(3, 1)
    # print(f"t_cw is {t_cw}")
    ypr = R2ypr(R_cw)
    # print(f"Pose: yaw={ypr[0]:.2f}°, pitch={ypr[1]:.2f}°, roll={ypr[2]:.2f}°")
    return R_cw, t_cw


def axis_samples(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=np.float64)
    values[-1] = stop
    return values


def sample_ground() -> np.ndarray:
    xs = axis_samples(X_MIN_M, X_MAX_M, X_STEP_M)
    ys = axis_samples(Y_MIN_M, Y_MAX_M, Y_STEP_M)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y, np.zeros_like(grid_x)], axis=-1).reshape(-1, 3)


def filter_analysis_ground(points_w: np.ndarray) -> np.ndarray:
    return points_w[points_w[:, 1] > Y_MIN_M + ANALYSIS_Y_MIN_EPS_M]


def grid_shape() -> tuple[int, int]:
    return (
        axis_samples(Y_MIN_M, Y_MAX_M, Y_STEP_M).size,
        axis_samples(X_MIN_M, X_MAX_M, X_STEP_M).size,
    )


def fisheye_dist(calib: CameraCalib) -> np.ndarray:
    dist = (
        calib.dist[:4]
        if calib.dist.size >= 4
        else np.pad(calib.dist, (0, 4 - calib.dist.size))
    )
    return dist.reshape(4, 1)


def project_ground(
    points_w: np.ndarray, calib: CameraCalib, R_cw: np.ndarray, t_cw: np.ndarray
) -> np.ndarray:
    H_i_g = IPM.TransformGround2Image(R_cw, t_cw)
    dist = fisheye_dist(calib).ravel()
    uv = [
        IPM.SpaceToPlane(
            H_i_g @ np.array([point[0], point[1], 1.0]), calib.K, dist, True
        )
        for point in points_w
    ]
    return np.asarray(uv, dtype=np.float64)


def backproject_to_ground(
    uv: np.ndarray,
    calib: CameraCalib,
    R_cw: np.ndarray,
    camera_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    undist = cv2.fisheye.undistortPoints(
        uv.reshape(-1, 1, 2), calib.K, fisheye_dist(calib)
    )
    rays_c = np.column_stack([undist.reshape(-1, 2), np.ones(uv.shape[0])])
    rays_w = rays_c @ R_cw

    denom = rays_w[:, 2]
    scale = np.full(uv.shape[0], np.nan)
    valid = np.abs(denom) > 1e-9
    scale[valid] = -camera_xyz[2] / denom[valid]
    valid &= scale > 0.0

    points_w = camera_xyz.reshape(1, 3) + scale[:, None] * rays_w
    return points_w[:, :2], valid


def in_image(uv: np.ndarray, calib: CameraCalib) -> np.ndarray:
    return (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < calib.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < calib.height)
    )


def mean_intrinsic(calibs: list[CameraCalib]) -> CameraCalib:
    widths = {c.width for c in calibs}
    heights = {c.height for c in calibs}
    models = {c.model_type for c in calibs}
    if len(widths) != 1 or len(heights) != 1 or len(models) != 1:
        raise ValueError("All intrinsics must have the same resolution and model type.")

    dist_len = max(c.dist.size for c in calibs)
    K = np.mean(np.stack([c.K for c in calibs]), axis=0)
    dist = np.mean(
        np.stack([np.pad(c.dist, (0, dist_len - c.dist.size)) for c in calibs]), axis=0
    )
    return CameraCalib(
        "mean_initial", K, dist, calibs[0].width, calibs[0].height, calibs[0].model_type
    )


def pack(calib: CameraCalib) -> np.ndarray:
    return np.array(
        [
            calib.K[0, 0],
            calib.K[1, 1],
            calib.K[0, 2],
            calib.K[1, 2],
            *fisheye_dist(calib).ravel(),
        ],
        dtype=np.float64,
    )


def unpack(params: np.ndarray, template: CameraCalib, name: str) -> CameraCalib:
    K = np.array(
        [[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]]
    )
    dist = np.zeros(max(template.dist.size, 14), dtype=np.float64)
    dist[:4] = params[4:8]
    return CameraCalib(
        name, K, dist, template.width, template.height, template.model_type
    )


def bounds_from_calibs(calibs: list[CameraCalib]) -> tuple[np.ndarray, np.ndarray]:
    params = np.stack([pack(calib) for calib in calibs])
    low = params.min(axis=0)
    high = params.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    lower = low - 0.5 * span
    upper = high + 0.5 * span
    lower[:2] = np.maximum(lower[:2], 1.0)
    lower[2] = max(0.0, lower[2])
    upper[2] = min(calibs[0].width - 1.0, upper[2])
    lower[3] = max(0.0, lower[3])
    upper[3] = min(calibs[0].height - 1.0, upper[3])
    return lower, upper


def mle_observations(
    calibs: list[CameraCalib],
    points_w: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
) -> list[MLEObservation]:
    observations = []
    for calib in calibs:
        uv = project_ground(points_w, calib, R_cw, t_cw)
        valid = in_image(uv, calib)
        if np.any(valid):
            observations.append(MLEObservation(points_w[valid, :2], uv[valid]))
    return observations


def mle_residuals(
    params: np.ndarray,
    template: CameraCalib,
    observations: list[MLEObservation],
    R_cw: np.ndarray,
    camera_xyz: np.ndarray,
) -> np.ndarray:
    candidate = unpack(params, template, "candidate")
    chunks = []
    for obs in observations:
        pred_xy, valid = backproject_to_ground(obs.uv, candidate, R_cw, camera_xyz)
        residual = pred_xy - obs.xy
        residual[~valid | ~np.isfinite(residual).all(axis=1)] = INVALID_RESIDUAL_M
        chunks.append(residual.ravel())
    return np.concatenate(chunks)


def fit_geom_mle(
    calibs: list[CameraCalib],
    points_w: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    camera_xyz: np.ndarray,
) -> tuple[CameraCalib, dict]:
    initial = mean_intrinsic(calibs)
    observations = mle_observations(calibs, points_w, R_cw, t_cw)
    lower, upper = bounds_from_calibs(calibs)
    x0 = np.clip(pack(initial), lower, upper)

    result = least_squares(
        mle_residuals,
        x0,
        bounds=(lower, upper),
        args=(initial, observations, R_cw, camera_xyz),
        loss="linear",
        x_scale=np.maximum(np.abs(x0), 1e-3),
        max_nfev=MLE_MAX_NFEV,
    )
    residual = mle_residuals(result.x, initial, observations, R_cw, camera_xyz).reshape(
        -1, 2
    )
    residual_xy = np.concatenate([obs.xy for obs in observations])
    return unpack(result.x, initial, "geom_mle_unified"), {
        "success": bool(result.success),
        "message": result.message,
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "observation_camera_count": len(observations),
        "observation_point_count": int(sum(obs.xy.shape[0] for obs in observations)),
        "mle_error_summary_m": summarize_error_report(
            residual, residual_xy, ERROR_ANALYSIS_CONFIG
        ),
        "bounds_lower": lower.tolist(),
        "bounds_upper": upper.tolist(),
    }


def save_outputs(
    extrinsics: ExtrinsicStats,
    geom_mle: CameraCalib,
    mle_stats: dict,
    intrinsics_count: int,
    sheet_name: str,
) -> None:
    write_json(
        MEAN_EXTRINSIC_JSON,
        {
            "cam_pose": extrinsics.mean_pose.tolist(),
            "cam_pose_comments": "Height, PitchAngle, RollAngle, YawAngle",
            "cam_translation_vector": extrinsics.mean_translation.tolist(),
            "cam_translation_vector_comments": ["x_offset", "y_offset", "z_offset"],
            "stats": {
                "count": extrinsics.count,
                "std_cam_pose": extrinsics.std_pose.tolist(),
                "std_cam_translation_vector": extrinsics.std_translation.tolist(),
            },
        },
    )
    write_json(
        GEOM_MLE_INTRINSIC_JSON,
        {
            "cam_distcoeffs": geom_mle.dist.tolist(),
            "cam_intrinsic": geom_mle.K.reshape(-1).tolist(),
            "model_type": geom_mle.model_type,
            "resolution_height": geom_mle.height,
            "resolution_width": geom_mle.width,
            "stats": {
                "count": intrinsics_count,
                "method": "geometric_mle",
                "likelihood": "iid isotropic Gaussian residuals on BEV ground-plane XY errors",
                "source_xlsx": str(STATS_XLSX),
                "source_sheet": sheet_name,
                "grid": {
                    "x_min_m": X_MIN_M,
                    "x_max_m": X_MAX_M,
                    "x_step_m": X_STEP_M,
                    "y_min_m": Y_MIN_M,
                    "y_max_m": Y_MAX_M,
                    "y_step_m": Y_STEP_M,
                    "include_boundary": True,
                    "analysis_y_min_exclusive_m": Y_MIN_M,
                },
                **mle_stats,
            },
        },
    )


def main() -> None:
    calibs, sheet_name = load_intrinsics(STATS_XLSX)
    extrinsics = load_extrinsic_stats(EXTRINSICS_DIR)

    x_offset = extrinsics.mean_translation[0] if USE_TRANSLATION_XY_OFFSET else 0.0
    y_offset = extrinsics.mean_translation[1] if USE_TRANSLATION_XY_OFFSET else 0.0
    camera_xyz = np.array(
        [
            BODY_ORIGIN_X_M + x_offset,
            BODY_ORIGIN_Y_M + y_offset,
            extrinsics.mean_translation[2],
        ],
        dtype=np.float64,
    )
    R_cw, t_cw = make_pose(camera_xyz, extrinsics.mean_pose)
    sampled_points_w = sample_ground()
    points_w = filter_analysis_ground(sampled_points_w)

    geom_mle, mle_stats = fit_geom_mle(calibs, points_w, R_cw, t_cw, camera_xyz)
    eval_stats = evaluate_geom_mle(
        calibs,
        geom_mle,
        points_w,
        R_cw,
        t_cw,
        camera_xyz,
        ERROR_ANALYSIS_CONFIG,
        project_ground,
        backproject_to_ground,
        in_image,
    )
    xy_full_plot_data = build_xy_reprojection_plot_data(
        calibs,
        geom_mle,
        points_w,
        R_cw,
        t_cw,
        camera_xyz,
        ERROR_ANALYSIS_CONFIG,
        project_ground,
        backproject_to_ground,
        in_image,
        intrinsic_sample_count=None,
        color_by_grid_point=False,
    )
    xy_sampled_plot_data = build_xy_reprojection_plot_data(
        calibs,
        geom_mle,
        points_w,
        R_cw,
        t_cw,
        camera_xyz,
        ERROR_ANALYSIS_CONFIG,
        project_ground,
        backproject_to_ground,
        in_image,
        intrinsic_sample_count=ERROR_ANALYSIS_CONFIG.xy_plot_intrinsic_sample_count,
        color_by_grid_point=True,
    )

    save_outputs(extrinsics, geom_mle, mle_stats, len(calibs), sheet_name)
    write_xy_error_html(
        XY_ERROR_HTML, points_w, xy_full_plot_data, ERROR_ANALYSIS_CONFIG
    )
    write_xy_error_png(XY_ERROR_PNG, points_w, xy_full_plot_data, ERROR_ANALYSIS_CONFIG)
    write_xy_error_html(
        XY_ERROR_SAMPLED_HTML, points_w, xy_sampled_plot_data, ERROR_ANALYSIS_CONFIG
    )
    write_xy_error_png(
        XY_ERROR_SAMPLED_PNG, points_w, xy_sampled_plot_data, ERROR_ANALYSIS_CONFIG
    )
    write_error_stats_html(ERROR_STATS_HTML, eval_stats)
    write_error_stats_png(ERROR_STATS_PNG, eval_stats)
    write_error_stats_csv(ERROR_STATS_CSV, eval_stats)

    print(
        f"Parsed intrinsics: {len(calibs)} cameras from {STATS_XLSX} sheet={sheet_name}"
    )
    print(f"Parsed extrinsics: {extrinsics.count} files from {EXTRINSICS_DIR}")
    print(
        f"Ground samples: {sampled_points_w.shape[0]} points, shape={grid_shape()}, "
        f"boundary-inclusive"
    )
    print(
        f"Analysis ground samples: {points_w.shape[0]} points, "
        f"shape={grid_shape_for_points(points_w)}, excluded y <= {Y_MIN_M:g}m"
    )
    print(
        f"XY full plot intrinsic samples: {xy_full_plot_data.intrinsic_sample_count} "
        f"of {len(calibs)} cameras"
    )
    print(
        f"XY sampled plot intrinsic samples: "
        f"{xy_sampled_plot_data.intrinsic_sample_count} of {len(calibs)} cameras"
    )
    print(
        f"Camera xyz: ({camera_xyz[0]:.3f}, {camera_xyz[1]:.3f}, {camera_xyz[2]:.3f})m"
    )
    print(
        f"geom_mle optimizer: success={mle_stats['success']}, nfev={mle_stats['nfev']}"
    )
    print_summary("independent", eval_stats["independent"])
    print_summary("geom_mle_unified", eval_stats["geom_mle_unified"])
    print(f"Wrote {MEAN_EXTRINSIC_JSON}")
    print(f"Wrote {GEOM_MLE_INTRINSIC_JSON}")
    print(f"Wrote {XY_ERROR_HTML}")
    print(f"Wrote {XY_ERROR_PNG}")
    print(f"Wrote {XY_ERROR_SAMPLED_HTML}")
    print(f"Wrote {XY_ERROR_SAMPLED_PNG}")
    print(f"Wrote {ERROR_STATS_HTML}")
    print(f"Wrote {ERROR_STATS_PNG}")
    print(f"Wrote {ERROR_STATS_CSV}")


if __name__ == "__main__":
    main()
