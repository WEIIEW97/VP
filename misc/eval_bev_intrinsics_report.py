"""Analysis and reporting helpers for BEV intrinsic error evaluation."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import CameraCalib


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    x_min_m: float
    x_max_m: float
    x_step_m: float
    y_min_m: float
    y_max_m: float
    y_step_m: float
    max_reproject_error_px: float
    xy_plot_intrinsic_sample_count: int


@dataclass
class XYReprojectionPlotData:
    grid_x: list[float | None]
    grid_y: list[float | None]
    samples: np.ndarray
    sample_colors: list[str]
    labels: list[str]
    intrinsic_sample_count: int
    intrinsic_total_count: int
    color_by_grid_point: bool


ProjectGroundFn = Callable[
    [np.ndarray, CameraCalib, np.ndarray, np.ndarray], np.ndarray
]
BackprojectGroundFn = Callable[
    [np.ndarray, CameraCalib, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]
InImageFn = Callable[[np.ndarray, CameraCalib], np.ndarray]


def summarize(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            key: float("nan") for key in ["mean", "median", "p90", "p95", "p99", "max"]
        } | {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def summarize_delta(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "abs_median": float("nan"),
            "abs_p90": float("nan"),
            "abs_p95": float("nan"),
        }
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "abs_median": float(np.median(abs_values)),
        "abs_p90": float(np.percentile(abs_values, 90)),
        "abs_p95": float(np.percentile(abs_values, 95)),
    }


def summarize_error_vectors(
    deltas: np.ndarray,
) -> dict[str, dict[str, float | int] | int]:
    if deltas.size == 0:
        norm = np.empty(0, dtype=np.float64)
    else:
        norm = np.linalg.norm(deltas, axis=1)
    norm_stats = summarize(norm)
    return {
        "count": int(deltas.shape[0]),
        "dx_m": summarize_delta(deltas[:, 0] if deltas.size else norm),
        "dy_m": summarize_delta(deltas[:, 1] if deltas.size else norm),
        "xy_norm_m": {
            "count": norm_stats["count"],
            "median": norm_stats["median"],
            "p90": norm_stats["p90"],
            "p95": norm_stats["p95"],
        },
    }


def summarize_error_bins(
    deltas: np.ndarray,
    xy: np.ndarray,
    axis: str,
    start: float,
    stop: float,
    step: float,
) -> list[dict[str, float | int | str | dict]]:
    coord_idx = 0 if axis == "x" else 1
    bins = np.arange(start, stop + step, step, dtype=np.float64)
    summaries = []
    for idx, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == 0:
            mask = (xy[:, coord_idx] >= bin_start) & (xy[:, coord_idx] <= bin_end)
        else:
            mask = (xy[:, coord_idx] > bin_start) & (xy[:, coord_idx] <= bin_end)
        if not np.any(mask):
            continue
        stats = summarize_error_vectors(deltas[mask])
        summaries.append(
            {
                "range_m": f"{bin_start:g}..{bin_end:g}",
                f"{axis}_min_m": float(bin_start),
                f"{axis}_max_m": float(bin_end),
                "count": stats["count"],
                "dx_m": stats["dx_m"],
                "dy_m": stats["dy_m"],
                "xy_norm_m": stats["xy_norm_m"],
            }
        )
    return summaries


def summarize_error_report(
    deltas: np.ndarray, xy: np.ndarray, config: ErrorAnalysisConfig
) -> dict[str, object]:
    return {
        "overall": summarize_error_vectors(deltas),
        "by_y_m": summarize_error_bins(
            deltas, xy, "y", config.y_min_m, config.y_max_m, config.y_step_m
        ),
        "by_x_m": summarize_error_bins(
            deltas, xy, "x", config.x_min_m, config.x_max_m, config.x_step_m
        ),
    }


def evaluate_geom_mle(
    calibs: list[CameraCalib],
    geom_mle: CameraCalib,
    points_w: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    camera_xyz: np.ndarray,
    config: ErrorAnalysisConfig,
    project_ground: ProjectGroundFn,
    backproject_to_ground: BackprojectGroundFn,
    in_image: InImageFn,
) -> dict[str, dict[str, object]]:
    independent_deltas = []
    independent_xy = []
    geom_mle_deltas = []
    geom_mle_xy = []
    for calib in calibs:
        true_uv = project_ground(points_w, calib, R_cw, t_cw)
        visible = in_image(true_uv, calib)

        own_xy, own_ok = backproject_to_ground(true_uv, calib, R_cw, camera_xyz)
        own_valid = visible & own_ok
        independent_deltas.append(own_xy[own_valid] - points_w[own_valid, :2])
        independent_xy.append(points_w[own_valid, :2])

        pred_xy, pred_ok = backproject_to_ground(true_uv, geom_mle, R_cw, camera_xyz)
        reproj_uv = project_ground(
            np.column_stack([pred_xy, np.zeros(pred_xy.shape[0])]), geom_mle, R_cw, t_cw
        )
        reproj_ok = (
            np.linalg.norm(reproj_uv - true_uv, axis=1) <= config.max_reproject_error_px
        )
        valid = visible & pred_ok & reproj_ok & np.isfinite(pred_xy).all(axis=1)
        geom_mle_deltas.append(pred_xy[valid] - points_w[valid, :2])
        geom_mle_xy.append(points_w[valid, :2])

    return {
        "independent": summarize_error_report(
            np.concatenate(independent_deltas), np.concatenate(independent_xy), config
        ),
        "geom_mle_unified": summarize_error_report(
            np.concatenate(geom_mle_deltas), np.concatenate(geom_mle_xy), config
        ),
    }


def grid_shape_for_points(points_w: np.ndarray) -> tuple[int, int]:
    return (
        np.unique(points_w[:, 1]).size,
        np.unique(points_w[:, 0]).size,
    )


def grid_line_xy(points_w: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    ny, nx = grid_shape_for_points(points_w)
    if points_w.shape[0] != ny * nx:
        raise ValueError(
            f"Expected a complete rectangular grid, got {points_w.shape[0]} "
            f"points for shape {(ny, nx)}"
        )
    points_grid = points_w.reshape(ny, nx, 3)
    xs: list[float | None] = []
    ys: list[float | None] = []

    def add(p0: np.ndarray, p1: np.ndarray) -> None:
        xs.extend([float(p0[0]), float(p1[0]), None])
        ys.extend([float(p0[1]), float(p1[1]), None])

    for y_idx in range(ny):
        for x_idx in range(nx - 1):
            add(points_grid[y_idx, x_idx], points_grid[y_idx, x_idx + 1])
    for x_idx in range(nx):
        for y_idx in range(ny - 1):
            add(points_grid[y_idx, x_idx], points_grid[y_idx + 1, x_idx])
    return xs, ys


def select_plot_calibs(
    calibs: list[CameraCalib], sample_count: int | None
) -> list[CameraCalib]:
    if sample_count is None or len(calibs) <= sample_count:
        return calibs
    indices = np.linspace(0, len(calibs) - 1, sample_count, dtype=np.int64)
    return [calibs[int(idx)] for idx in indices]


def grid_point_colors(count: int) -> list[str]:
    cmap = plt.get_cmap("turbo", count)
    return [mcolors.to_hex(cmap(idx)) for idx in range(count)]


def build_xy_reprojection_plot_data(
    calibs: list[CameraCalib],
    geom_mle: CameraCalib,
    points_w: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    camera_xyz: np.ndarray,
    config: ErrorAnalysisConfig,
    project_ground: ProjectGroundFn,
    backproject_to_ground: BackprojectGroundFn,
    in_image: InImageFn,
    intrinsic_sample_count: int | None,
    color_by_grid_point: bool,
) -> XYReprojectionPlotData:
    reference_uv = project_ground(points_w, geom_mle, R_cw, t_cw)
    reference_visible = in_image(reference_uv, geom_mle)
    plot_calibs = select_plot_calibs(calibs, intrinsic_sample_count)
    point_colors = grid_point_colors(points_w.shape[0])

    xy_samples = []
    sample_colors = []
    labels = []
    for calib in plot_calibs:
        pred_xy, pred_ok = backproject_to_ground(reference_uv, calib, R_cw, camera_xyz)
        reproj_uv = project_ground(
            np.column_stack([pred_xy, np.zeros(pred_xy.shape[0])]), calib, R_cw, t_cw
        )
        reproj_ok = (
            np.linalg.norm(reproj_uv - reference_uv, axis=1)
            <= config.max_reproject_error_px
        )
        valid = (
            reference_visible & pred_ok & reproj_ok & np.isfinite(pred_xy).all(axis=1)
        )

        valid_idx = np.flatnonzero(valid)
        xy_samples.append(pred_xy[valid])
        if color_by_grid_point:
            sample_colors.extend([point_colors[idx] for idx in valid_idx])
        else:
            sample_colors.extend(["rgba(35, 115, 190, 0.35)"] * valid_idx.size)
        labels.extend(
            [
                (
                    f"grid point #{idx}<br>"
                    f"{calib.name}<br>grid x={points_w[idx, 0]:.2f}m, y={points_w[idx, 1]:.2f}m"
                    f"<br>reprojected x={pred_xy[idx, 0]:.3f}m, y={pred_xy[idx, 1]:.3f}m"
                    f"<br>dx={pred_xy[idx, 0] - points_w[idx, 0]:.3f}m, dy={pred_xy[idx, 1] - points_w[idx, 1]:.3f}m"
                )
                for idx in valid_idx
            ]
        )

    samples = np.concatenate(xy_samples, axis=0) if xy_samples else np.empty((0, 2))
    grid_x, grid_y = grid_line_xy(points_w)
    return XYReprojectionPlotData(
        grid_x=grid_x,
        grid_y=grid_y,
        samples=samples,
        sample_colors=sample_colors,
        labels=labels,
        intrinsic_sample_count=len(plot_calibs),
        intrinsic_total_count=len(calibs),
        color_by_grid_point=color_by_grid_point,
    )


def write_xy_error_html(
    path: Path,
    points_w: np.ndarray,
    plot_data: XYReprojectionPlotData,
    config: ErrorAnalysisConfig,
) -> None:
    if plot_data.color_by_grid_point:
        point_colors: list[str] | str = grid_point_colors(points_w.shape[0])
        reference_text = [
            f"grid point #{idx}<br>reference x={p[0]:.2f}m, y={p[1]:.2f}m"
            for idx, p in enumerate(points_w)
        ]
    else:
        point_colors = "rgb(20, 20, 20)"
        reference_text = [f"reference x={p[0]:.2f}m, y={p[1]:.2f}m" for p in points_w]
    if plot_data.intrinsic_sample_count == plot_data.intrinsic_total_count:
        scatter_name = (
            "independent intrinsics reprojected xy "
            f"(all {plot_data.intrinsic_total_count} intrinsics)"
        )
        title = "Ground XY Reprojection Grid: full independent-intrinsic xy scatter"
    else:
        scatter_name = (
            "independent intrinsics reprojected xy "
            f"({plot_data.intrinsic_sample_count} sampled intrinsics)"
        )
        title = (
            "Ground XY Reprojection Grid: sampled independent-intrinsic scatter "
            "colored by reference grid point"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_data.grid_x,
            y=plot_data.grid_y,
            mode="lines",
            name="unified/reference xy grid",
            line=dict(color="rgb(20, 20, 20)", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=points_w[:, 0],
            y=points_w[:, 1],
            mode="markers",
            name="unified/reference grid points",
            marker=dict(
                size=7,
                color=point_colors,
                symbol="circle-open",
                line=dict(width=2, color="rgb(20, 20, 20)"),
            ),
            text=reference_text,
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=plot_data.samples[:, 0] if plot_data.samples.size else [],
            y=plot_data.samples[:, 1] if plot_data.samples.size else [],
            mode="markers",
            name=scatter_name,
            marker=dict(
                size=5,
                color=plot_data.sample_colors,
                opacity=0.72,
                line=dict(width=0),
            ),
            text=plot_data.labels,
            hoverinfo="text",
        )
    )
    x_margin = max(config.x_step_m, 1.0)
    y_margin = max(config.y_step_m, 2.0)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1100,
        height=900,
        margin=dict(l=10, r=10, t=58, b=10),
        xaxis=dict(
            title="ground x (m)",
            range=[config.x_min_m - x_margin, config.x_max_m + x_margin],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="ground depth y (m)",
            range=[config.y_min_m, config.y_max_m + y_margin],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)


def write_xy_error_png(
    path: Path,
    points_w: np.ndarray,
    plot_data: XYReprojectionPlotData,
    config: ErrorAnalysisConfig,
) -> None:
    point_colors = grid_point_colors(points_w.shape[0])
    if plot_data.intrinsic_sample_count == plot_data.intrinsic_total_count:
        scatter_label = (
            "independent intrinsics reprojected xy "
            f"(all {plot_data.intrinsic_total_count} intrinsics)"
        )
        title = "Ground XY reprojection grid: full scatter"
    else:
        scatter_label = (
            "independent intrinsics reprojected xy "
            f"({plot_data.intrinsic_sample_count} sampled intrinsics)"
        )
        title = "Ground XY reprojection grid colored by reference grid point"
    grid_x_plot = np.array(
        [np.nan if value is None else value for value in plot_data.grid_x],
        dtype=np.float64,
    )
    grid_y_plot = np.array(
        [np.nan if value is None else value for value in plot_data.grid_y],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(9.5, 11.0), dpi=180)
    ax.plot(
        grid_x_plot,
        grid_y_plot,
        color="black",
        linewidth=1.2,
        label="unified/reference xy grid",
    )
    if plot_data.color_by_grid_point:
        ax.scatter(
            points_w[:, 0],
            points_w[:, 1],
            s=28,
            c=point_colors,
            edgecolors="black",
            linewidths=1.1,
            label="unified/reference grid points",
            zorder=3,
        )
    else:
        ax.scatter(
            points_w[:, 0],
            points_w[:, 1],
            s=28,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label="unified/reference grid points",
            zorder=3,
        )
    if plot_data.samples.size:
        if plot_data.color_by_grid_point:
            ax.scatter(
                plot_data.samples[:, 0],
                plot_data.samples[:, 1],
                s=12,
                c=plot_data.sample_colors,
                alpha=0.72,
                linewidths=0,
                label=scatter_label,
                zorder=2,
            )
        else:
            ax.scatter(
                plot_data.samples[:, 0],
                plot_data.samples[:, 1],
                s=9,
                color="#1f77b4",
                alpha=0.28,
                linewidths=0,
                label=scatter_label,
                zorder=2,
            )
    x_margin = max(config.x_step_m, 1.0)
    y_margin = max(config.y_step_m, 2.0)
    ax.set_xlim(config.x_min_m - x_margin, config.x_max_m + x_margin)
    ax.set_ylim(config.y_min_m, config.y_max_m + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ground x (m)")
    ax.set_ylabel("ground depth y (m)")
    ax.set_title(title)
    ax.grid(True, color="0.86", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bin_centers(items: list[dict[str, object]], axis: str) -> np.ndarray:
    return np.array(
        [(item[f"{axis}_min_m"] + item[f"{axis}_max_m"]) * 0.5 for item in items],
        dtype=np.float64,
    )


def bin_labels(items: list[dict[str, object]]) -> list[str]:
    return [str(item["range_m"]) for item in items]


def stat_series(items: list[dict[str, object]], key: str) -> np.ndarray:
    if key == "dx_abs_p95":
        values = [item["dx_m"]["abs_p95"] for item in items]
    elif key == "dy_abs_p95":
        values = [item["dy_m"]["abs_p95"] for item in items]
    elif key == "norm_p95":
        values = [item["xy_norm_m"]["p95"] for item in items]
    else:
        raise ValueError(f"Unsupported stat series: {key}")
    return np.asarray(values, dtype=np.float64)


def stat_hover_text(items: list[dict[str, object]]) -> list[str]:
    text = []
    for item in items:
        dx = item["dx_m"]
        dy = item["dy_m"]
        norm = item["xy_norm_m"]
        text.append(
            f"range={item['range_m']}m<br>"
            f"count={item['count']}<br>"
            f"dx median={dx['median']:.4f}m, |dx| p95={dx['abs_p95']:.4f}m<br>"
            f"dy median={dy['median']:.4f}m, |dy| p95={dy['abs_p95']:.4f}m<br>"
            f"norm p95={norm['p95']:.4f}m"
        )
    return text


STAT_LINES = {
    "dx_abs_p95": ("|dx| p95", "#2f6fb3"),
    "dy_abs_p95": ("|dy| p95", "#c23b32"),
    "norm_p95": ("XY norm p95", "#2f8f46"),
}


def add_stat_subplot(
    fig: go.Figure,
    items: list[dict[str, object]],
    axis: str,
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    centers = bin_centers(items, axis)
    labels = bin_labels(items)
    hover_text = stat_hover_text(items)
    for key, (name, color) in STAT_LINES.items():
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=stat_series(items, key),
                mode="lines+markers",
                name=name,
                legendgroup=key,
                showlegend=showlegend,
                line=dict(color=color, width=2.4),
                marker=dict(size=7),
                customdata=labels,
                text=hover_text,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )


def write_error_stats_html(
    path: Path, eval_stats: dict[str, dict[str, object]]
) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "geom_mle_unified by depth y",
            "geom_mle_unified by lateral x",
        ),
        horizontal_spacing=0.09,
    )
    add_stat_subplot(fig, eval_stats["geom_mle_unified"]["by_y_m"], "y", 1, 1, True)
    add_stat_subplot(fig, eval_stats["geom_mle_unified"]["by_x_m"], "x", 1, 2, False)

    fig.update_layout(
        title="Directional BEV Error Statistics: geom_mle_unified",
        template="plotly_white",
        width=1300,
        height=540,
        margin=dict(l=20, r=20, t=70, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="ground depth y bin center (m)", row=1, col=1)
    fig.update_xaxes(title_text="ground lateral x bin center (m)", row=1, col=2)
    fig.update_yaxes(title_text="error (m)", row=1, col=1)
    fig.update_yaxes(title_text="error (m)", row=1, col=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)


def plot_stat_panel(
    ax,
    items: list[dict[str, object]],
    axis: str,
    title: str,
) -> None:
    centers = bin_centers(items, axis)
    for key, (name, color) in STAT_LINES.items():
        ax.plot(
            centers,
            stat_series(items, key),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel(f"ground {'depth y' if axis == 'y' else 'lateral x'} bin center (m)")
    ax.set_ylabel("error (m)")
    ax.grid(True, color="0.86", linewidth=0.8)


def write_error_stats_png(path: Path, eval_stats: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=170, sharey=False)
    plot_stat_panel(
        axes[0],
        eval_stats["geom_mle_unified"]["by_y_m"],
        "y",
        "geom_mle_unified by depth y",
    )
    plot_stat_panel(
        axes[1],
        eval_stats["geom_mle_unified"]["by_x_m"],
        "x",
        "geom_mle_unified by lateral x",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Directional BEV Error Statistics: geom_mle_unified", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def p95_csv_row(
    model: str,
    group: str,
    axis: str,
    stats: dict[str, object],
    range_m: str = "",
    axis_min_m: float | str = "",
    axis_max_m: float | str = "",
    axis_center_m: float | str = "",
) -> dict[str, object]:
    return {
        "model": model,
        "group": group,
        "axis": axis,
        "range_m": range_m,
        "axis_min_m": axis_min_m,
        "axis_max_m": axis_max_m,
        "axis_center_m": axis_center_m,
        "count": stats["count"],
        "dx_abs_p95_m": stats["dx_m"]["abs_p95"],
        "dy_abs_p95_m": stats["dy_m"]["abs_p95"],
        "xy_norm_p95_m": stats["xy_norm_m"]["p95"],
    }


def write_error_stats_csv(path: Path, eval_stats: dict[str, dict[str, object]]) -> None:
    model = "geom_mle_unified"
    stats = eval_stats[model]
    rows = [p95_csv_row(model, "overall", "", stats["overall"])]

    for axis, group in (("y", "by_y_m"), ("x", "by_x_m")):
        for item in stats[group]:
            axis_min_m = float(item[f"{axis}_min_m"])
            axis_max_m = float(item[f"{axis}_max_m"])
            rows.append(
                p95_csv_row(
                    model=model,
                    group=group,
                    axis=axis,
                    stats=item,
                    range_m=str(item["range_m"]),
                    axis_min_m=axis_min_m,
                    axis_max_m=axis_max_m,
                    axis_center_m=(axis_min_m + axis_max_m) * 0.5,
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "group",
        "axis",
        "range_m",
        "axis_min_m",
        "axis_max_m",
        "axis_center_m",
        "count",
        "dx_abs_p95_m",
        "dy_abs_p95_m",
        "xy_norm_p95_m",
    ]
    pd.DataFrame(rows, columns=fieldnames).to_csv(
        path,
        index=False,
        float_format="%.6f",
    )


def print_bin_summary(prefix: str, items: list[dict[str, object]]) -> None:
    print(f"  {prefix}:")
    for item in items:
        dx = item["dx_m"]
        dy = item["dy_m"]
        norm = item["xy_norm_m"]
        print(
            f"    {item['range_m']}m: count={item['count']}, "
            f"dx_med={dx['median']:.4f}m, "
            f"dx_abs_p90={dx['abs_p90']:.4f}m, "
            f"dx_abs_p95={dx['abs_p95']:.4f}m, "
            f"dy_med={dy['median']:.4f}m, "
            f"dy_abs_p90={dy['abs_p90']:.4f}m, "
            f"dy_abs_p95={dy['abs_p95']:.4f}m, "
            f"norm_p95={norm['p95']:.4f}m"
        )


def print_summary(name: str, stats: dict[str, object]) -> None:
    print(f"{name}:")
    overall = stats["overall"]
    dx = overall["dx_m"]
    dy = overall["dy_m"]
    norm = overall["xy_norm_m"]
    print(
        f"  overall: count={overall['count']}, "
        f"dx_med={dx['median']:.4f}m, "
        f"dx_abs_p95={dx['abs_p95']:.4f}m, "
        f"dy_med={dy['median']:.4f}m, "
        f"dy_abs_p95={dy['abs_p95']:.4f}m, "
        f"norm_p95={norm['p95']:.4f}m"
    )
    print_bin_summary("by_y", stats["by_y_m"])
    print_bin_summary("by_x", stats["by_x_m"])
