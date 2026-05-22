# BEV Intrinsics Error Evaluation

This note explains what `misc/eval_bev_intrinsics_error.py` does, the algorithmic logic behind it, and how it compares independent camera intrinsics against a unified BEV intrinsic model.

The implementation is split into two files:

- `misc/eval_bev_intrinsics_error.py`
  - Loads calibration data, prepares geometry, fits `geom_mle_unified`, and orchestrates outputs.
- `misc/eval_bev_intrinsics_report.py`
  - Owns error analysis, bin summaries, XY plot data construction, HTML/PNG plot writers, CSV export, and terminal summaries.

## Purpose

The script fits one unified fisheye intrinsic model for BEV use from many per-camera intrinsic calibrations, then evaluates how much BEV ground-plane error is introduced when that unified model replaces each camera's own intrinsic model.

It does not use real images or detected image features for the evaluation. Instead, it builds synthetic geometric observations by projecting sampled ground-plane points through each camera's own calibrated intrinsics.

## Inputs

- `data/stats/stats.xlsx`
  - Contains per-camera intrinsic parameters.
  - The script parses `cam_intrinsic`, `cam_distcoeffs`, resolution, and `model_type`.
  - The expected model is `KANNALA_BRANDT`.

- `data/stats/extrinsics/*/extrinsics_r.json`
  - Contains per-camera extrinsic estimates.
  - Each file provides `cam_pose` and `cam_translation_vector`.

## Outputs

- `data/stats/mean_extrinsics.json`
  - The averaged extrinsic pose and translation used by the evaluation.

- `data/stats/geom_mle_intrinsics.json`
  - The fitted unified intrinsic model.
  - Also stores optimizer statistics and the fitting residual summary.

- `data/stats/xy_error_plot.html`
  - A flat XY grid plot in the ground plane.
  - It draws the unified/reference XY grid and overlays the XY points recovered by each independent intrinsic model.
  - This is the full-points plot and uses all available intrinsics.

- `data/stats/xy_error_plot.png`
  - A static PNG version of the full-points XY grid plot.

- `data/stats/xy_error_sampled_plot.html`
  - A sampled XY grid plot for readability.
  - The scatter overlay uses 5 deterministic sampled intrinsics.
  - Points are colored by reference XY grid point: all recovered samples belonging to the same grid point share one color.

- `data/stats/xy_error_sampled_plot.png`
  - A static PNG version of the sampled, grid-point-colored XY grid plot.

- `data/stats/error_stats_plot.html`
  - An interactive statistical analysis plot for `geom_mle_unified` directional BEV error.
  - It plots `|dx| p95`, `|dy| p95`, and XY norm p95 by y-depth bins and x-lateral bins.

- `data/stats/error_stats_plot.png`
  - A static PNG version of the `geom_mle_unified` statistical analysis plot.

- `data/stats/error_stats_p95.csv`
  - A tidy CSV table for `geom_mle_unified` statistical analysis.
  - It keeps only p95 metrics: `dx_abs_p95_m`, `dy_abs_p95_m`, and `xy_norm_p95_m`.
  - It includes one `overall` row plus rows for each y-depth bin and x-lateral bin.

## Coordinate and Camera Setup

The script first averages the extrinsics:

- Height and translation are averaged directly.
- Pitch, roll, and yaw are averaged with circular mean to avoid angle wrapping issues.

It then builds:

- `camera_xyz`: camera position in the world/body frame.
- `R_cw`: world-to-camera rotation.
- `t_cw`: world-to-camera translation.

The ground plane is assumed to be:

```text
z = 0
```

The sampled BEV grid is:

```text
x: -4.5m to 4.5m, step 1.5m
y:  0.0m to 40.0m, step 5.0m
```

Boundary points are included, so the current grid contains:

```text
9 x 7 = 63 ground points
```

The `y=0m` row is sampled as a boundary row but is excluded from fitting,
statistics, and plots. At this row the ground points are near the camera
optical-plane horizon for the current extrinsics, so fisheye inverse projection
can become numerically unstable and can dominate the closest depth bin. The
current analysis grid is therefore:

```text
8 x 7 = 56 ground points, with y > 0m
```

## Projection and Backprojection

For projection, the script reuses the IPM utilities:

- `vp.ipm.IPM.TransformGround2Image`
- `vp.ipm.IPM.SpaceToPlane`

This maps sampled world ground points into image pixels:

```text
ground point (x, y, z=0) -> image point (u, v)
```

For backprojection, the script:

1. Undistorts image points using the candidate intrinsic model.
2. Converts undistorted normalized coordinates into camera rays.
3. Transforms rays into the world frame.
4. Intersects each ray with the ground plane `z=0`.
5. Produces a recovered ground position `pred_xy`.

The BEV error is measured in meters:

```text
residual = pred_xy - original_xy
```

## Unified Intrinsic Fitting

The fitted unified model is named:

```text
geom_mle_unified
```

The optimizer estimates 8 parameters:

```text
fx, fy, cx, cy, k1, k2, k3, k4
```

The initial value is the arithmetic mean of all camera intrinsics.

The optimization bounds are derived from the min/max parameter values across all cameras, expanded by 50%. The focal lengths are constrained to stay positive, and the principal point is constrained to remain inside the image.

The fitting observations are generated as follows:

1. For each camera intrinsic calibration, project the shared ground grid into image UV.
2. Keep only UV points that are inside the image.
3. Store each valid pair:

```text
(original ground xy, projected uv)
```

During optimization, the candidate unified intrinsic model backprojects those UV points to the ground plane. The least-squares objective minimizes the ground-plane XY residuals over all cameras and visible grid points.

In probabilistic terms, the script treats the BEV XY residuals as independent isotropic Gaussian errors. Under that assumption, minimizing squared residuals is equivalent to maximum likelihood estimation, which is why the output is called `geometric_mle`.

## Comparison Logic

The script compares two paths.

### 1. Independent Intrinsics

Each camera uses its own intrinsic model for both projection and backprojection:

```text
ground xy
  -> project with that camera's own intrinsics
  -> uv
  -> backproject with the same camera's own intrinsics
  -> recovered xy
```

This is a self-consistency check. The error should be near zero because the same model is used in both directions.

### 2. Unified Geometric MLE Intrinsics

Each camera still generates UV observations using its own intrinsic model, but those UV points are backprojected using the unified model:

```text
ground xy
  -> project with each camera's own intrinsics
  -> uv
  -> backproject with geom_mle_unified
  -> recovered xy
```

The script then compares:

```text
recovered xy vs original ground xy
```

This measures how much BEV ground-plane error is introduced when the unified intrinsic model is used as a replacement for each camera's original intrinsics.

The XY grid plot uses a reference grid in ground-plane coordinates:

```text
sampled unified/reference xy grid
  -> project to uv with geom_mle_unified
  -> backproject uv to xy with each independent intrinsic model
  -> draw recovered independent xy points around the reference xy grid
```

The black grid is the reference XY grid. The full-points plot draws all
independent-intrinsic reprojected XY samples. The sampled plot draws the same
geometry with only 5 deterministic sampled intrinsics.
The plotted reference grid uses the filtered analysis grid, so the unstable
`y=0m` boundary row is not drawn.

For the sampled XY grid plot only, each reference XY grid point owns one color,
and every recovered independent-intrinsic point associated with that same grid
point is drawn in that color. This keeps the classification view readable while
leaving fitting, statistical evaluation, and the full-points plot unchanged.

The unified path also applies validity checks:

- The original projected UV must be inside the image.
- The unified backprojection must produce a valid ground-plane intersection.
- The predicted ground point must reproject back to the original UV within `MAX_REPROJECT_ERROR_PX`, currently `1.0px`.

## Metrics

Errors are summarized as 2D ground-plane vectors, not only as scalar distance. The report includes:

- `overall`
- `by_y_m`: depth intervals using `Y_STEP_M`, currently `5m`
- `by_x_m`: lateral intervals using `X_STEP_M`, currently `1.5m`

For y bins, the intervals are:

```text
0..5m, 5..10m, 10..15m, ...
```

Because the `y=0m` row is excluded from analysis, the first `0..5m` bin only
contains the `y=5m` grid row in the current setup.

For x bins, the intervals are:

```text
-4.5..-3m, -3..-1.5m, ..., 3..4.5m
```

Boundary handling is `[start, end]` for the first bin and `(start, end]` for later bins.

Each non-empty bin reports:

- `count`
- `dx_m`: signed `mean`/`median`, plus `abs_median`, `abs_p90`, `abs_p95`
- `dy_m`: signed `mean`/`median`, plus `abs_median`, `abs_p90`, `abs_p95`
- `xy_norm_m`: scalar error norm `median`, `p90`, `p95`

The printed summaries, plots, and fitting statistics stored inside `geom_mle_intrinsics.json` all use meters.

The CSV table is designed for spreadsheet or plotting use. Its columns are:

```text
model, group, axis, range_m, axis_min_m, axis_max_m, axis_center_m,
count, dx_abs_p95_m, dy_abs_p95_m, xy_norm_p95_m
```

## Current Example Result

On the current dataset, the script reports:

```text
Parsed intrinsics: 60 cameras
Parsed extrinsics: 18 files
Ground samples: 63 points, shape=(9, 7)
Analysis ground samples: 56 points, shape=(8, 7), excluded y <= 0m
Camera xyz: (-0.383, 0.073, 0.594)m
geom_mle optimizer: success=True, nfev=37
independent:
  overall: count=3360, dx_med=0.0000m, dx_abs_p95=0.0000m, dy_med=0.0000m, dy_abs_p95=0.0000m, norm_p95=0.0000m
  by_y:
    0..5m: count=420, dx_med=0.0000m, dx_abs_p90=0.0000m, dx_abs_p95=0.0000m, dy_med=0.0000m, dy_abs_p90=0.0000m, dy_abs_p95=0.0000m, norm_p95=0.0000m
  ...
geom_mle_unified:
  overall: count=3157, dx_med=0.0001m, dx_abs_p95=5.0306m, dy_med=-0.2508m, dy_abs_p95=46.9145m, norm_p95=47.2096m
  by_y:
    0..5m: count=420, dx_med=0.0001m, dx_abs_p90=0.7586m, dx_abs_p95=1.0842m, dy_med=0.0032m, dy_abs_p90=1.3116m, dy_abs_p95=1.5622m, norm_p95=1.8298m
  by_x:
    -4.5..-3m: count=902, dx_med=0.0738m, dx_abs_p90=2.9791m, dx_abs_p95=6.2084m, dy_med=-0.2556m, dy_abs_p90=24.2099m, dy_abs_p95=48.0106m, norm_p95=48.1599m
  ...
```

The independent path has nearly zero error because it uses each camera's own intrinsics for both projection and backprojection.

The unified path has much larger BEV error. This indicates that replacing all per-camera intrinsics with one unified fisheye intrinsic model causes significant ground-plane localization error under the current assumptions.

## Important Interpretation Notes

- This is a geometry-only evaluation.
- It does not evaluate image quality, real BEV stitching quality, or real feature reprojection error.
- The comparison depends heavily on the sampled ground range, especially far-range points.
- The result also depends on using a single averaged extrinsic pose for all cameras.
- Large errors can come from true intrinsic variation, from using one averaged extrinsic, from far-ground sensitivity, or from the chosen optimization objective.

## High-Level Takeaway

The script asks one specific question:

```text
Can one unified fisheye intrinsic model reproduce the BEV ground-plane geometry
of many independently calibrated cameras?
```

For the current data and grid range, the answer appears to be no: each camera's own intrinsic model is self-consistent, while the unified model introduces large BEV ground-plane errors.
