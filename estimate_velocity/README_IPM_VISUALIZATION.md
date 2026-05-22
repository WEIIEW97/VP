# IPM (Inverse Perspective Mapping) Visualization

## Overview

The velocity estimation demo now includes a **bird's-eye view** (IPM) visualization that shows:
- Ground features projected to real-world coordinates
- Optical flow velocity vectors
- Distance rings for spatial reference
- Color-coded features by distance

## Visualization Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    Camera View                │   IPM Bird's-Eye │
│  ┌─────────────────────────────────────┐    │   ┌─────────────┐│
│  │                                      │    │   │    40m      ││
│  │  Detection Region (65%)  ← gray     │    │   │  ●  ●  ●    ││
│  │  ─────────────────────────────────  │    │   │  ●  ●  ●    ││
│  │  Ground Focus (70%)      ← cyan     │    │   │  ●  ●  ●    ││
│  │  ═════════════════════════════════  │    │   │  ●  ●  ●  20m│
│  │                                      │    │   │  ●  ●  ●    ││
│  │  ● Feature points (blue)            │    │   │  ●  ●  ●    ││
│  │  → Flow vectors (green arrows)      │    │   │  ●  ●  ●  10m│
│  │                                      │    │   │    ●  ●     ││
│  └─────────────────────────────────────┘    │   │    ● ● ●  5m││
│                                              │   │      ▲      ││
│  Velocity: 3.5 km/h (GT: 3.5)              │   │   Camera    ││
│  Opt:3.2 IMU:3.4 flow:1.234px              │   └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Camera View (Left)
- **Blue dots**: All detected feature points
- **Green arrows**: Optical flow vectors for ground features
- **Gray line (65%)**: Detection region boundary
- **Cyan line (70%)**: Ground focus region (features below this are used)
- Shows actual camera perspective with annotations

### IPM Bird's-Eye View (Right)
- **Coordinate System**: 
  - Camera at bottom center (cyan circle)
  - Forward direction: upward
  - Lateral: left/right
  - Scale: ~5cm/pixel horizontal, ~8cm/pixel vertical

- **Distance Rings**: 
  - Circles at 5m, 10m, 15m, 20m, 30m, 40m
  - Labeled with distance
  - Grid lines for lateral position

- **Feature Points**:
  - Color-coded by distance:
    - **Green**: Near (< 10m)
    - **Yellow-Orange**: Mid (10-20m)
    - **Red**: Far (> 20m)
  - Size: 4 pixels
  
- **Velocity Vectors**:
  - Green arrows showing motion direction
  - Length proportional to optical flow magnitude
  - Point upward for forward motion

## Usage

The IPM visualization is enabled by default when `show_flow=True`:

```python
# Automatic in demo.py
demo_motorev_data(data_folder, camera_height=1.2, show_flow=True)
```

### Interactive Controls

- **Space/Enter**: Pause/Resume
- **'q'**: Quit visualization
- **Window**: Drag to reposition

## Technical Details

### Distance Calculation

Features are projected to ground plane using empirical distance mapping:

```python
# Empirical distance model for forward-facing dashcam
# (camera_height ~1.2m, typical pitch ~5-10°)

Image Row (y-normalized)  →  Ground Distance
─────────────────────────────────────────────
0.65 - 0.75 (far ground)   →  20-40m
0.75 - 0.85 (mid ground)   →  8-20m  
0.85 - 1.00 (near ground)  →  3-8m
```

### Lateral Position

Calculated using pinhole camera model:

```
x_ground = distance × (x_img - cx) / fx
```

Where:
- `x_img`: pixel x-coordinate
- `cx`: principal point x (960px)
- `fx`: focal length x (1036px)
- `distance`: forward distance (from empirical mapping)

### Coordinate Transform

```
Camera Space → Ground Plane (Bird's-Eye)
────────────────────────────────────────
(x_img, y_img, depth) → (x_ground, z_ground)

Canvas coordinates:
canvas_x = camera_x + x_ground / meters_per_pixel_x
canvas_y = camera_y - z_ground / meters_per_pixel_y
```

## Interpretation Guide

### Good Velocity Estimation
```
IPM View Shows:
✓ 120-150 features distributed across 3-40m
✓ Feature density highest at 5-15m (most reliable)
✓ Velocity vectors consistent direction/magnitude
✓ No outlier features far from cluster
```

### Poor Velocity Estimation
```
IPM View Shows:
✗ < 50 features (insufficient data)
✗ All features clustered at one distance
✗ Vectors point in random directions
✗ Many outliers far from main cluster
```

## Validation

Use the IPM view to validate:

1. **Feature Distribution**: Should span 3-40m range
2. **Ground Focus**: Most features at 5-15m (sweet spot)
3. **Flow Consistency**: Vectors should be parallel (forward motion)
4. **Distance Realism**: Check if distances match scene geometry

## Example Output

For 3-4 km/h dataset:
```
Ground features: 127, distances: 2.5-7.0m (mean:3.5m)
Final velocity: 2.24 km/h (raw=2.24, scale=1.000 [initial])

IPM shows:
- ~127 green/yellow dots clustered 3-7m ahead
- Short upward arrows (small flow)
- Symmetric distribution (straight motion)
```

For 20 km/h dataset:
```
Ground features: 132, distances: 3.0-34.4m (mean:8.4m)
Final velocity: 5.00 km/h (raw=5.00, scale=1.000 [initial])

IPM shows:
- ~132 dots spanning 3-35m
- Longer upward arrows (larger flow)
- Wide distribution (features at many distances)
```

## Troubleshooting

### Issue: IPM view shows no features
**Cause**: No features detected in ground region  
**Fix**: Check detection region boundaries, lower `qualityLevel`

### Issue: Features clustered at one distance
**Cause**: Camera pitch angle incorrect for empirical mapping  
**Fix**: Adjust distance mapping in `_calculate_ground_distance()`

### Issue: Velocity vectors point sideways
**Cause**: Lateral motion or incorrect feature tracking  
**Fix**: Check camera stability, improve feature selection

### Issue: Outlier features at 40m+
**Cause**: Features on distant objects (not ground)  
**Fix**: Already filtered by sanity check (< 50m)

## Future Enhancements

- [ ] True IPM using camera pitch angle (from IMU or calibration)
- [ ] Velocity heatmap overlay
- [ ] Feature lifetime tracking (show persistence)
- [ ] Distance histogram visualization
- [ ] Export IPM frames to video

## References

- Camera intrinsics: `camera_intrinsics` dict in `demo.py`
- Distance mapping: `_calculate_ground_distance()` in `optical_flow_estimator.py`
- IPM rendering: `create_ipm_birdseye_view()` in `demo.py`
