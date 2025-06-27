import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

pd.set_option("display.float_format", "{:.3f}".format)

from ipm import *

def plot_pose_analysis(recorder):
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Reference Points Movement (2D Trajectory)
    ax1 = fig.add_subplot(gs[0, 0])
    conditions = recorder['condition'].unique()
    
    for cond in conditions:
        if cond == 'original':
            continue
        subset = recorder[recorder['condition'] == cond]
        for i in range(1, 9):
            points = np.array(subset[f'ref{i}'].tolist())
            ax1.plot(points[:, 0], points[:, 1], 'o-', 
                    label=f'{cond} - ref{i}')
    
    original_points = np.array(recorder.iloc[0][['ref1','ref2','ref3','ref4',
                                               'ref5','ref6','ref7','ref8']].tolist())
    ax1.plot(original_points[:, 0], original_points[:, 1], 'kx', 
            markersize=10, label='Original')
    
    ax1.set_title('Reference Points Movement')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # Plot 2: Delta Magnitude vs Angle Offset
    ax2 = fig.add_subplot(gs[0, 1])
    for cond in conditions:
        if cond == 'original':
            continue
        subset = recorder[recorder['condition'] == cond]
        deltas = np.array(subset['delta'].tolist())
        delta_mags = np.linalg.norm(deltas, axis=1)
        ax2.plot(subset['angle_offset'], delta_mags, 'o-', 
                label=cond)
    
    ax2.set_title('Delta Magnitude vs Angle Offset')
    ax2.set_xlabel('Angle Offset (degrees)')
    ax2.set_ylabel('Delta Magnitude')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Individual Delta Components
    ax3 = fig.add_subplot(gs[1, :])
    for cond in conditions:
        if cond == 'original':
            continue
        subset = recorder[recorder['condition'] == cond]
        deltas = np.array(subset['delta'].tolist())
        for i, component in enumerate(['X', 'Y']):
            ax3.plot(subset['angle_offset'], deltas[:, i], 'o-',
                    label=f'{cond} {component}')
    
    ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title('Delta Components (X/Y) vs Angle Offset')
    ax3.set_xlabel('Angle Offset (degrees)')
    ax3.set_ylabel('Delta Value')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_pose_degree_offset():
    K_src = [1024.439310, 0, 971.522106, 0, 1024.705578, 509.292897, 0, 0, 1]
    dist = [
        1.334315,
        0.540933,
        -0.000252,
        0.000129,
        0.022290,
        1.751091,
        0.995260,
        0.140756,
    ]
    image_size = [1920, 1080]
    K_src = np.array(K_src).reshape(3, 3)
    dist = np.array(dist)
    image_size = np.array(image_size)

    tx = -0.003224691764243655
    ty = 0.042556410188584026
    tz = 0.5820035505399808

    pose_gather = PoseGather()
    ext_ypr = pose_gather.get_pose_imu(
        3.1726789812509337, 3.0963451407720486, -0.06777965451125129
    )

    ipm = IPM(K_src, dist, image_size, ext_ypr[0], ext_ypr[1], ext_ypr[2], tx, ty, tz)

    grid_xy = [
        [-4.5, 20],
        [-0.5, 20],
        [0.5, 20],
        [4.5, 20],
        [-4.5, 5],
        [-0.5, 5],
        [0.5, 5],
        [4.5, 5],
    ]
    grid_xy = np.array(grid_xy).reshape((-1, 2))

    imu_pitch = 3.0963451862335205
    imu_roll = 0.02717495107268181
    imu_yaw = 3.1726789474487305

    ypr = pose_gather.get_pose_imu(imu_yaw, imu_pitch, imu_roll)

    grid_uv = ipm.ProjectBEVXYs2PointUVs(
        grid_xy, yaw_c_g=ypr[0], pitch_c_g=ypr[1], roll_c_g=ypr[2]
    )
    # inv_grid_xy = ipm.ProjectPointsUV2BEVXY(grid_uv, yaw_c_g=ypr[0], pitch_c_g=ypr[1], roll_c_g=ypr[2])
    lb, ub, step = -3, 3, 0.5
    angle_offset = np.arange(lb, ub + step, step)

    conditions = ["yaw(roll)", "pitch(yaw)", "roll(pitch)"]

    recorder = pd.DataFrame(
        columns=[
            "condition",
            "angle_offset",
            "ref1",
            "ref2",
            "ref3",
            "ref4",
            "ref5",
            "ref6",
            "ref7",
            "ref8",
        ]
    )

    # Set 'original' condition (baseline)
    recorder.loc[0] = {
        "condition": "original",
        "angle_offset": 0,
        "ref1": list(grid_xy[0, :]),  # Store as [x, y]
        "ref2": list(grid_xy[1, :]),
        "ref3": list(grid_xy[2, :]),
        "ref4": list(grid_xy[3, :]),
        "ref5": list(grid_xy[4, :]),
        "ref6": list(grid_xy[5, :]),
        "ref7": list(grid_xy[6, :]),
        "ref8": list(grid_xy[7, :]),
    }

    # Get original points for delta calculation
    original_points = grid_xy.copy()

    for condition in conditions:
        for offset in angle_offset:
            if condition == "yaw(roll)":
                grid_xy_offset = ipm.ProjectPointsUV2BEVXY(
                    grid_uv, yaw_c_g=ypr[0] + offset, pitch_c_g=ypr[1], roll_c_g=ypr[2]
                )
            elif condition == "pitch(yaw)":
                grid_xy_offset = ipm.ProjectPointsUV2BEVXY(
                    grid_uv, yaw_c_g=ypr[0], pitch_c_g=ypr[1] + offset, roll_c_g=ypr[2]
                )
            elif condition == "roll(pitch)":
                grid_xy_offset = ipm.ProjectPointsUV2BEVXY(
                    grid_uv, yaw_c_g=ypr[0], pitch_c_g=ypr[1], roll_c_g=ypr[2] + offset
                )

            # Calculate delta (e.g., mean offset across all points)
            # delta = np.mean(grid_xy_offset - original_points, axis=0)

            # Create new row
            new_row = pd.DataFrame(
                [
                    {
                        "condition": condition,
                        "angle_offset": offset,
                        "ref1": [
                            round(grid_xy_offset[0, 0], 3),
                            round(grid_xy_offset[0, 1], 3),
                        ],
                        "ref2": [
                            round(grid_xy_offset[1, 0], 3),
                            round(grid_xy_offset[1, 1], 3),
                        ],
                        "ref3": [
                            round(grid_xy_offset[2, 0], 3),
                            round(grid_xy_offset[2, 1], 3),
                        ],
                        "ref4": [
                            round(grid_xy_offset[3, 0], 3),
                            round(grid_xy_offset[3, 1], 3),
                        ],
                        "ref5": [
                            round(grid_xy_offset[4, 0], 3),
                            round(grid_xy_offset[4, 1], 3),
                        ],
                        "ref6": [
                            round(grid_xy_offset[5, 0], 3),
                            round(grid_xy_offset[5, 1], 3),
                        ],
                        "ref7": [
                            round(grid_xy_offset[6, 0], 3),
                            round(grid_xy_offset[6, 1], 3),
                        ],
                        "ref8": [
                            round(grid_xy_offset[7, 0], 3),
                            round(grid_xy_offset[7, 1], 3),
                        ],
                        # "delta": [round(delta[0], 3), round(delta[1], 3)],
                    }
                ]
            )

            recorder = pd.concat([recorder, new_row], ignore_index=True)
    return recorder


if __name__ == "__main__":
    rec = test_pose_degree_offset()
    csv_path = '../data/angle_pose_offset_analysis.csv'
    rec.to_csv(csv_path, index=False)
