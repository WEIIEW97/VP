import numpy as np

def skew_symmetric(v):
    """Skew-symmetric matrix (wedge operator)"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def rodrigues_exp(phi):
    """Exponential map from so(3) to SO(3)"""
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3)
    axis = phi / angle
    K = skew_symmetric(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def solve_refined_R(R_bc_init, yaw_obs, pitch_obs, iterations=10):
    """
    Optimize camera-to-body rotation R_bc using observed yaw/pitch from lanes.
    Convention: d_body = R_bc * d_cam
    """

    # 1. Construct observed vanishing direction in camera frame
    # Camera frame: x right, y down, z forward
    d_cam = np.array([
        np.cos(pitch_obs) * np.sin(yaw_obs),
        -np.sin(pitch_obs),
        np.cos(pitch_obs) * np.cos(yaw_obs)
    ])
    d_cam = d_cam / np.linalg.norm(d_cam)

    # 2. Target direction in body frame (assume forward)
    d_body_target = np.array([0, 0, 1])

    R_bc_current = R_bc_init.copy()

    for i in range(iterations):
        # Predicted direction in body frame
        d_body_pred = R_bc_current @ d_cam
        d_body_pred = d_body_pred / np.linalg.norm(d_body_pred)

        # Residual: direction error (use cross product for better geometry)
        residual = np.cross(d_body_pred, d_body_target)

        # Jacobian: -[d_pred]_x
        J = -skew_symmetric(d_body_pred)

        # Solve J * delta_phi = residual (least squares)
        delta_phi, _, _, _ = np.linalg.lstsq(J, residual, rcond=None)

        # Update rotation (left perturbation)
        R_bc_current = rodrigues_exp(delta_phi) @ R_bc_current

        # Convergence check
        cost = np.linalg.norm(residual)
        if cost < 1e-10:
            break

    return R_bc_current


if __name__ == "__main__":
    R_initial = np.eye(3)
    y_obs = np.radians(2.0)
    p_obs = np.radians(1.0)

    refined_R = solve_refined_R(R_initial, y_obs, p_obs)

    print("Initial R:\n", R_initial)
    print("Refined R:\n", refined_R)

    # Check: refined_R @ d_cam should be close to [0, 0, 1]
    d_cam_test = np.array([
        np.cos(p_obs) * np.sin(y_obs),
        -np.sin(p_obs),
        np.cos(p_obs) * np.cos(y_obs)
    ])
    print("Check result:", refined_R @ d_cam_test)
