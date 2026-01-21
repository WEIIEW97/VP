import numpy as np
from scipy.optimize import least_squares

def skew_symmetric(v):
    """Compute the skew-symmetric matrix of a vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def rodrigues_exp(phi):
    """Rodrigues formula: Map from Lie algebra so(3) to Lie group SO(3)"""
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3)
    axis = phi / angle
    K = skew_symmetric(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def euler_to_rotation(roll, pitch, yaw, order='xyz'):
    """Euler angles to rotation matrix (XYZ order: roll-x, pitch-y, yaw-z)"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    if order == 'xyz':
        return Rz @ Ry @ Rx  # Note: actual application order is reverse of multiplication order
    elif order == 'zyx':
        return Rx @ Ry @ Rz
    else:
        raise ValueError("Unsupported Euler order")

def rotation_to_euler(R, order='xyz'):
    """Rotation matrix to Euler angles"""
    if order == 'xyz':
        # Extract Euler angles in XYZ order
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0
    else:
        # Implementation for other orders
        pass
    
    return roll, pitch, yaw

def solve_refined_R_fixed_roll(R_bc_init, yaw_obs, pitch_obs, method='optimization'):
    """
    Fix roll angle, only optimize yaw and pitch
    
    Args:
        R_bc_init: Initial rotation matrix (camera->IMU)
        yaw_obs: Observed yaw angle (radians)
        pitch_obs: Observed pitch angle (radians)
        method: Optimization method ['direct', 'optimization']
    """
    
    # Method 1: Direct calculation (fixed roll)
    if method == 'direct':
        # Extract current roll angle
        roll_curr, _, _ = rotation_to_euler(R_bc_init, order='xyz')
        
        # Use observed yaw/pitch, keep current roll
        R_refined = euler_to_rotation(roll_curr, pitch_obs, yaw_obs, order='xyz')
        
        return R_refined
    
    # Method 2: Optimization method (recommended)
    elif method == 'optimization':
        # Extract current Euler angles
        roll_curr, pitch_curr, yaw_curr = rotation_to_euler(R_bc_init, order='xyz')
        
        # Observed direction vector (camera coordinate system)
        d_cam_obs = np.array([
            np.cos(pitch_obs) * np.sin(yaw_obs),
            -np.sin(pitch_obs),
            np.cos(pitch_obs) * np.cos(yaw_obs)
        ])
        
        # Target direction (IMU coordinate system, forward)
        d_imu_target = np.array([0, 0, 1])
        
        # Define optimization problem: only optimize yaw and pitch, fix roll
        def cost_function(params):
            # params: [yaw, pitch]
            yaw_new, pitch_new = params
            
            # Build rotation matrix with new yaw/pitch and fixed roll
            R_new = euler_to_rotation(roll_curr, pitch_new, yaw_new, order='xyz')
            
            # Calculate direction vector
            d_imu_pred = R_new @ d_cam_obs
            
            # Calculate error
            error = d_imu_target - d_imu_pred
            
            # Can also add smoothing term to avoid sudden changes
            smooth_weight = 0.1
            smooth_error = smooth_weight * np.array([
                yaw_new - yaw_curr,
                pitch_new - pitch_curr
            ])
            
            return np.concatenate([error, smooth_error])
        
        # Initial guess
        x0 = [yaw_curr, pitch_curr]
        
        # Set bounds (optional)
        bounds = ([-np.pi, -np.pi/2], [np.pi, np.pi/2])
        
        # Optimize
        result = least_squares(cost_function, x0, bounds=bounds, method='trf')
        
        # Extract optimized parameters
        yaw_opt, pitch_opt = result.x
        
        # Reconstruct rotation matrix
        R_refined = euler_to_rotation(roll_curr, pitch_opt, yaw_opt, order='xyz')
        
        return R_refined
    
    else:
        raise ValueError(f"Unknown method: {method}")

# Method 3: Incremental optimization (more stable version)
def solve_refined_R_incremental(R_bc_init, yaw_obs, pitch_obs, alpha=0.1):
    """
    Incremental update method, combining current values and observations
    
    Args:
        alpha: Learning rate, controls update speed
    """
    # Extract current Euler angles
    roll_curr, pitch_curr, yaw_curr = rotation_to_euler(R_bc_init, order='xyz')
    
    # Smooth update
    yaw_new = (1 - alpha) * yaw_curr + alpha * yaw_obs
    pitch_new = (1 - alpha) * pitch_curr + alpha * pitch_obs
    
    # Reconstruct rotation matrix (keep roll unchanged)
    R_refined = euler_to_rotation(roll_curr, pitch_new, yaw_new, order='xyz')
    
    return R_refined

# --- Test code ---
if __name__ == "__main__":
    # Test data
    # Assume initial extrinsic parameters have errors
    R_init = euler_to_rotation(
        np.radians(0.5),   # roll: 0.5度
        np.radians(1.0),   # pitch: 1.0度  
        np.radians(2.0),   # yaw: 2.0度
        order='xyz'
    )
    
    # Observed lane angles
    yaw_obs = np.radians(1.5)   # Observed yaw: 1.5 degrees
    pitch_obs = np.radians(0.8) # Observed pitch: 0.8 degrees
    
    print("Initial rotation matrix:")
    print(R_init)
    
    # Test different methods
    methods = ['direct', 'optimization', 'incremental']
    
    for method in methods:
        if method == 'incremental':
            R_refined = solve_refined_R_incremental(R_init, yaw_obs, pitch_obs, alpha=0.3)
        else:
            R_refined = solve_refined_R_fixed_roll(R_init, yaw_obs, pitch_obs, method=method)
        
        # Extract Euler angles for inspection
        roll_ref, pitch_ref, yaw_ref = rotation_to_euler(R_refined, order='xyz')
        
        print(f"\nMethod '{method}' result:")
        print(f"Roll: {np.degrees(roll_ref):.2f}°, Pitch: {np.degrees(pitch_ref):.2f}°, Yaw: {np.degrees(yaw_ref):.2f}°")
        
        # Verify direction
        d_cam_obs = np.array([
            np.cos(pitch_obs) * np.sin(yaw_obs),
            -np.sin(pitch_obs),
            np.cos(pitch_obs) * np.cos(yaw_obs)
        ])
        
        d_imu = R_refined @ d_cam_obs
        print(f"Direction vector: {d_imu}")
        print(f"Dot product with forward [0,0,1]: {d_imu[2]:.6f}")