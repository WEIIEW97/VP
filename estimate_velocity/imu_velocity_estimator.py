import numpy as np
from collections import deque


class IMUVelocityEstimator:
    """Estimate velocity from IMU acceleration data"""
    
    def __init__(self, window_size=20, gravity=9.81):
        """
        Args:
            window_size: number of samples for moving average filter
            gravity: gravitational acceleration in m/s^2
        """
        self.window_size = window_size
        self.gravity = gravity
        
        self.velocity = 0.0
        self.prev_timestamp = None
        
        self.accel_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        
    def estimate(self, accel_xyz, timestamp, orientation_matrix=None):
        """
        Estimate velocity from IMU acceleration
        
        Args:
            accel_xyz: 3D acceleration vector [ax, ay, az] in m/s^2 (body frame)
            timestamp: timestamp in seconds
            orientation_matrix: 3x3 rotation matrix from body to world frame (optional)
            
        Returns:
            velocity: estimated velocity in km/h
        """
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return 0.0
            
        dt = timestamp - self.prev_timestamp
        
        if dt <= 0 or dt > 1.0:
            self.prev_timestamp = timestamp
            return self.velocity * 3.6
            
        # Transform acceleration to world frame if orientation is available
        if orientation_matrix is not None:
            accel_world = orientation_matrix @ accel_xyz
            # Remove gravity (assuming z-axis is up in world frame)
            accel_world[2] -= self.gravity
        else:
            # Simple approximation: assume forward acceleration is the x-component
            accel_world = accel_xyz.copy()
            
        # Forward acceleration (assuming x is forward in body frame)
        accel_forward = accel_world[0]
        
        # Add to buffer and apply moving average filter
        self.accel_buffer.append(accel_forward)
        filtered_accel = np.mean(self.accel_buffer)
        
        # Integrate acceleration to velocity
        self.velocity += filtered_accel * dt
        
        # Apply decay to prevent drift (complementary filter)
        decay_factor = 0.98
        self.velocity *= decay_factor
        
        # Add to velocity buffer
        self.velocity_buffer.append(abs(self.velocity))
        
        # Use median velocity to reduce noise
        velocity_mps = np.median(self.velocity_buffer) if len(self.velocity_buffer) > 5 else abs(self.velocity)
        
        self.prev_timestamp = timestamp
        
        # Convert to km/h
        return velocity_mps * 3.6
    
    def estimate_from_magnitude(self, accel_magnitude, timestamp):
        """
        Simplified estimation using acceleration magnitude
        
        Args:
            accel_magnitude: magnitude of acceleration in m/s^2
            timestamp: timestamp in seconds
            
        Returns:
            velocity classification based on acceleration pattern
        """
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return 0.0
            
        dt = timestamp - self.prev_timestamp
        
        if dt <= 0 or dt > 1.0:
            self.prev_timestamp = timestamp
            return self.velocity * 3.6
            
        # Remove gravity baseline
        accel_net = abs(accel_magnitude - self.gravity)
        
        self.accel_buffer.append(accel_net)
        
        # Calculate velocity change heuristic
        if len(self.accel_buffer) >= self.window_size:
            avg_accel = np.mean(self.accel_buffer)
            std_accel = np.std(self.accel_buffer)
            
            # Heuristic: higher acceleration variation indicates higher speed
            velocity_indicator = avg_accel + std_accel * 2.0
            
            # Map to velocity (empirical scaling)
            velocity_mps = velocity_indicator * 5.0
            
            self.prev_timestamp = timestamp
            return velocity_mps * 3.6
        
        self.prev_timestamp = timestamp
        return 0.0
    
    def reset(self):
        """Reset estimator state"""
        self.velocity = 0.0
        self.prev_timestamp = None
        self.accel_buffer.clear()
        self.velocity_buffer.clear()
