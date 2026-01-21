import numpy as np
from collections import deque


class VelocityFusion:
    """Fuse velocity estimates from multiple sources"""
    
    def __init__(self, alpha_optical=0.6, alpha_imu=0.4, buffer_size=10):
        """
        Args:
            alpha_optical: weight for optical flow estimate
            alpha_imu: weight for IMU estimate
            buffer_size: size of velocity buffer for smoothing
        """
        self.alpha_optical = alpha_optical
        self.alpha_imu = alpha_imu
        self.buffer_size = buffer_size
        
        self.velocity_buffer = deque(maxlen=buffer_size)
        self.last_valid_velocity = 0.0
        
    def fuse(self, optical_velocity, imu_velocity, optical_confidence=1.0, imu_confidence=1.0):
        """
        Fuse optical flow and IMU velocity estimates
        
        Args:
            optical_velocity: velocity from optical flow in km/h (can be None)
            imu_velocity: velocity from IMU in km/h (can be None)
            optical_confidence: confidence in optical flow estimate [0, 1]
            imu_confidence: confidence in IMU estimate [0, 1]
            
        Returns:
            fused_velocity: fused velocity estimate in km/h
        """
        # Handle missing estimates
        if optical_velocity is None and imu_velocity is None:
            return self.last_valid_velocity
            
        if optical_velocity is None:
            fused = imu_velocity
        elif imu_velocity is None:
            fused = optical_velocity
        else:
            # Weighted fusion
            total_weight = self.alpha_optical * optical_confidence + self.alpha_imu * imu_confidence
            
            if total_weight > 0:
                fused = (
                    self.alpha_optical * optical_confidence * optical_velocity +
                    self.alpha_imu * imu_confidence * imu_velocity
                ) / total_weight
            else:
                fused = (optical_velocity + imu_velocity) / 2.0
                
        # Add to buffer
        self.velocity_buffer.append(fused)
        
        # Smooth using median filter
        if len(self.velocity_buffer) >= 3:
            smoothed = np.median(self.velocity_buffer)
        else:
            smoothed = fused
            
        self.last_valid_velocity = smoothed
        
        return smoothed
    
    def adaptive_fuse(self, optical_velocity, imu_velocity):
        """
        Adaptive fusion that automatically adjusts weights based on consistency
        
        Args:
            optical_velocity: velocity from optical flow in km/h
            imu_velocity: velocity from IMU in km/h
            
        Returns:
            fused_velocity: fused velocity estimate in km/h
        """
        if optical_velocity is None and imu_velocity is None:
            return self.last_valid_velocity
            
        if optical_velocity is None:
            return imu_velocity
            
        if imu_velocity is None:
            return optical_velocity
            
        # Calculate consistency score
        diff = abs(optical_velocity - imu_velocity)
        max_val = max(optical_velocity, imu_velocity, 1.0)
        consistency = 1.0 - min(diff / max_val, 1.0)
        
        # Adjust weights based on consistency
        if consistency > 0.7:
            # High consistency: trust both sources
            weight_optical = 0.6
            weight_imu = 0.4
        elif optical_velocity > 30.0:
            # High speed: trust optical flow more
            weight_optical = 0.7
            weight_imu = 0.3
        elif optical_velocity < 10.0:
            # Low speed: trust IMU more (optical flow less reliable)
            weight_optical = 0.3
            weight_imu = 0.7
        else:
            # Medium speed: balanced
            weight_optical = 0.5
            weight_imu = 0.5
            
        fused = weight_optical * optical_velocity + weight_imu * imu_velocity
        
        # Add to buffer and smooth
        self.velocity_buffer.append(fused)
        
        if len(self.velocity_buffer) >= 3:
            smoothed = np.median(self.velocity_buffer)
        else:
            smoothed = fused
            
        self.last_valid_velocity = smoothed
        
        return smoothed
    
    def reset(self):
        """Reset fusion state"""
        self.velocity_buffer.clear()
        self.last_valid_velocity = 0.0
