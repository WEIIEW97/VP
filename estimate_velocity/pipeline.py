import numpy as np
import cv2
from .optical_flow_estimator import OpticalFlowEstimator
from .imu_velocity_estimator import IMUVelocityEstimator
from .velocity_fusion import VelocityFusion
from .velocity_classifier import VelocityClassifier, SpeedRange


class VelocityEstimationPipeline:
    """Complete pipeline for velocity estimation and classification"""
    
    def __init__(self, camera_intrinsics=None, camera_height=1.5):
        """
        Args:
            camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
            camera_height: camera height above ground in meters
        """
        self.optical_estimator = OpticalFlowEstimator(camera_intrinsics, camera_height)
        self.imu_estimator = IMUVelocityEstimator()
        self.fusion = VelocityFusion()
        self.classifier = VelocityClassifier()
        
    def process(self, frame, timestamp, imu_accel=None, orientation=None):
        """
        Process single frame with optional IMU data
        
        Args:
            frame: image frame (BGR or grayscale)
            timestamp: timestamp in seconds
            imu_accel: IMU acceleration [ax, ay, az] in m/s^2 (optional)
            orientation: orientation matrix 3x3 (optional)
            
        Returns:
            dict with keys:
                - 'velocity': fused velocity in km/h
                - 'speed_range': SpeedRange enum
                - 'range_string': human-readable speed range
                - 'optical_velocity': velocity from optical flow
                - 'imu_velocity': velocity from IMU
        """
        # Estimate from optical flow
        optical_velocity = self.optical_estimator.estimate_continuous(frame, timestamp)
        
        # Estimate from IMU if available
        imu_velocity = None
        if imu_accel is not None:
            imu_velocity = self.imu_estimator.estimate(imu_accel, timestamp, orientation)
            
        # Fuse estimates
        velocity = self.fusion.adaptive_fuse(optical_velocity, imu_velocity)
        
        # Classify speed range
        speed_range = self.classifier.classify(velocity)
        range_string = self.classifier.get_range_string(speed_range)
        
        return {
            'velocity': velocity,
            'speed_range': speed_range,
            'range_string': range_string,
            'optical_velocity': optical_velocity,
            'imu_velocity': imu_velocity
        }
    
    def process_batch(self, frames, timestamps, imu_data=None, orientations=None):
        """
        Process batch of frames
        
        Args:
            frames: list of image frames
            timestamps: list of timestamps in seconds
            imu_data: list of IMU acceleration vectors (optional)
            orientations: list of orientation matrices (optional)
            
        Returns:
            list of results (same format as process())
        """
        results = []
        
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            imu_accel = imu_data[i] if imu_data is not None else None
            orientation = orientations[i] if orientations is not None else None
            
            result = self.process(frame, ts, imu_accel, orientation)
            results.append(result)
            
        return results
    
    def reset(self):
        """Reset all estimators"""
        self.optical_estimator.prev_gray = None
        self.imu_estimator.reset()
        self.fusion.reset()
        self.classifier.reset()
