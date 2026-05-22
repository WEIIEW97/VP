import numpy as np
import cv2
from optical_flow_estimator import OpticalFlowEstimator
from imu_velocity_estimator import IMUVelocityEstimator
from velocity_fusion import VelocityFusion
from velocity_classifier import VelocityClassifier, SpeedRange


class VelocityEstimationPipeline:
    """Complete pipeline for velocity estimation and classification"""
    
    def __init__(self, camera_intrinsics=None, camera_height=1.5, velocity_scale=1.0, 
                 auto_calibrate=True):
        """
        Args:
            camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
            camera_height: camera height above ground in meters
            velocity_scale: initial calibration scale factor
            auto_calibrate: enable online calibration using IMU feedback
        """
        self.optical_estimator = OpticalFlowEstimator(camera_intrinsics, camera_height, 
                                                       velocity_scale, auto_calibrate)
        self.imu_estimator = IMUVelocityEstimator()
        self.fusion = VelocityFusion()
        self.classifier = VelocityClassifier()
        self.auto_calibrate = auto_calibrate
        
    def set_velocity_scale(self, scale):
        """Update velocity calibration scale"""
        self.optical_estimator.velocity_scale = scale
    
    def get_velocity_scale(self):
        """Get current velocity calibration scale"""
        return self.optical_estimator.velocity_scale
    
    def is_calibrated(self):
        """Check if optical flow has been calibrated"""
        return self.optical_estimator.is_calibrated
        
    def process(self, frame, timestamp, imu_accel=None, orientation=None, debug=False, return_visualization=False):
        """
        Process single frame with optional IMU data
        
        Args:
            frame: image frame (BGR or grayscale)
            timestamp: timestamp in seconds
            imu_accel: IMU acceleration [ax, ay, az] in m/s^2 (optional)
            orientation: orientation matrix 3x3 (optional)
            debug: print debug information
            return_visualization: return visualization data for optical flow
            
        Returns:
            dict with keys:
                - 'velocity': fused velocity in km/h
                - 'speed_range': SpeedRange enum
                - 'range_string': human-readable speed range
                - 'optical_velocity': velocity from optical flow
                - 'imu_velocity': velocity from IMU
                - 'vis_data': visualization data (if return_visualization=True)
        """
        # Estimate from optical flow
        optical_result = self.optical_estimator.estimate_continuous(frame, timestamp, debug=debug, return_visualization=return_visualization)
        
        # Parse optical flow result
        if return_visualization:
            if optical_result and len(optical_result) == 3:
                optical_velocity, vis_data, optical_raw = optical_result
            else:
                optical_velocity, vis_data, optical_raw = None, None, None
        else:
            if optical_result and len(optical_result) == 2:
                optical_velocity, optical_raw = optical_result
            else:
                optical_velocity, optical_raw = None, None
            vis_data = None
        
        # Estimate from IMU if available
        imu_velocity = None
        if imu_accel is not None:
            imu_velocity = self.imu_estimator.estimate(imu_accel, timestamp, orientation, debug=debug)
            
            # Use IMU velocity to calibrate optical flow online
            if self.auto_calibrate and imu_velocity is not None and optical_raw is not None:
                self.optical_estimator.update_calibration(optical_raw, imu_velocity, debug=debug)
            
        # Fuse estimates
        velocity = self.fusion.adaptive_fuse(optical_velocity, imu_velocity)
        
        # Classify speed range
        speed_range = self.classifier.classify(velocity)
        range_string = self.classifier.get_range_string(speed_range)
        
        result = {
            'velocity': velocity,
            'speed_range': speed_range,
            'range_string': range_string,
            'optical_velocity': optical_velocity,
            'imu_velocity': imu_velocity
        }
        
        if return_visualization:
            result['vis_data'] = vis_data
            
        return result
    
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
