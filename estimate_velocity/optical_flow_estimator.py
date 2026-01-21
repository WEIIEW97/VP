import cv2
import numpy as np


class OpticalFlowEstimator:
    """Estimate velocity from consecutive image frames using optical flow"""
    
    def __init__(self, camera_intrinsics=None, camera_height=1.5):
        """
        Args:
            camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
            camera_height: camera height above ground in meters
        """
        self.intrinsics = camera_intrinsics
        self.camera_height = camera_height
        self.prev_gray = None
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
    def estimate_from_frames(self, frame1, frame2, dt):
        """
        Estimate velocity from two consecutive frames
        
        Args:
            frame1: first image (BGR or grayscale)
            frame2: second image (BGR or grayscale)
            dt: time interval between frames in seconds
            
        Returns:
            velocity: estimated velocity in km/h, or None if estimation fails
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        h, w = gray1.shape
        
        # Detect features in bottom half of image (ground plane)
        mask = np.zeros_like(gray1)
        mask[h//2:, :] = 255
        
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7,
            mask=mask
        )
        
        p0 = cv2.goodFeaturesToTrack(gray1, **feature_params)
        
        if p0 is None or len(p0) < 10:
            return None
            
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **self.lk_params)
        
        if p1 is None:
            return None
            
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) < 10:
            return None
        
        # Calculate flow vectors
        flow_vectors = good_new - good_old
        
        # Focus on vertical flow (forward motion) in bottom region
        bottom_mask = good_old[:, 1] > h * 0.6
        if np.sum(bottom_mask) < 5:
            return None
            
        flow_y = flow_vectors[bottom_mask, 1]
        
        # Median flow in pixel/frame
        median_flow = np.median(flow_y)
        
        # Convert pixel flow to velocity (km/h)
        velocity_mps = self._pixel_flow_to_velocity(median_flow, dt, h)
        
        if velocity_mps is None:
            return None
            
        velocity_kmh = velocity_mps * 3.6
        
        return abs(velocity_kmh)
    
    def _pixel_flow_to_velocity(self, pixel_flow, dt, image_height):
        """
        Convert pixel flow to real-world velocity
        
        Simple approximation: v = (pixel_flow / dt) * (height / focal_length) * scale
        """
        if abs(dt) < 1e-6:
            return None
            
        # Use approximate focal length if not provided
        if self.intrinsics and 'fy' in self.intrinsics:
            fy = self.intrinsics['fy']
        else:
            # Approximate: focal length ~ image_height (for typical camera FOV)
            fy = image_height
            
        # Scale factor based on camera height
        scale_factor = self.camera_height / fy
        
        # velocity in m/s
        velocity_mps = abs(pixel_flow / dt) * scale_factor
        
        # Apply empirical scaling (typical for ground plane at bottom of image)
        velocity_mps *= 1.5
        
        return velocity_mps
    
    def estimate_continuous(self, frame, timestamp):
        """
        Estimate velocity continuously from frame stream
        
        Args:
            frame: current image frame
            timestamp: timestamp in seconds
            
        Returns:
            velocity in km/h or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_timestamp = timestamp
            return None
            
        dt = timestamp - self.prev_timestamp
        
        if dt < 0.01:
            return None
            
        velocity = self.estimate_from_frames(self.prev_gray, gray, dt)
        
        self.prev_gray = gray
        self.prev_timestamp = timestamp
        
        return velocity
