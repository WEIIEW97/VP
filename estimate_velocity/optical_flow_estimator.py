import cv2
import numpy as np


class OpticalFlowEstimator:
    """Estimate velocity from consecutive image frames using optical flow"""
    
    def __init__(self, camera_intrinsics=None, camera_height=1.5, velocity_scale=1.0, 
                 auto_calibrate=True):
        """
        Args:
            camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
            camera_height: camera height above ground in meters
            velocity_scale: initial calibration scale factor (will be auto-adjusted if auto_calibrate=True)
            auto_calibrate: enable online calibration using IMU or fusion feedback
        """
        self.intrinsics = camera_intrinsics
        self.camera_height = camera_height
        self.velocity_scale = velocity_scale
        self.auto_calibrate = auto_calibrate
        self.prev_gray = None
        self.prev_flow_magnitude = None
        
        # Online calibration state
        self.calibration_samples = []  # (optical_raw, reference_velocity) pairs
        self.max_calibration_samples = 30  # Reduced for faster adaptation
        self.calibration_update_interval = 5  # Update scale every 5 samples (was 10)
        self.sample_count = 0
        self.is_calibrated = False
        
        # Adaptive LK params - will be adjusted based on detected flow
        self.lk_params_low_speed = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        
        self.lk_params_high_speed = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.lk_params = self.lk_params_low_speed.copy()
    
    def update_calibration(self, optical_raw_velocity, reference_velocity, debug=False):
        """
        Update velocity scale based on reference velocity (e.g., from IMU)
        
        Args:
            optical_raw_velocity: raw optical flow velocity (before scaling)
            reference_velocity: reference velocity in km/h (e.g., from IMU)
            debug: print debug info
        """
        if not self.auto_calibrate or reference_velocity is None or optical_raw_velocity is None:
            return
        
        # More lenient thresholds for collecting calibration samples
        if abs(optical_raw_velocity) < 0.05 or abs(reference_velocity) < 0.2:
            return  # Too slow, not reliable for calibration
        
        # Debug: print sample collection
        if len(self.calibration_samples) < 5 or self.sample_count % 20 == 0:
            print(f"  [Calib Sample] optical_raw={optical_raw_velocity:.2f}, "
                  f"imu={reference_velocity:.2f} km/h, "
                  f"ratio={reference_velocity/optical_raw_velocity:.3f}")
        
        # Add sample
        self.calibration_samples.append((optical_raw_velocity, reference_velocity))
        self.sample_count += 1
        
        # Keep only recent samples
        if len(self.calibration_samples) > self.max_calibration_samples:
            self.calibration_samples.pop(0)
        
        # Update scale periodically (require fewer samples initially)
        min_samples = 5 if not self.is_calibrated else 10
        if self.sample_count % self.calibration_update_interval == 0 and len(self.calibration_samples) >= min_samples:
            # Calculate scale from recent samples
            opticals = np.array([s[0] for s in self.calibration_samples])
            references = np.array([s[1] for s in self.calibration_samples])
            
            # Use robust estimation (median of ratios)
            ratios = references / opticals
            # Filter outliers (remove top and bottom 20%)
            ratios_sorted = np.sort(ratios)
            n = len(ratios_sorted)
            start_idx = int(n * 0.2)
            end_idx = int(n * 0.8)
            if end_idx > start_idx:
                filtered_ratios = ratios_sorted[start_idx:end_idx]
                new_scale = np.median(filtered_ratios)
                
                # Smooth update to avoid jumps
                # First calibration: use more of new scale (alpha=0.9)
                # Later updates: blend more conservatively (alpha=0.4)
                alpha = 0.4 if self.is_calibrated else 0.9
                old_scale = self.velocity_scale
                self.velocity_scale = alpha * new_scale + (1 - alpha) * self.velocity_scale
                self.is_calibrated = True
                
                if debug or True:  # Always print calibration updates
                    print(f"  [Calibration] Scale: {old_scale:.3f} -> {self.velocity_scale:.3f} "
                          f"(calculated: {new_scale:.3f}, samples: {len(self.calibration_samples)}, "
                          f"alpha: {alpha:.1f})")
        
    def estimate_from_frames(self, frame1, frame2, dt, debug=False, return_visualization=False):
        """
        Estimate velocity from two consecutive frames
        
        Args:
            frame1: first image (BGR or grayscale)
            frame2: second image (BGR or grayscale)
            dt: time interval between frames in seconds
            debug: print debug information
            return_visualization: return visualization data (feature points, flow vectors)
            
        Returns:
            velocity: estimated velocity in km/h, or None if estimation fails
            If return_visualization=True, returns tuple: (velocity, vis_data)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        h, w = gray1.shape
        
        if debug:
            print(f"  [OpticalFlow] Image size: {w}x{h}, dt={dt:.4f}s")
        
        # Adaptive LK params based on previous flow magnitude
        if self.prev_flow_magnitude is not None:
            if self.prev_flow_magnitude > 2.0:  # High speed: >2 pixels/frame
                self.lk_params = self.lk_params_high_speed.copy()
                if debug:
                    print(f"  [OpticalFlow] Using HIGH speed params (prev flow: {self.prev_flow_magnitude:.2f}px)")
            else:
                self.lk_params = self.lk_params_low_speed.copy()
                if debug:
                    print(f"  [OpticalFlow] Using LOW speed params (prev flow: {self.prev_flow_magnitude:.2f}px)")
        
        # Detect features in bottom portion of image (ground plane focus)
        # Bottom 35% is most likely to be ground, avoiding sky/horizon/distant objects
        mask = np.zeros_like(gray1)
        ground_start_row = int(h * 0.65)  # Start at 65% down from top
        mask[ground_start_row:, :] = 255  # Bottom 35% = ground region
        
        feature_params = dict(
            maxCorners=150,  # Detect more features for better ground coverage
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7,
            mask=mask  # Only detect in ground region
        )
        
        p0 = cv2.goodFeaturesToTrack(gray1, **feature_params)
        
        if p0 is None or len(p0) < 10:
            if debug:
                print(f"  [OpticalFlow] Not enough features: {len(p0) if p0 is not None else 0}")
            return None
            
        if debug:
            print(f"  [OpticalFlow] Detected {len(p0)} features")
            
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **self.lk_params)
        
        if p1 is None:
            if debug:
                print(f"  [OpticalFlow] Optical flow computation failed")
            return None
            
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) < 10:
            if debug:
                print(f"  [OpticalFlow] Not enough tracked features: {len(good_new)}")
            return None
            
        if debug:
            print(f"  [OpticalFlow] Tracked {len(good_new)} features")
        
        # Calculate flow vectors
        flow_vectors = good_new - good_old
        
        # Focus on vertical flow (forward motion) in bottom region
        # Use stricter threshold (bottom 30%) to ensure ground plane features
        bottom_threshold = h * 0.70  # Features must be below 70% (i.e., in bottom 30%)
        bottom_mask = good_old[:, 1] > bottom_threshold
        
        if np.sum(bottom_mask) < 5:
            # Not enough features in bottom region
            if debug:
                print(f"  [OpticalFlow] Not enough features in ground region: {np.sum(bottom_mask)}")
            return None
        
        # Get features and their flows in bottom region
        ground_features = good_old[bottom_mask]
        flow_y = flow_vectors[bottom_mask, 1]
        
        # Calculate per-feature distance and velocity
        # Get camera parameters
        if self.intrinsics and 'fy' in self.intrinsics:
            fy = self.intrinsics['fy']
            cy = self.intrinsics.get('cy', h / 2.0)
        else:
            fy = h  # Approximate
            cy = h / 2.0
        
        # Calculate distance to each ground feature
        feature_distances = []
        feature_velocities = []
        valid_flows = []
        
        for i, (y_pixel, pixel_flow) in enumerate(zip(ground_features[:, 1], flow_y)):
            distance = self._calculate_ground_distance(y_pixel, h, fy, cy)
            if distance is not None and abs(pixel_flow) > 0.001:  # Valid distance and flow
                # v = pixel_flow_rate * distance / fy
                pixel_flow_rate = abs(pixel_flow / dt)
                velocity_mps = pixel_flow_rate * distance / fy
                
                feature_distances.append(distance)
                feature_velocities.append(velocity_mps)
                valid_flows.append(pixel_flow)
        
        if len(feature_velocities) < 5:
            if debug:
                print(f"  [OpticalFlow] Not enough valid ground features: {len(feature_velocities)}")
            return None if not return_visualization else (None, None)
        
        # Use median velocity (robust to outliers)
        velocity_mps = np.median(feature_velocities)
        mean_distance = np.mean(feature_distances)
        median_flow = np.median(valid_flows)
        mean_flow = np.mean(valid_flows)
        
        if debug:
            print(f"  [OpticalFlow] Ground features: {len(feature_velocities)}, "
                  f"distances: {np.min(feature_distances):.1f}-{np.max(feature_distances):.1f}m (mean:{mean_distance:.1f}m)")
            print(f"  [OpticalFlow] Median flow: {median_flow:.4f} px, "
                  f"mean: {mean_flow:.4f} px, "
                  f"std: {np.std(valid_flows):.4f} px, "
                  f"velocities: {np.min(feature_velocities):.2f}-{np.max(feature_velocities):.2f} m/s")
        
        # Store flow magnitude for adaptive params
        self.prev_flow_magnitude = abs(mean_flow)
        
        # Check if velocity is reasonable
        if velocity_mps < 0.01:  # < 0.036 km/h
            if debug:
                print(f"  [OpticalFlow] Velocity too small: {velocity_mps:.4f} m/s")
            return None if not return_visualization else (None, None)
        
        # Convert to km/h
        velocity_kmh_raw = velocity_mps * 3.6  # Raw velocity before scaling
        velocity_kmh = velocity_kmh_raw * self.velocity_scale
        
        if debug:
            calib_status = "calibrated" if self.is_calibrated else "initial"
            print(f"  [OpticalFlow] Final velocity: {velocity_kmh:.2f} km/h "
                  f"(raw={velocity_kmh_raw:.2f}, scale={self.velocity_scale:.3f} [{calib_status}])")
        
        result = abs(velocity_kmh)
        
        if return_visualization:
            vis_data = {
                'p0': good_old,
                'p1': good_new,
                'flow_vectors': flow_vectors[bottom_mask],
                'all_flow_vectors': flow_vectors,
                'bottom_mask': bottom_mask,
                'flow': mean_flow,
                'raw_velocity': abs(velocity_kmh_raw),
                'feature_distances': feature_distances
            }
            return result, vis_data, abs(velocity_kmh_raw)
        
        return result, abs(velocity_kmh_raw)
    
    def _calculate_ground_distance(self, y_pixel, image_height, fy, cy, pitch_deg=None):
        """
        Calculate real-world distance to a ground plane point for forward-facing camera
        
        For a forward-facing camera with pitch angle:
        - Distance depends on: camera height, pitch angle, and pixel row
        - Formula: Z = h / tan(pitch + atan((y - cy) / fy))
        
        Since pitch is often unknown, we use an empirical mapping based on image position:
        - Bottom of image (y = height): closer distance (~3-5m)
        - Middle-bottom (y = 0.8 * height): medium distance (~8-15m)
        - Top of visible ground (y = 0.65 * height): far distance (~20-40m)
        """
        if self.intrinsics and 'cy' in self.intrinsics:
            cy_actual = self.intrinsics['cy']
        else:
            cy_actual = cy if cy is not None else image_height / 2.0
        
        # Normalized position in image (0 = top, 1 = bottom)
        y_normalized = y_pixel / image_height
        
        # Empirical distance mapping for forward-facing dashcam
        # Based on typical automotive camera setup (height ~1.2m, pitch ~5-10°)
        # This creates a non-linear mapping from image row to ground distance
        
        if y_normalized < 0.65:  # Above ground region
            return None
        elif y_normalized < 0.75:  # Far ground (near horizon)
            # Far region: 20-40m
            distance = 20.0 + (y_normalized - 0.65) / 0.10 * 20.0
        elif y_normalized < 0.85:  # Mid ground
            # Mid region: 8-20m  
            distance = 8.0 + (y_normalized - 0.75) / 0.10 * 12.0
        else:  # Near ground
            # Near region: 3-8m
            distance = 3.0 + (y_normalized - 0.85) / 0.15 * 5.0
        
        # Sanity check
        if distance < 2.0 or distance > 50.0:
            return None
            
        return distance
    
    def _pixel_flow_to_velocity(self, pixel_flow, dt, image_height, debug=False):
        """
        Convert pixel flow to real-world velocity using pinhole camera model
        
        For a point on the ground at distance d from camera:
        - Image plane: y_img (pixels from image center)
        - Real world: Z (distance forward), Y (height)
        - Projection: y_img = fy * Y / Z
        
        For ground plane (Y = -camera_height):
        - y_img = -fy * camera_height / Z
        - Distance: Z = -fy * camera_height / y_img
        
        Velocity relationship:
        - dy_img/dt = -fy * camera_height * (-dZ/dt) / Z^2
        - dZ/dt (forward velocity) = (dy_img/dt) * Z^2 / (fy * camera_height)
        """
        if abs(dt) < 1e-6:
            return None
            
        # Use focal length
        if self.intrinsics and 'fy' in self.intrinsics:
            fy = self.intrinsics['fy']
        else:
            fy = image_height  # Approximate
            
        # Calculate pixel flow rate (pixels per second)
        pixel_flow_rate = abs(pixel_flow / dt)
        
        # Physical model for ground plane velocity estimation:
        # For forward motion at velocity v, a ground point at distance Z moves with angular velocity:
        #   ω = v / Z (rad/s)
        # The pixel flow in image is:
        #   pixel_flow_rate = fy * ω = fy * v / Z
        # Solving for v:
        #   v = pixel_flow_rate * Z / fy
        
        # Use a representative distance for features in the bottom region
        # Features at y = 0.8 * height (bottom 20%) are good for velocity estimation
        representative_y = image_height * 0.8
        cy = self.intrinsics.get('cy', image_height / 2.0) if self.intrinsics else image_height / 2.0
        
        estimated_distance = self._calculate_ground_distance(representative_y, image_height, fy, cy)
        
        if estimated_distance is None:
            # Fallback to empirical value
            estimated_distance = 10.0
        
        # Calculate velocity
        velocity_mps = pixel_flow_rate * estimated_distance / fy
        
        if debug:
            print(f"  [OpticalFlow] pixel_flow={pixel_flow:.4f}px, dt={dt:.4f}s, "
                  f"rate={pixel_flow_rate:.2f}px/s, "
                  f"fy={fy:.1f}, dist={estimated_distance:.1f}m, "
                  f"v={velocity_mps:.3f}m/s ({velocity_mps*3.6:.2f}km/h)")
        
        return velocity_mps
    
    def estimate_continuous(self, frame, timestamp, debug=False, return_visualization=False):
        """
        Estimate velocity continuously from frame stream
        
        Args:
            frame: current image frame
            timestamp: timestamp in seconds
            debug: print debug information
            return_visualization: return visualization data
            
        Returns:
            If return_visualization=False: (velocity_kmh, raw_velocity_kmh)
            If return_visualization=True: (velocity_kmh, vis_data, raw_velocity_kmh)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_timestamp = timestamp
            if debug:
                print(f"  [OpticalFlow] First frame, initializing")
            return (None, None, None) if return_visualization else (None, None)
            
        dt = timestamp - self.prev_timestamp
        
        if dt < 0.01:
            if debug:
                print(f"  [OpticalFlow] dt too small: {dt:.6f}s")
            return (None, None, None) if return_visualization else (None, None)
            
        result = self.estimate_from_frames(self.prev_gray, gray, dt, debug=debug, return_visualization=return_visualization)
        
        self.prev_gray = gray
        self.prev_timestamp = timestamp
        
        return result
