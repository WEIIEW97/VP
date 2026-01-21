import numpy as np
from enum import Enum


class SpeedRange(Enum):
    """Speed range categories"""
    VERY_LOW = 0   # < 5 km/h
    LOW = 1        # 5-30 km/h  
    MEDIUM = 2     # 30-60 km/h
    HIGH = 3       # > 60 km/h


class VelocityClassifier:
    """Classify velocity into different speed ranges"""
    
    def __init__(self, 
                 threshold_very_low=5.0,
                 threshold_low=30.0,
                 threshold_medium=60.0,
                 hysteresis=2.0):
        """
        Args:
            threshold_very_low: threshold between very_low and low (km/h)
            threshold_low: threshold between low and medium (km/h)
            threshold_medium: threshold between medium and high (km/h)
            hysteresis: hysteresis for preventing rapid class changes (km/h)
        """
        self.threshold_very_low = threshold_very_low
        self.threshold_low = threshold_low
        self.threshold_medium = threshold_medium
        self.hysteresis = hysteresis
        
        self.current_range = SpeedRange.VERY_LOW
        
    def classify(self, velocity):
        """
        Classify velocity into speed range with hysteresis
        
        Args:
            velocity: velocity in km/h
            
        Returns:
            SpeedRange: classified speed range
        """
        v = abs(velocity)
        
        # Apply hysteresis based on current state
        if self.current_range == SpeedRange.VERY_LOW:
            if v > self.threshold_very_low + self.hysteresis:
                if v < self.threshold_low:
                    self.current_range = SpeedRange.LOW
                elif v < self.threshold_medium:
                    self.current_range = SpeedRange.MEDIUM
                else:
                    self.current_range = SpeedRange.HIGH
                    
        elif self.current_range == SpeedRange.LOW:
            if v < self.threshold_very_low - self.hysteresis:
                self.current_range = SpeedRange.VERY_LOW
            elif v > self.threshold_low + self.hysteresis:
                if v < self.threshold_medium:
                    self.current_range = SpeedRange.MEDIUM
                else:
                    self.current_range = SpeedRange.HIGH
                    
        elif self.current_range == SpeedRange.MEDIUM:
            if v < self.threshold_low - self.hysteresis:
                if v < self.threshold_very_low:
                    self.current_range = SpeedRange.VERY_LOW
                else:
                    self.current_range = SpeedRange.LOW
            elif v > self.threshold_medium + self.hysteresis:
                self.current_range = SpeedRange.HIGH
                
        elif self.current_range == SpeedRange.HIGH:
            if v < self.threshold_medium - self.hysteresis:
                if v < self.threshold_very_low:
                    self.current_range = SpeedRange.VERY_LOW
                elif v < self.threshold_low:
                    self.current_range = SpeedRange.LOW
                else:
                    self.current_range = SpeedRange.MEDIUM
                    
        return self.current_range
    
    def classify_simple(self, velocity):
        """
        Simple classification without hysteresis
        
        Args:
            velocity: velocity in km/h
            
        Returns:
            SpeedRange: classified speed range
        """
        v = abs(velocity)
        
        if v < self.threshold_very_low:
            return SpeedRange.VERY_LOW
        elif v < self.threshold_low:
            return SpeedRange.LOW
        elif v < self.threshold_medium:
            return SpeedRange.MEDIUM
        else:
            return SpeedRange.HIGH
    
    def get_range_string(self, speed_range):
        """
        Get human-readable string for speed range
        
        Args:
            speed_range: SpeedRange enum value
            
        Returns:
            str: speed range description
        """
        if speed_range == SpeedRange.VERY_LOW:
            return f"< {self.threshold_very_low} km/h"
        elif speed_range == SpeedRange.LOW:
            return f"{self.threshold_very_low}-{self.threshold_low} km/h"
        elif speed_range == SpeedRange.MEDIUM:
            return f"{self.threshold_low}-{self.threshold_medium} km/h"
        else:
            return f"> {self.threshold_medium} km/h"
    
    def reset(self):
        """Reset classifier state"""
        self.current_range = SpeedRange.VERY_LOW
