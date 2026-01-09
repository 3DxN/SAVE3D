"""
Utility Functions

Progress tracking and helper functions for preprocessing pipeline.
"""

import time


class SimpleProgress:
    """
    Simple progress tracker with elapsed time display
    
    Usage:
        progress = SimpleProgress()
        progress.update("Step 1: Loading data...")
        # ... do work ...
        progress.update("Step 2: Processing...")
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.step = 0
    
    def update(self, message):
        """Print progress message with elapsed time"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed/60:.1f}min] {message}")
    
    def elapsed_minutes(self):
        """Return elapsed time in minutes"""
        return (time.time() - self.start_time) / 60.0
    
    def elapsed_str(self):
        """Return formatted elapsed time string"""
        minutes = self.elapsed_minutes()
        if minutes < 1:
            return f"{minutes * 60:.1f}s"
        elif minutes < 60:
            return f"{minutes:.1f}min"
        else:
            hours = minutes / 60
            return f"{hours:.1f}h"
