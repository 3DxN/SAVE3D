"""
Centralized State Management - Baseline
"""

class AppState:
    def __init__(self):
        # === View State ===
        self.active_3d_view = None