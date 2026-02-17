"""
Centralized State Management
"""

import numpy as np


class AppState:
    """
    Centralized application state
    """
    
    def __init__(self):
        # === Host Mode ===
        self.host_mode = 'image'                 # 'image' | 'skeleton'
        
        # Note: morph_mode is stored in each mode class (ImageHost, SkeletonHost)
        # Note: selection-related state is stored in SkeletonHost class
        
        # === Tracked State ===
        self.tracked_state = {
            'label_id': None,
            'label_name': None,
            'z_L0': None,
            'z_L2': None,
            'cc_id': None,
        }
        
        # === Sphere ===
        self.sphere_position = None
        
        # === Cache ===
        self._current_cc_mask_L0 = None
        self._current_cc_z_L0 = None
        self._last_plane_center = None
        self._last_plane_size = None
        
        # === CC Path ===
        self.cc_path_3d = None
        
        # === View State ===
        self.active_3d_view = None
        self.auto_center_enabled = False

        # === Camera Sync ===
        self.camera_sync_enabled = False

        # === Skeleton Filter ===
        self.filtered_spatial_cc_ids = set()
