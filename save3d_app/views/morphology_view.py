"""
Morphology 3D View 
NOTE: Only setup function here. All operations are in modes/*.
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtWidgets, QtCore

from ..controls import _set_active_view

class MorphologyViewController:
    """
    Morphology 3D view 
    """
    
    def __init__(self, app):
        """
        Initialize with app reference
        """
        self.app = app
        
        # PyVista components (set during setup)
        self.plotter = None
        
        # Mesh actors - Image Host
        self.outer_volume_actors = {}
        self.inner_mesh_actors = {}
        self.label_mode_actors = {}
        
        # Mesh actors - Skeleton Host
        self.global_morph_actors = {}
        
        # Guide plane actors - Image Host
        self.plane_actor = None
        self.contour_actor = None
        self.scale_line_actor = None
        self.scale_text_actor = None
        
        # Guide plane actors - Skeleton Host
        self.skeleton_host_plane_actor = None
        self.skeleton_host_contour_actor = None
        self.skeleton_host_scale_actor = None
        self.skeleton_host_scale_text_actor = None
        
        # Info label (set during setup)
        self.morphology_info = None

        # Global morphology meshes (for Skeleton Host Navigation mode)
        self.global_morph_meshes = {}
        self.morph_meshes_loaded = False
    
    def _setup_morphology_panel(self):
        """Setup Morphology view"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        
        info = QtWidgets.QGroupBox("Morphology View @ L2")
        info_layout = QtWidgets.QVBoxLayout(info)
        
        self.morphology_info = QtWidgets.QLabel("Outer/Inner volumes")
        self.morphology_info.setWordWrap(True)
        info_layout.addWidget(self.morphology_info)
        
        layout.addWidget(info)
        
        self.plotter = QtInteractor(self.app)
        self.plotter.enable_trackball_style()
        self.plotter.set_background([0.95, 0.95, 0.95])
        
        try:
            self.plotter.enable_anti_aliasing()
            print("  ✓ Morphology: Anti-aliasing enabled")
        except Exception as e:
            print(f"  ⚠ Morphology: Anti-aliasing not available: {e}")
        
        # Enable shadows
        try:
            self.plotter.enable_shadows = True
            print("  ✓ Morphology: Shadows enabled")
        except Exception as e:
            print(f"  ⚠ Morphology: Shadows not available: {e}")
        
        # Add three-light setup
        try:
            z_flip = 1

            # Light 1: main light from upper right front
            light1 = pv.Light(
                position=(10, 10, 10 * z_flip),
                focal_point=(0, 0, 0),
                color='white'
            )
            
            # Light 2: fill light from lower left front
            light2 = pv.Light(
                position=(-10, -10, 10 * z_flip),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.6
            )
            
            # Light 3: back light from upper front
            light3 = pv.Light(
                position=(0, 10, -10 * z_flip),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.4
            )
            
            self.plotter.add_light(light1)
            self.plotter.add_light(light2)
            self.plotter.add_light(light3)
            print("  ✓ Morphology: Three-light setup added")
        except Exception as e:
            print(f"  ⚠ Morphology: Custom lighting not available: {e}")
        
        try:
            self.plotter.add_axes(color='black', line_width=2)
        except:
            pass
        
        self.plotter.add_text(
            'Morphology View',
            position='upper_left',
            font_size=10,
            color='black'
        )

        # Camera in front, looking into screen
        self.plotter.camera.position = (0, 0, -1000)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, -1, 0)
        
        layout.addWidget(self.plotter.interactor, stretch=1)
        
        # Set focus on click
        self.plotter.interactor.mousePressEvent = lambda e: _set_active_view(self.app, 'morphology', e)
        
        # Disable VTK default keyboard events (avoid arrow keys being overridden by zoom/rotate)
        try:
            self.plotter.iren.remove_observers("KeyPressEvent")
            self.plotter.iren.remove_observers("CharEvent")
        except:
            pass
        
        # Reset actor dicts
        self.outer_volume_actors = {}
        self.inner_mesh_actors = {}
        self.plane_actor = None
        self.contour_actor = None
        self.scale_line_actor = None
        self.scale_text_actor = None
        
        # Opacity depends on whether inner mask exists
        if self.app.data.has_inner_mask:
            self.outer_opacity = 0.2  # Transparent outer when inner exists
            self.inner_opacity = 0.8  # Solid inner
        else:
            self.outer_opacity = 0.8  # Solid outer when no inner
            self.inner_opacity = 0.0  # No inner
        
        # XY crop settings for morphology view
        self.crop_buffer_um = 50.0  # Buffer around CC center
        self.crop_y_min = None
        self.crop_y_max = None
        self.crop_x_min = None
        self.crop_x_max = None
        
        return container
    
    def _load_global_morphology_meshes(self):
        """
        Lazy load prebuilt morphology meshes for Skeleton Host Navigation mode.
        Delegates to DataLoader and caches references locally.
        """
        if self.morph_meshes_loaded:
            return
        
        # Load via DataLoader
        self.app.data._load_global_morphology_meshes()
        
        # Cache references locally
        self.global_morph_meshes = self.app.data.global_morph_meshes
        self.morph_meshes_loaded = self.app.data.morph_meshes_loaded
