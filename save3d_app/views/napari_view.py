"""
Napari 2D View
NOTE: Only setup function here. All operations are in modes/*.
"""

import numpy as np
import napari
from qtpy import QtWidgets, QtCore

from ..utils import _hex_to_rgba
from .control_panel import _create_control_panel


class NapariViewController:
    """
    2D Napari view 
    """
    
    def __init__(self, app):
        """
        Initialize with app reference
        
        Args:
            app: main SkeletonTracker instance
        """
        self.app = app
        
        # Napari components (set during setup)
        self.viewer = None
        self.img_layer = None
        self.lab_layer = None
        self.boundary_layer = None
    
    def _setup_napari_panel(self):
        """Setup Napari 2D viewer"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        
        self.viewer = napari.Viewer(show=False)
        napari_window = self.viewer.window._qt_window

        try:
            self.viewer.window._qt_window.dockLayerControls.setVisible(False)
            self.viewer.window._qt_window.dockLayerList.setVisible(False)
        except AttributeError:
            try:
                for dock in self.viewer.window._qt_window.findChildren(QtWidgets.QDockWidget):
                    dock_name = dock.objectName().lower()
                    if 'layer' in dock_name or 'control' in dock_name:
                        dock.setVisible(False)
                        print(f"  Hidden dock: {dock.objectName()}")
            except Exception as e:
                print(f"  ⚠ Could not hide layer controls: {e}")
        
        self.viewer.axes.visible = True
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'μm'
        
        # Get data references
        data = self.app.data
        
        # Add histology
        is_rgb = len(data.img_full.shape) == 4
        if len(data.img_levels) > 1:
            self.img_layer = self.viewer.add_image(
                data.img_levels,
                name='Histology',
                rgb=is_rgb,
                opacity=0.8,
                multiscale=True,
                scale=data.voxel_size_L0,
                cache=True
            )
        else:
            self.img_layer = self.viewer.add_image(
                data.img_levels[0],
                name='Histology',
                rgb=is_rgb,
                opacity=0.8,
                scale=data.voxel_size_L0,
                cache=True
            )
        
        # Add boundary layer FIRST (so it's below labels)
        self.boundary_layer = self.viewer.add_shapes(
            name='CC Boundary',
            ndim=3,
            edge_color='black',
            edge_width=5,
            face_color='transparent',
            scale=data.voxel_size_L0
        )
        
        # BUILD COLORMAP BEFORE ADDING LABELS
        from napari.utils.colormaps import DirectLabelColormap
        
        color_dict = {}
        color_dict[None] = np.array([0, 0, 0, 0])  # Background
        
        for label_id, label_name in enumerate(data.label_names, start=1):
            hex_color = data.label_colors[label_name]
            rgba = np.array(_hex_to_rgba(hex_color))
            color_dict[label_id] = rgba
        
        print(f"\n[COLORS] Building colormap:")
        for label_id, label_name in enumerate(data.label_names, start=1):
            rgba = color_dict[label_id]
            print(f"  {label_id}: {label_name} = {data.label_colors[label_name]}")
        
        custom_colormap = DirectLabelColormap(color_dict=color_dict)
        
        # Add labels LAST (so it's on top and clickable) WITH COLORMAP
        if len(data.lab_levels) > 1:
            self.lab_layer = self.viewer.add_labels(
                data.lab_levels,
                name='Labels',
                opacity=0.5,
                colormap=custom_colormap,
                multiscale=True,
                scale=data.voxel_size_L0,
                cache=True
            )
        else:
            self.lab_layer = self.viewer.add_labels(
                data.lab_levels[0],
                name='Labels',
                opacity=0.5,
                scale=data.voxel_size_L0,
                colormap=custom_colormap,
                cache=True
            )
        
        print(f"[COLORS] ✓ Added labels with custom colormap")
        
        # CRITICAL: Ensure labels layer is on top and interactive
        self.viewer.layers.selection.active = self.lab_layer
        
        print(f"[LAYER ORDER] From bottom to top:")
        for i, layer in enumerate(self.viewer.layers):
            print(f"  {i}: {layer.name}")
        
        # Create control panel (from controls.py)
        controls = _create_control_panel(self.app)
        
        layout.addWidget(napari_window, stretch=1)
        layout.addWidget(controls, stretch=0)
        
        return container