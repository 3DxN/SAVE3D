"""
SAVE-3D Baseline
"""

from pathlib import Path
import numpy as np
from qtpy import QtWidgets, QtCore

from .state import AppState
from .data import DataLoader
from .views import NapariViewController, MorphologyViewController


class BaselineViewer(QtWidgets.QWidget):
    
    def __init__(self, zarr_path: Path, parent=None):
        """Initialize application"""
        super().__init__(parent)
        self.zarr_path = Path(zarr_path)
        
        # === State ===
        self.state = AppState()
        
        # === Data ===
        self.data = DataLoader().load_all(zarr_path)
        
        # === View controllers ===
        self.napari_view = NapariViewController(self)
        self.morphology_view = MorphologyViewController(self)
        
        # Note: active_3d_view is stored in self.state, set by _set_active_view in controls.py
        
        # === UI Setup ===
        self._setup_ui()
        self._connect_callbacks()
        
        print("[OK] SAVE-3D Baseline Viewer initialized")
    
    # =========================================================================
    # UI SETUP
    # =========================================================================
    
    def _setup_ui(self):
        """Setup SAVE-3D Baseline UI"""
        self.setWindowTitle('SAVE-3D Baseline')
        
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            width = int(screen_geometry.width() * 0.9)
            height = int(screen_geometry.height() * 0.9)
            self.resize(width, height)
        else:
            self.resize(1800, 1000)
        
        self.setMinimumSize(1400, 800)
        
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        
        # Left panel: Napari + control panel (from napari_view._setup_napari_panel)
        left_panel = self.napari_view._setup_napari_panel()
        self.main_splitter.addWidget(left_panel)
        
        # Right panel: Skeleton + Morphology
        right_panel = self._setup_right_panels()
        self.main_splitter.addWidget(right_panel)
        
        total_width = self.width()
        self.main_splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
        
        main_layout.addWidget(self.main_splitter)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        print("[UI] 3-panel layout initialized")

    def _setup_right_panels(self):
        morphology_panel = self.morphology_view._setup_morphology_panel()
        return morphology_panel
        
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def _connect_callbacks(self):
        viewer = self.napari_view.viewer
        viewer.dims.events.current_step.connect(self._on_slice_changed)
        print("[OK] Callbacks connected")
    
    # =========================================================================
    # KEYBOARD / EVENTS
    # =========================================================================
    
    def _on_slice_changed(self, event):
        z, y, x = self.napari_view.viewer.dims.current_step
        # Only update 3D crosshair if not dragging (release triggers explicit update)
        if not getattr(self.napari_view, '_dragging', False):
            self._update_crosshair(z, y, x)

    def _update_crosshair(self, z, y, x):
        self.morphology_view.update_crosshair(z, y, x)

    def keyPressEvent(self, event):
        """Handle keyboard events for 3D view panning"""
        pan_step = 50.0  # μm per key press
        
        key = event.key()
        
        if key in [QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            if self.state.active_3d_view is None:
                print("[PAN] Click on a 3D view first to select it")
                event.accept()
                return
            
            # Select plotter based on active view
            plotter = self.morphology_view.plotter
            
            try:
                cam = plotter.camera
                
                # Calculate screen-space right and up vectors
                pos = np.array(cam.position)
                focal = np.array(cam.focal_point)
                up = np.array(cam.up)
                
                # View direction (from camera to focal point)
                view_dir = focal - pos
                view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)
                
                # Right vector = view_dir × up
                right = np.cross(view_dir, up)
                right = right / (np.linalg.norm(right) + 1e-10)
                
                # Recalculate up to ensure orthogonality
                up_corrected = np.cross(right, view_dir)
                up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-10)
                
                # Determine movement direction
                if key == QtCore.Qt.Key_Left:
                    delta = right * pan_step
                elif key == QtCore.Qt.Key_Right:
                    delta = -right * pan_step
                elif key == QtCore.Qt.Key_Up:
                    delta = -up_corrected * pan_step
                elif key == QtCore.Qt.Key_Down:
                    delta = up_corrected * pan_step
                
                # Move both focal point and camera position (maintain relative position)
                new_focal = focal + delta
                new_pos = pos + delta
                
                cam.focal_point = tuple(new_focal)
                cam.position = tuple(new_pos)
                plotter.render()
                
                print(f"[PAN] {self.state.active_3d_view}: delta={delta}")
                
            except Exception as e:
                print(f"[PAN] Error: {e}")
            
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up on close"""
        super().closeEvent(event)