"""
SAVE-3D Application

Main application class - top-level wrapper.
Assembles views from views/ module and connects them together.

Structure:
    - Initialization (__init__): Create all modules
    - UI assembly (_setup_ui): Call view controllers' setup methods
    - Timer setup (_setup_timers)
    - Signal connections (_connect_callbacks)
    - Qt lifecycle (keyPressEvent, closeEvent)
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from qtpy import QtWidgets, QtCore
import time

from .state import AppState
from .data import DataLoader
from .views import NapariViewController, SkeletonViewController, MorphologyViewController
from .controls import _sync_cameras
from .modes import ImageHost, SkeletonHost
from .core import CCPathTracker, MeshBuilder


class SAVE3DViewer(QtWidgets.QWidget):
    """
    SAVE-3D: 3-panel viewer
    
    Panels:
        - Left: Napari 2D slice viewer + control panel
        - Top-right: 3D skeleton view
        - Bottom-right: 3D morphology view
    """
    
    def __init__(self, zarr_path: Path, parent=None):
        """Initialize application"""
        super().__init__(parent)
        self.zarr_path = Path(zarr_path)
        
        # === State ===
        self.state = AppState()
        
        # === Data ===
        self.data = DataLoader().load_all(zarr_path)
        
        # === Core modules ===
        self.cc_tracker = CCPathTracker(
            self.data.cc_metadata,
            self.data.outer_masks_L2,
            self.data.inner_masks_L2,
            self.data.label_names,
            self.data.voxel_size_L2
        )
        self.mesh_builder = MeshBuilder(self.data.voxel_size_L2)
        
        # === View controllers ===
        self.napari_view = NapariViewController(self)
        self.skeleton_view = SkeletonViewController(self)
        self.morphology_view = MorphologyViewController(self)
        
        # === Host modes ===
        self.image_host = ImageHost(self)
        self.skeleton_host = SkeletonHost(self)
        
        # === Timers ===
        self._slice_update_timer = None
        self._camera_sync_timer = None
        self._last_slice_time = time.time()
        self._slice_change_count = 0
        self._adaptive_delay = 50
        
        # === Thread pool for async operations ===
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Note: active_3d_view is stored in self.state, set by _set_active_view in controls.py
        
        # === UI Setup ===
        self._setup_ui()
        self._setup_timers()
        self._connect_callbacks()
        
        # === Initialize views ===
        self.skeleton_view._initialize_skeleton_view()
        
        print("[OK] SAVE-3D Viewer initialized")
    
    # =========================================================================
    # UI SETUP
    # =========================================================================
    
    def _setup_ui(self):
        """Setup 3-panel UI"""
        self.setWindowTitle('SAVE-3D: 3-Panel 3D Viewer')
        
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
        """Setup right panels (skeleton + morphology)"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        v_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        v_splitter.setChildrenCollapsible(False)
        
        # Skeleton panel (from skeleton_view._setup_skeleton_panel)
        skeleton_panel = self.skeleton_view._setup_skeleton_panel()
        v_splitter.addWidget(skeleton_panel)
        
        # Morphology panel (from morphology_view._setup_morphology_panel)
        morphology_panel = self.morphology_view._setup_morphology_panel()
        v_splitter.addWidget(morphology_panel)
        
        v_splitter.setSizes([500, 500])
        
        layout.addWidget(v_splitter)
        
        return container
    
    # =========================================================================
    # TIMERS
    # =========================================================================
    
    def _setup_timers(self):
        """Setup application timers"""
        # Slice update timer (for adaptive throttling)
        self._slice_update_timer = QtCore.QTimer()
        self._slice_update_timer.setSingleShot(True)
        self._slice_update_timer.timeout.connect(self.image_host._delayed_slice_update)
        
        # Camera sync timer
        self._camera_sync_timer = QtCore.QTimer()
        self._camera_sync_timer.timeout.connect(lambda: _sync_cameras(self))
        
        print("[OK] Timers configured")
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def _connect_callbacks(self):
        """Connect signal callbacks"""
        print("\n=== Connecting Callbacks ===")
        
        # Napari label layer callbacks
        labels_layer = self.napari_view.lab_layer
        
        # Enable pan for normal use
        labels_layer.mouse_pan = True
        labels_layer.mouse_zoom = True
        
        # Double-click for CC selection
        if hasattr(labels_layer, 'mouse_double_click_callbacks'):
            labels_layer.mouse_double_click_callbacks.append(self.image_host._on_click_label)
            print("✓ Added double-click callback for CC selection")
        else:
            # Fallback: use drag callback
            labels_layer.mouse_pan = False
            labels_layer.mouse_drag_callbacks.append(self.image_host._on_click_label)
            print("✓ Added drag callback (pan disabled)")
        
        # Slice change
        viewer = self.napari_view.viewer
        viewer.dims.events.current_step.connect(self.image_host._on_slice_changed)
        
        print("✓ Connected slice change callback")
        print("[OK] Callbacks connected")
        print("  • Pan: Single-click + drag")
        print("  • Select CC: Double-click on label")
        print("  • Zoom: Scroll wheel\n")
    
    # =========================================================================
    # KEYBOARD / EVENTS
    # =========================================================================
    
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
            if self.state.active_3d_view == 'skeleton':
                plotter = self.skeleton_view.plotter
            else:
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
        self.executor.shutdown(wait=False)
        super().closeEvent(event)
