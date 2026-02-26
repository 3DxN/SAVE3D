"""
Image Host Mode

Handles all operations when Napari 2D view is the primary host.

Morph Modes:
    0: Instance - show single instance prebuilt mesh
    1: Label - show entire label prebuilt mesh  
    2: Z Navigation - minimal updates on z scroll

OPTIMIZATIONS:
    - LRU cache for 2D CC labeling (avoids re-computing on scroll)
    - LRU cache for slice data (avoids re-reading from Zarr)
    - GPU-accelerated CC labeling when available
    - Aggressive throttling during rapid scrolling
"""

import time
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import measure
from qtpy import QtCore, QtWidgets
import pyvista as pv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..controls import _center_on_component, _zoom_to_path_bbox, _clear_morphology_actors
from ..utils import _hex_to_rgb

# GPU-accelerated CC labeling (optional)
_GPU_CC_AVAILABLE = False
_cupy_label = None

try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as cupy_label
    _cupy_label = cupy_label
    _GPU_CC_AVAILABLE = True
except ImportError:
    pass


# Debug mode - set to False for production to reduce print overhead
_DEBUG_SLICE_UPDATE = False


class _SliceDataCache:
    """LRU cache for slice data to avoid repeated Zarr reads"""
    
    def __init__(self, maxsize=16):
        self.maxsize = maxsize
        self.cache = {}  # {(data_type, label_name, z): array}
        self.access_order = []
    
    def get_outer_slice(self, outer_mask, label_name, z):
        """Get cached outer mask slice or read from Zarr"""
        key = ('outer', label_name, z)
        
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Read from Zarr
        data = outer_mask[z].compute()
        
        # Cache
        self.cache[key] = data
        self.access_order.append(key)
        
        # Evict if over capacity
        while len(self.cache) > self.maxsize:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        return data
    
    def get_label_slice(self, lab_full, z):
        """Get cached label slice or read from Zarr"""
        key = ('label', 'full', z)
        
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Read from Zarr
        data = lab_full[z].compute()
        
        # Cache
        self.cache[key] = data
        self.access_order.append(key)
        
        # Evict if over capacity
        while len(self.cache) > self.maxsize:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        return data
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


class _CCLabelCache:
    """LRU-style cache for 2D CC labeling results"""
    
    def __init__(self, maxsize=32):
        self.maxsize = maxsize
        self.cache = {}  # {(label_id, z): (labeled, num_features)}
        self.access_order = []  # Most recently accessed keys
    
    def get(self, label_id, z, mask):
        """Get cached result or compute and cache"""
        key = (label_id, z)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Compute
        if _GPU_CC_AVAILABLE:
            try:
                mask_gpu = cp.asarray(mask)
                labeled_gpu, num_features = _cupy_label(mask_gpu)
                labeled = cp.asnumpy(labeled_gpu)
                del mask_gpu, labeled_gpu
                result = (labeled, int(num_features))
            except Exception:
                result = ndimage.label(mask)
        else:
            result = ndimage.label(mask)
        
        # Cache
        self.cache[key] = result
        self.access_order.append(key)
        
        # Evict if over capacity
        while len(self.cache) > self.maxsize:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        return result
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


# Global cache instances
_cc_label_cache = _CCLabelCache(maxsize=64)
_slice_data_cache = _SliceDataCache(maxsize=32)

class _MeshCache:
    """
    LRU cache for loaded meshes to avoid redundant disk I/O.
    Thread-safe for parallel loading.
    """
    
    def __init__(self, maxsize=200, max_memory_mb=1000):
        self.maxsize = maxsize
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}  # {filepath_str: pv.PolyData}
        self.access_order = []
        self.memory_usage = 0
        self._lock = threading.Lock()
    
    def _estimate_mesh_size(self, mesh):
        """Estimate memory usage of a PyVista mesh in bytes"""
        if mesh is None:
            return 0
        return mesh.n_points * 24 + mesh.n_cells * 32
    
    def get(self, filepath):
        """Get cached mesh or None"""
        with self._lock:
            key = str(filepath)
            if key in self.cache:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, filepath, mesh):
        """Cache a mesh"""
        with self._lock:
            key = str(filepath)
            if key in self.cache:
                return
            
            mesh_size = self._estimate_mesh_size(mesh)
            
            # Evict if over limits
            while (self.memory_usage + mesh_size > self.max_memory_bytes or 
                   len(self.cache) >= self.maxsize) and self.access_order:
                oldest = self.access_order.pop(0)
                if oldest in self.cache:
                    old_mesh = self.cache.pop(oldest)
                    self.memory_usage -= self._estimate_mesh_size(old_mesh)
            
            self.cache[key] = mesh
            self.access_order.append(key)
            self.memory_usage += mesh_size
    
    def clear(self):
        """Clear all cached meshes"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.memory_usage = 0


# Global mesh cache
_mesh_cache = _MeshCache(maxsize=200, max_memory_mb=1000)


def _load_single_mesh(filepath):
    """Load a single mesh, using cache if available."""
    cached = _mesh_cache.get(filepath)
    if cached is not None:
        return cached
    
    try:
        mesh = pv.read(str(filepath))
        _mesh_cache.put(filepath, mesh)
        return mesh
    except Exception as e:
        print(f"  [WARN] Failed to load mesh {filepath}: {e}")
        return None


def _load_meshes_parallel(mesh_paths, max_workers=6):
    """
    Load multiple meshes in parallel.
    
    Args:
        mesh_paths: List of (key, filepath) tuples
        max_workers: Number of parallel threads
    
    Returns:
        Dict of {key: mesh}
    """
    results = {}
    total = len(mesh_paths)
    
    if total == 0:
        return results
    
    # Small number: sequential is fine
    if total <= 3:
        for key, filepath in mesh_paths:
            mesh = _load_single_mesh(filepath)
            if mesh is not None:
                results[key] = mesh
        return results
    
    # Parallel loading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for key, filepath in mesh_paths:
            future = executor.submit(_load_single_mesh, filepath)
            future_to_key[future] = key
        
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                mesh = future.result()
                if mesh is not None:
                    results[key] = mesh
            except Exception as e:
                print(f"  [WARN] Mesh load failed for {key}: {e}")
    
    return results


class ImageHost:
    """
    Image Host mode - 2D Napari drives navigation
    """
    
    def __init__(self, app):
        """
        Initialize with app reference
        """
        self.app = app
        
        # Morph mode: 0=Instance, 1=Label, 2=Z Navigation
        self.morph_mode = 1  # Default to Label mode
        
        # Label mode caching
        self._current_label_mode_label = None
        
        # Slice update pending
        self._pending_slice = None
    
    # =========================================================================
    # ENTRY POINTS - External callbacks
    # =========================================================================
    
    def _switch_to_image_host(self):
        """Switch to Image Host mode"""
        app = self.app
        
        print("\n[HOST MODE] → IMAGE HOST")
        app.state.host_mode = 'image'
        app.skeleton_host.current_skeleton_instance_id = -1
        
        # Show/hide mode widgets
        app.image_morph_mode_widget.setVisible(True)
        app.morph_mode_widget.setVisible(False)
        
        # Hide sphere widget
        if app.skeleton_view.sphere_widget:
            try:
                sphere_center = app.skeleton_view.sphere_widget.GetCenter()
                app.state.sphere_position = np.array(sphere_center)
                print(f"  ✓ Saved sphere position: {sphere_center}")
            except:
                pass
            app.skeleton_view.sphere_widget.Off()
        
        # Update marker_actor to same position
        if app.state.tracked_state['label_id'] is not None:
            self._update_skeleton_marker()
            if app.skeleton_view.black_marker_actor:
                app.skeleton_view.black_marker_actor.visibility = True
            print(f"  ✓ Marker updated")
        
        # Clear selection highlight and state
        app.skeleton_host._clear_selection_highlight()
        app.skeleton_host.selected_skeleton_indices = set()
        app.skeleton_host.selection_drawing = False
        if hasattr(app, 'draw_selection_btn'):
            app.draw_selection_btn.setChecked(False)

        # Clear _selected_kdtree (avoid residue)
        if hasattr(app.skeleton_host, '_selected_kdtree'):
            app.skeleton_host._selected_kdtree = None
        if hasattr(app.skeleton_host, '_selected_list'):
            app.skeleton_host._selected_list = None

        app.skeleton_view.plotter.render()

        # Remove skeleton host plane actors
        for attr in ['skeleton_host_plane_actor', 'skeleton_host_scale_actor', 'skeleton_host_scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)

        # Remove skeleton host contour
        if app.morphology_view.skeleton_host_contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
            except:
                pass
            app.morphology_view.skeleton_host_contour_actor = None

        # Remove global morph actors
        for mesh_name, data in app.morphology_view.global_morph_actors.items():
            try:
                app.morphology_view.plotter.remove_actor(data['actor'])
            except:
                pass
        app.morphology_view.global_morph_actors = {}

        # Adjust splitter ratio
        if hasattr(app, 'main_splitter'):
            total = sum(app.main_splitter.sizes())
            app.main_splitter.setSizes([int(total * 0.6), int(total * 0.4)])

        # Rebuild views if tracking
        if app.state.tracked_state['label_id'] is not None:
            print("  ✓ Rebuilding morphology view for Image Host")
            self._update_2d_view()
            self._update_morphology_view()

        app.morph_mode_widget.setVisible(False)

        # Enable dims widget
        try:
            dims_widget = app.napari_view.viewer.window._qt_viewer.dims
            dims_widget.setEnabled(True)
        except:
            pass

        app.skeleton_view.plotter.render()
        app.morphology_view.plotter.render()

        print("  ✓ Image Host mode active")
        
    def _on_click_label(self, layer, event):
        """Handle label click - VERIFIED WORKING"""
        app = self.app
        
        print(f"\n{'='*80}")
        print(f"[CLICK EVENT]")
        print(f"{'='*80}")

        if app.state.host_mode == 'skeleton':
            return
        
        pos = app.napari_view.viewer.cursor.position
        if pos is None or len(pos) < 3:
            print(f"✗ No cursor position")
            return
        
        # Napari coords @ L0
        pos_data = app.napari_view.lab_layer.world_to_data(pos)
        z_L0, y_L0, x_L0 = (int(round(p)) for p in pos_data[:3])
        print(f"Napari coords @ L0: ({z_L0}, {y_L0}, {x_L0})")
        
        # Bounds check
        if (z_L0 < 0 or y_L0 < 0 or x_L0 < 0 or 
            z_L0 >= app.data.lab_full.shape[0] or 
            y_L0 >= app.data.lab_full.shape[1] or 
            x_L0 >= app.data.lab_full.shape[2]):
            print(f"✗ Out of bounds")
            return
        
        # Read label
        try:
            lid = int(app.data.lab_full[z_L0, y_L0, x_L0].compute())
            print(f"Label ID: {lid}")
            
            if lid == 0:
                print(f"✗ Background")
                return
            
            if lid > len(app.data.label_names):
                print(f"✗ Invalid label ID")
                return
            
            label_name = app.data.label_names[lid - 1]
            print(f"✓ Label name: {label_name}")
            
        except Exception as e:
            print(f"✗ Error reading label: {e}")
            return
        
        # Convert to mask coords @ L2 (relative to original 1x)
        z_L2 = z_L0 // 4
        y_L2 = y_L0 // 4
        x_L2 = x_L0 // 4
        
        print(f"Mask coords @ L2: ({z_L2}, {y_L2}, {x_L2})")
        
        # Find CC
        cc_id = app.cc_tracker._find_clicked_cc(lid, z_L2, y_L2, x_L2)
        
        if cc_id is None:
            print(f"✗ No CC found")
            return
        
        # Update state
        app.state.tracked_state = {
            'label_id': lid,
            'label_name': label_name,
            'z_L0': z_L0,
            'z_L2': z_L2,
            'cc_id': cc_id
        }
        
        print(f"\n✅ SUCCESS!")
        print(f"  Tracking: {label_name} CC#{cc_id} @ z_L0={z_L0}, z_L2={z_L2}")
        print(f"{'='*80}\n")
        
        # Update all views
        self._update_all_views()
        QtCore.QTimer.singleShot(50, self._force_refresh_boundary)
            
    def _on_slice_changed(self, event=None):
        """Handle slice change with aggressive throttling for smooth scrolling"""
        app = self.app

        # Skip if in Skeleton Host mode
        if app.state.host_mode == 'skeleton':
            # Just update info label, don't propagate
            current_slice = int(app.napari_view.viewer.dims.current_step[0])
            total_slices = app.data.lab_full.shape[0]
            z_position_um = current_slice * app.data.voxel_size_L0[0]
            
            app.slice_info.setText(
                f"Slice: {current_slice} / {total_slices}  |  "
                f"Z = {z_position_um:.1f} μm  |  "
                f"(Skeleton is HOST)"
            )
            return
    
        current_slice = int(app.napari_view.viewer.dims.current_step[0])
        total_slices = app.data.lab_full.shape[0]
        z_position_um = current_slice * app.data.voxel_size_L0[0]
        
        if app.state.tracked_state['label_id'] is not None:
            app.slice_info.setText(
                f"Slice: {current_slice} / {total_slices}  |  "
                f"Z = {z_position_um:.1f} μm  |  "
                f"Tracking: {app.state.tracked_state['label_name']}"
            )
        else:
            app.slice_info.setText(
                f"Slice: {current_slice} / {total_slices}  |  "
                f"Z = {z_position_um:.1f} μm  |  "
                f"No CC tracked"
            )
        
        # OPTIMIZED: More aggressive throttling for smoother scrolling
        current_time = time.time()
        time_since_last = (current_time - app._last_slice_time) * 1000
        
        # Detect rapid scrolling (< 50ms between events)
        if time_since_last < 50:
            app._slice_change_count += 2  # Increase faster during rapid scroll
            # During rapid scroll, use longer delay to batch updates
            app._adaptive_delay = min(200, 100 + app._slice_change_count * 10)
        elif time_since_last < 100:
            app._slice_change_count += 1
            app._adaptive_delay = min(150, 50 + app._slice_change_count * 5)
        else:
            # Slow scrolling - reset counter and use shorter delay
            app._slice_change_count = max(0, app._slice_change_count - 3)
            app._adaptive_delay = max(16, 30 - app._slice_change_count * 2)  # 16ms = ~60fps
        
        app._last_slice_time = current_time
        self._pending_slice = current_slice
        app._slice_update_timer.start(app._adaptive_delay)

    def _on_morph_mode_changed_image(self, button):
        """Handle morphology mode change in Image Host"""
        app = self.app
        
        mode_id = app.image_morph_mode_group.id(button)
        mode_names = {0: 'Instance', 1: 'Label', 2: 'Z Navigation'}
        print(f"\n[IMAGE MORPH MODE] Changed to: {mode_names[mode_id]}")
        
        self.morph_mode = mode_id
        
        # Rebuild morphology view with new mode
        if app.state.tracked_state['label_id'] is not None:
            self._update_morphology_view()

    # =========================================================================
    # SLICE HANDLING - Timer & propagation
    # =========================================================================

    def _delayed_slice_update(self):
        """Delayed slice update with tracking propagation"""
        if self._pending_slice is not None:
            self._on_slice_update(self._pending_slice)

    def _on_slice_update(self, z_L0):
        """Update tracking - detect instance boundary crossing"""
        app = self.app
        
        if app.state.tracked_state['label_id'] is None:
            return
        
        z_L2 = z_L0 // 4
        
        # Store old state
        old_z_L0 = app.state.tracked_state['z_L0']
        old_z_L2 = app.state.tracked_state['z_L2']
        old_label_id = app.state.tracked_state['label_id']
        
        # Always update z_L0
        app.state.tracked_state['z_L0'] = z_L0
        
        # Check if same L2 layer
        if z_L2 == old_z_L2:
            # Update 2D view only
            self._update_2d_view()
            return
        
        # Different L2 layer - use path_3d
        if app.state.cc_path_3d is None:
            if _DEBUG_SLICE_UPDATE:
                print(f"[TRACK] No path_3d available")
            return
        
        if z_L2 not in app.state.cc_path_3d:
            # Out of path
            if _DEBUG_SLICE_UPDATE:
                print(f"[TRACK] z={z_L2} not in path")
            app.napari_view.boundary_layer.data = []
            app.state.tracked_state['z_L2'] = z_L2

            # Clear contour when out of path
            if app.show_plane_contour_chk.isChecked() and app.morphology_view.contour_actor:
                try:
                    app.morphology_view.plotter.remove_actor(app.morphology_view.contour_actor)
                except:
                    pass
                app.morphology_view.contour_actor = None
                app.morphology_view.plotter.render()
            
            if app.morphology_view.plane_actor is not None:
                self._update_morphology_plane()
            return
        
        # Get CC at new Z
        ccs_at_z = app.state.cc_path_3d[z_L2]
        if len(ccs_at_z) == 0:
            return
        
        new_label_id, new_cc_id = ccs_at_z[0]
        new_label_name = app.data.label_names[new_label_id - 1]
        
        # CRITICAL: Check if we crossed a label boundary
        label_changed = (new_label_id != old_label_id)
        
        # Check if we crossed an instance boundary (same label, but disconnected)
        instance_changed = False
        if not label_changed and self.morph_mode != 2:
            # Same label, but check if it's a different instance
            instance_changed = app.cc_tracker._crossed_instance_boundary(
                old_z_L2, z_L2, new_label_id, app.state.cc_path_3d
            )
        
        if _DEBUG_SLICE_UPDATE:
            print(f"[TRACK] z={z_L2}: {new_label_name} CC#{new_cc_id}")
        
        if label_changed:
            print(f"[TRACK] ⚠️  LABEL CHANGED: {app.data.label_names[old_label_id-1]} → {new_label_name}")
        
        if instance_changed:
            print(f"[TRACK] ⚠️  INSTANCE CHANGED (same label, different instance)")
        
        # Update tracked state
        app.state.tracked_state['z_L2'] = z_L2
        app.state.tracked_state['label_id'] = new_label_id
        app.state.tracked_state['label_name'] = new_label_name
        app.state.tracked_state['cc_id'] = new_cc_id
        
        # Rebuild if: label changed OR instance changed (in current-only mode)
        needs_rebuild = False
        
        if label_changed:
            needs_rebuild = True
        elif instance_changed:
            needs_rebuild = True
        
        # Rebuild based on mode
        if self.morph_mode == 0:  # Instance mode
            if needs_rebuild:
                if _DEBUG_SLICE_UPDATE:
                    print(f"[TRACK] Rebuilding morphology for new instance")
                self._update_2d_view()
                self._update_morphology_view()
                self._update_skeleton_marker()
            else:
                self._update_views_on_z_scroll()
                self._update_skeleton_marker()
        
        elif self.morph_mode == 1:  # Label mode
            if label_changed:
                # Only rebuild when label changes
                if _DEBUG_SLICE_UPDATE:
                    print(f"[TRACK] Rebuilding morphology for new label")
                self._update_2d_view()
                self._update_morphology_view()
                self._update_skeleton_marker()
            else:
                # Same label, only update path and guide plane
                self._update_2d_view()
                self._update_skeleton_marker()
                
                # Update path (track new instance)
                path_3d = app.cc_tracker._propagate_cc_path(
                    new_label_id, z_L2, new_cc_id, cross_label=False
                )
                if path_3d:
                    app.state.cc_path_3d = path_3d
                
                self._update_image_host_guide_plane()
        
        else:  # Z Navigation mode (mode 2)
            self._update_views_on_z_scroll()
            self._update_skeleton_marker()

        if app.state.auto_center_enabled:
            QtCore.QTimer.singleShot(50, lambda: _center_on_component(app, from_button=False))

    # =========================================================================
    # COORDINATORS - Multi-view orchestration
    # =========================================================================
    
    def _update_all_views(self):
        """Update all views - COMPLETE REBUILD (for initial click)"""
        app = self.app
        
        if app.state.tracked_state['label_id'] is None:
            return
        
        print(f"\n[UPDATE ALL VIEWS] Full rebuild for {app.state.tracked_state['label_name']} CC#{app.state.tracked_state['cc_id']}")
        
        self._update_2d_view()
        self._update_skeleton_view()
        self._update_morphology_view()  # Full rebuild
        
        try:
            app.napari_view.boundary_layer.events.data()
            app.napari_view.viewer.window.qt_viewer.canvas.update()
        except Exception as e:
            print(f"[WARN] Force refresh failed: {e}")
    
    def _update_views_on_z_scroll(self):
        """Update only lightweight views on Z scroll (NO morphology rebuild)"""
        app = self.app
        
        if app.state.tracked_state['label_id'] is None:
            return
        
        if _DEBUG_SLICE_UPDATE:
            print(f"\n[Z SCROLL] Light update @ z={app.state.tracked_state['z_L2']}")
        
        self._update_2d_view()                  # Update napari boundary
        self._update_skeleton_marker()          # Move black marker
        self._update_image_host_guide_plane()   # Move plane only

    # =========================================================================
    # NAPARI OPS - 2D boundary updates
    # =========================================================================
    
    def _update_2d_view(self):
        """Update 2D boundary - OPTIMIZED with caching for smooth scrolling"""
        app = self.app
        
        label_name = app.state.tracked_state['label_name']
        z_L0 = app.state.tracked_state['z_L0']
        z_L2 = app.state.tracked_state['z_L2']
        cc_id = app.state.tracked_state['cc_id']
        label_id = app.state.tracked_state['label_id']
        
        if _DEBUG_SLICE_UPDATE:
            print(f"\n[2D] Updating @ z_L0={z_L0}, z_L2={z_L2}")
            print(f"[2D] Tracking: {label_name} CC#{cc_id}")
        
        if label_name not in app.data.outer_masks_L2:
            app.napari_view.boundary_layer.data = []
            return
        
        outer_mask = app.data.outer_masks_L2[label_name]
        
        if z_L2 >= outer_mask.shape[0]:
            app.napari_view.boundary_layer.data = []
            return
        
        # === OPTIMIZED: Use slice cache to avoid repeated Zarr reads ===
        slice_L2 = _slice_data_cache.get_outer_slice(outer_mask, label_name, z_L2)
        
        # Use cached CC labeling for faster scroll
        labeled_L2, _ = _cc_label_cache.get(label_id, z_L2, slice_L2 > 0)
        cc_mask_L2 = (labeled_L2 == (cc_id + 1))
        
        if not cc_mask_L2.any():
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] CC#{cc_id} not found at z_L2={z_L2}")
            app.napari_view.boundary_layer.data = []
            return
        
        # Find CC centroid @ L2
        y_coords, x_coords = np.where(cc_mask_L2)
        centroid_y_L2 = np.mean(y_coords)
        centroid_x_L2 = np.mean(x_coords)
        
        # Convert to L0 coordinates
        centroid_y_L0 = centroid_y_L2 * 4
        centroid_x_L0 = centroid_x_L2 * 4
        
        # === OPTIMIZED: Use slice cache for L0 labels ===
        labels_slice_L0 = _slice_data_cache.get_label_slice(app.data.lab_full, z_L0)
        
        # Create binary mask for label_id
        label_mask_L0 = (labels_slice_L0 == label_id)
        
        if not label_mask_L0.any():
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] Label {label_id} not found at z_L0={z_L0}")
            app.napari_view.boundary_layer.data = []
            return
        
        # Label CCs in L0 mask (with caching)
        # Use z_L0 * 1000 + label_id as unique key to avoid collision with L2 cache
        labeled_L0, num_cc_L0 = _cc_label_cache.get(label_id + 1000, z_L0, label_mask_L0)
        
        # Find CC containing centroid
        centroid_y_int = int(round(centroid_y_L0))
        centroid_x_int = int(round(centroid_x_L0))
        
        # Clamp to bounds
        centroid_y_int = max(0, min(centroid_y_int, labeled_L0.shape[0] - 1))
        centroid_x_int = max(0, min(centroid_x_int, labeled_L0.shape[1] - 1))
        
        target_cc_label = labeled_L0[centroid_y_int, centroid_x_int]
        
        if target_cc_label == 0:
            # Centroid not exactly on CC, search nearby (OPTIMIZED: vectorized search)
            search_radius = 20  # pixels @ L0
            
            # Use a small bounding box search instead of loop
            y_min = max(0, centroid_y_int - search_radius)
            y_max = min(labeled_L0.shape[0], centroid_y_int + search_radius + 1)
            x_min = max(0, centroid_x_int - search_radius)
            x_max = min(labeled_L0.shape[1], centroid_x_int + search_radius + 1)
            
            local_labeled = labeled_L0[y_min:y_max, x_min:x_max]
            local_nonzero = np.argwhere(local_labeled > 0)
            
            if len(local_nonzero) > 0:
                # Find closest non-zero point
                local_center = np.array([centroid_y_int - y_min, centroid_x_int - x_min])
                distances = np.sum((local_nonzero - local_center) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                closest_local = local_nonzero[closest_idx]
                target_cc_label = local_labeled[closest_local[0], closest_local[1]]
                if _DEBUG_SLICE_UPDATE:
                    print(f"[2D] Found CC via vectorized search")
            else:
                if _DEBUG_SLICE_UPDATE:
                    print(f"[2D] No CC found near centroid, falling back to L2")
                # Fallback: use L2 upsampled (has error but at least shows something)
                cc_mask_L0 = zoom(cc_mask_L2, 4, order=0)
                target_cc_label = -1  # Mark as fallback
        
        if target_cc_label > 0:
            # Extract from L0 labels
            cc_mask_L0 = (labeled_L0 == target_cc_label)
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] Using L0 labels CC (label={target_cc_label}, area={np.sum(cc_mask_L0)})")
        elif target_cc_label == -1:
            # Fallback already set
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] Using L2 upsampled (fallback)")

        # Store for guide plane contour use
        app.state._current_cc_mask_L0 = cc_mask_L0
        app.state._current_cc_z_L0 = z_L0

        # Extract contour
        contours = measure.find_contours(cc_mask_L0, 0.5)
        
        # Clear old boundaries
        app.napari_view.boundary_layer.data = []
        
        if contours:
            for contour in contours:
                polygon = np.column_stack([
                    np.full(len(contour), z_L0),
                    contour[:, 0],
                    contour[:, 1]
                ])
                app.napari_view.boundary_layer.add_polygons(polygon)
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] ✓ Boundary updated ({len(contours)} contours)")
        else:
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] No contours found")
        
        app.napari_view.boundary_layer.refresh()
        # OPTIMIZED: Remove processEvents during scroll - let Qt batch updates
        # QtWidgets.QApplication.processEvents()

        if app.state.auto_center_enabled:
            QtCore.QTimer.singleShot(50, lambda: _center_on_component(app, from_button=False))

    def _force_refresh_boundary(self):
        """Force refresh boundary layer after click"""
        app = self.app
        
        try:
            app.napari_view.boundary_layer.refresh()
            app.napari_view.viewer.dims.events.current_step()  # Trigger redraw
            QtWidgets.QApplication.processEvents()
            if _DEBUG_SLICE_UPDATE:
                print("[2D] ✓ Forced boundary refresh")
        except Exception as e:
            if _DEBUG_SLICE_UPDATE:
                print(f"[2D] Refresh error: {e}")

    # =========================================================================
    # SKELETON OPS - Marker position updates
    # =========================================================================
    
    def _update_skeleton_view(self):
        """Update skeleton marker (meshes are static)"""
        app = self.app
        
        print(f"\n[SKELETON UPDATE]")
        
        # Clear old marker
        if app.skeleton_view.black_marker_actor:
            try:
                app.skeleton_view.plotter.remove_actor(app.skeleton_view.black_marker_actor)
                print(f"  Removed old marker")
            except Exception as e:
                print(f"  Warning removing old marker: {e}")
            app.skeleton_view.black_marker_actor = None
        
        # Get marker position from metadata
        marker_pos, _ = app.cc_tracker._get_marker_position(
            app.state.tracked_state['label_id'],
            app.state.tracked_state['z_L2'],
            app.state.tracked_state['cc_id']
        )

        # Use kdtree to determine if on skeleton (consistent with Skeleton Host)
        if marker_pos is not None:
            dist, idx = app.data.skeleton_kdtree.query(marker_pos)
            skeleton_radius_um = 3 * app.data.voxel_size_L2[0]
            threshold = skeleton_radius_um * 3.0
            has_skeleton = (dist <= threshold)
        else:
            has_skeleton = False
        
        print(f"  Marker position: {marker_pos}")
        print(f"  Has skeleton: {has_skeleton}")
        
        if marker_pos is None:
            print(f"  ✗ No marker position found")
            app.skeleton_view.skeleton_info.setText(
                f"Tracking: {app.state.tracked_state['label_name']} CC#{app.state.tracked_state['cc_id']}\n"
                f"⚠ No marker position"
            )
            return
        
        # Verify position is within bounds
        if app.data.label_names[0] in app.data.outer_masks_L2:
            shape_L2 = app.data.outer_masks_L2[app.data.label_names[0]].shape
            x_max = (shape_L2[2] - 1) * app.data.voxel_size_L2[2]
            y_max = (shape_L2[1] - 1) * app.data.voxel_size_L2[1]
            z_max = (shape_L2[0] - 1) * app.data.voxel_size_L2[0]
            
            print(f"  Volume bounds: X=[0, {x_max:.1f}], Y=[0, {y_max:.1f}], Z=[0, {z_max:.1f}] μm")
            
            if not (0 <= marker_pos[0] <= x_max and 
                    0 <= marker_pos[1] <= y_max and 
                    0 <= marker_pos[2] <= z_max):
                print(f"  ⚠ Marker position OUT OF BOUNDS!")
        
        try:
            # Create sphere at marker position
            # Match skeleton thickness: skeleton uses ball(3) dilation = 3 voxels radius
            # Marker should be 3x thicker than skeleton for clear visibility
            skeleton_radius_voxels = 3  # From binary_dilation(label_mask, ball(3))
            skeleton_radius_um = skeleton_radius_voxels * app.data.voxel_size_L2[0]
            radius = skeleton_radius_um * 3.0  # 3x thicker than skeleton
            
            sphere = pv.Sphere(radius=radius, center=marker_pos)
            
            color = 'black' if has_skeleton else 'gray'
            opacity = 0.5  # Semi-transparent for visibility
            
            app.skeleton_view.black_marker_actor = app.skeleton_view.plotter.add_mesh(
                sphere,
                color=color,
                opacity=opacity,
                name='marker'
            )

            if app.state.host_mode == 'image':
                app.state.sphere_position = np.array(marker_pos)
            
            print(f"  ✓ Added marker sphere:")
            print(f"    - Position: {marker_pos}")
            print(f"    - Skeleton radius: {skeleton_radius_um:.1f} μm (ball(3) @ {app.data.voxel_size_L2[0]:.2f} μm/voxel)")
            print(f"    - Marker radius: {radius:.1f} μm (3x skeleton)")
            print(f"    - Marker diameter: {radius*2:.1f} μm")
            print(f"    - Color: {color}")
            print(f"    - Opacity: {opacity}")
            
        except Exception as e:
            print(f"  ✗ Failed to add marker: {e}")
            import traceback
            traceback.print_exc()
        
        # Force render
        try:
            app.skeleton_view.plotter.render()
            print(f"  ✓ Skeleton view rendered")
        except Exception as e:
            print(f"  ✗ Render failed: {e}")
        
        # Update info
        app.skeleton_view.skeleton_info.setText(
            f"Tracking: {app.state.tracked_state['label_name']} CC#{app.state.tracked_state['cc_id']}\n"
            f"Position: ({marker_pos[0]:.1f}, {marker_pos[1]:.1f}, {marker_pos[2]:.1f}) μm\n"
            f"Marker: {'✓ Skeleton' if has_skeleton else '✗ No intersection'}"
        )

        if app.state.auto_center_enabled:
            QtCore.QTimer.singleShot(50, lambda: _center_on_component(app, from_button=False))

    def _update_skeleton_marker(self):
        """Update ONLY skeleton marker position (no rebuild)"""
        app = self.app
        
        if app.skeleton_view.black_marker_actor is None:
            # First time - need full update
            self._update_skeleton_view()
            return
        
        label_name = app.state.tracked_state['label_name']
        z_L2 = app.state.tracked_state['z_L2']
        cc_id = app.state.tracked_state['cc_id']
        label_id = app.state.tracked_state['label_id']
        
        # Get marker position from metadata
        label_str = str(label_id)
        
        if (label_str in app.data.cc_metadata['labels'] and
            str(z_L2) in app.data.cc_metadata['labels'][label_str]['layers']):
            
            layer_data = app.data.cc_metadata['labels'][label_str]['layers'][str(z_L2)]
            
            if cc_id < len(layer_data['outer']):
                cc_info = layer_data['outer'][cc_id]
                marker_info = cc_info.get('skeleton_marker', {})
                marker_pos = marker_info.get('position')
                
                if marker_pos:
                    # Update marker position
                    try:
                        skeleton_radius_voxels = 3
                        skeleton_radius_um = skeleton_radius_voxels * app.data.voxel_size_L2[0]
                        radius = skeleton_radius_um * 3.0

                        # Check if on skeleton using kdtree
                        dist, idx = app.data.skeleton_kdtree.query(marker_pos)
                        threshold = skeleton_radius_um * 3.0
                        has_skeleton = (dist <= threshold)
                        color = 'black' if has_skeleton else 'gray'
                        
                        sphere = pv.Sphere(radius=radius, center=marker_pos)
                        app.skeleton_view.black_marker_actor.mapper.SetInputData(sphere)

                        app.skeleton_view.black_marker_actor.GetProperty().SetColor(
                            pv.Color(color).float_rgb
                        )
                        
                        app.skeleton_view.plotter.render()
                        
                        if app.state.host_mode == 'image':
                            app.state.sphere_position = np.array(marker_pos)
                        if _DEBUG_SLICE_UPDATE:
                            print(f"  ✓ Marker moved to z={z_L2}")
                    except Exception as e:
                        if _DEBUG_SLICE_UPDATE:
                            print(f"  ✗ Marker update failed: {e}")
                        # Fallback to full rebuild
                        self._update_skeleton_view()

    # =========================================================================
    # MORPHOLOGY OPS - 3D rendering
    # =========================================================================
    
    def _update_morphology_view(self):
        """Update morphology view - main entry point for Image Host"""
        app = self.app
        
        label_name = app.state.tracked_state['label_name']
        z_L2 = app.state.tracked_state['z_L2']
        cc_id = app.state.tracked_state['cc_id']
        label_id = app.state.tracked_state['label_id']
        
        print(f"\n{'='*80}")
        print(f"[MORPHOLOGY] Starting from {label_name} CC#{cc_id} @ z={z_L2}")
        print(f"[MORPHOLOGY] Image Mode: {self.morph_mode}")
        print(f"{'='*80}")
        
        # === Label Mode same label fast path (only case that doesn't clear actors) ===
        if self.morph_mode == 1:
            if (hasattr(self, '_current_label_mode_label') and 
                self._current_label_mode_label == label_name and
                len(app.morphology_view.label_mode_actors) > 0):
                # Same label, only update guide plane + contour
                print(f"[MORPHOLOGY] Same label {label_name}, update plane only")
                
                path_3d = app.cc_tracker._propagate_cc_path(
                    label_id, z_L2, cc_id, cross_label=False
                )
                if path_3d:
                    app.state.cc_path_3d = path_3d
                
                self._update_image_host_guide_plane()
                
                if app.show_plane_contour_chk.isChecked():
                    self._update_plane_contour()
                
                app.morphology_view.plotter.render()
                return
        
        # === All other cases need to clear old actors ===
        _clear_morphology_actors(app)
        
        # === Mode dispatch ===
        if self.morph_mode == 0:
            self._render_image_instance_mode()
            return
        elif self.morph_mode == 1:
            # Different label, re-render
            self._render_image_label_mode()
            return
        
        # === Mode 2: Z Navigation (original logic) ===
        import time
        t_total = time.time()
        
        t0 = time.time()
        path_3d = app.cc_tracker._propagate_cc_path(
            label_id, z_L2, cc_id, cross_label=True
        )
        print(f"  [TIMING] _propagate_cc_path: {time.time() - t0:.3f}s")

        if not path_3d:
            print("[ERROR] No path found")
            return
        
        # Store for Napari & Skeleton tracking
        app.state.cc_path_3d = path_3d
        
        # === Step 2: Determine render_path based on mode ===
        render_path = path_3d
        print(f"[MORPHOLOGY] Rendering ALL labels in path")
        
        if not render_path:
            print("[ERROR] No render path")
            return
        
        print(f"  Render path spans z=[{min(render_path.keys())}:{max(render_path.keys())}]")
        
        # === Step 3: Calculate bbox ===
        t0 = time.time()
        xy_bbox = app.cc_tracker._calculate_union_xy_bbox_from_path(render_path)
        print(f"  [TIMING] _calculate_union_xy_bbox: {time.time() - t0:.3f}s")
        
        # === Step 4: Extract masks ===
        t0 = time.time()
        label_masks = app.cc_tracker._extract_path_ccs_in_bbox(render_path, xy_bbox)
        print(f"  [TIMING] _extract_path_ccs_in_bbox: {time.time() - t0:.3f}s")

        # === Step 5: Create surfaces per label ===
        t0 = time.time()
        z_min = min(render_path.keys())
        
        for current_label_name, masks in label_masks.items():
            if np.sum(masks['outer']) == 0:
                continue
            
            label_color = _hex_to_rgb(app.data.label_colors[current_label_name])
            
            # Outer mesh
            t_mesh = time.time()
            outer_mesh = app.mesh_builder._create_surface(masks['outer'], xy_bbox, z_min)
            print(f"    [TIMING] _create_surface outer: {time.time() - t_mesh:.3f}s")
            
            if outer_mesh and outer_mesh.n_points > 0:
                app.morphology_view.outer_volume_actors[current_label_name] = app.morphology_view.plotter.add_mesh(
                    outer_mesh,
                    color=label_color,
                    opacity=app.morphology_view.outer_opacity,
                    smooth_shading=True,
                    show_edges=False,
                    name=f'outer_{current_label_name}'
                )
                print(f"    ✓ {current_label_name} outer: {outer_mesh.n_points:,} pts")
            
            # Inner mesh (only if has_inner_mask)
            if app.data.has_inner_mask and np.sum(masks['inner']) > 0:
                t_mesh = time.time()
                inner_mesh = app.mesh_builder._create_surface(masks['inner'], xy_bbox, z_min)
                print(f"    [TIMING] _create_surface inner: {time.time() - t_mesh:.3f}s")

                if inner_mesh and inner_mesh.n_points > 0:
                    app.morphology_view.inner_mesh_actors[current_label_name] = app.morphology_view.plotter.add_mesh(
                        inner_mesh,
                        color=label_color,
                        opacity=app.morphology_view.inner_opacity,
                        smooth_shading=True,
                        show_edges=False,
                        name=f'inner_{current_label_name}'
                    )
                    print(f"    ✓ {current_label_name} inner: {inner_mesh.n_points:,} pts")

        print(f"  [TIMING] Step 5 (surfaces + add_mesh): {time.time() - t0:.3f}s")
        # === Step 6: Zoom ===
        _zoom_to_path_bbox(app, xy_bbox, z_L2)
        
        # === Step 7: Guide plane ===
        self._update_image_host_guide_plane()

        if app.show_plane_contour_chk.isChecked():
            app.show_plane_contour_chk.blockSignals(True)
            app.show_plane_contour_chk.setChecked(False)
            app.show_plane_contour_chk.setChecked(True)
            app.show_plane_contour_chk.blockSignals(False)
            self._update_plane_contour()
        
        app.morphology_view.plotter.render()
        
        print(f"\n  [TIMING] TOTAL: {time.time() - t_total:.3f}s")
        print(f"\n[MORPHOLOGY] Complete!")
        print(f"{'='*80}\n")

    def _render_image_instance_mode(self):
        """Image Host Mode 0: Show current instance using prebuilt mesh"""
        app = self.app
        
        label_name = app.state.tracked_state['label_name']
        z_L2 = app.state.tracked_state['z_L2']
        cc_id = app.state.tracked_state['cc_id']
        label_id = app.state.tracked_state['label_id']
        
        print(f"[INSTANCE MODE] {label_name} CC#{cc_id} @ z_L2={z_L2}")
        
        if app.data.image_host_mesh_info is None:
            print("[ERROR] image_host_meshes.json not loaded")
            return
        
        if label_name not in app.data.image_host_mesh_info['labels']:
            print(f"[ERROR] {label_name} not in image_host_mesh_info")
            return
        
        # === Step 1: Build cc_path_3d (same as Z Navigation) ===
        path_3d = app.cc_tracker._propagate_cc_path(
            label_id, z_L2, cc_id, cross_label=True
        )
        
        if not path_3d:
            print("[ERROR] No path found")
            return
        
        app.state.cc_path_3d = path_3d
        
        print(f"  Instance path: z=[{min(app.state.cc_path_3d.keys())}:{max(app.state.cc_path_3d.keys())}]")
        
        # === Step 2: Look up prebuilt mesh (using L2 coords directly) ===
        label_info = app.data.image_host_mesh_info['labels'][label_name]
        z_str = str(z_L2)
        
        if z_str not in label_info['z_2dcc_to_3dcc']:
            print(f"[ERROR] z={z_L2} not in lookup table")
            return
        
        z_lookup = label_info['z_2dcc_to_3dcc'][z_str]
        cc_id_str = str(cc_id)
        
        if cc_id_str not in z_lookup:
            print(f"[ERROR] cc_id={cc_id} not in z={z_L2} lookup")
            return
        
        cc_3d_id = z_lookup[cc_id_str]
        print(f"  → 3D CC {cc_3d_id}")
        
        # === Step 3: Find corresponding mesh info ===
        mesh_info = None
        for m in label_info['meshes']:
            if m['id'] == cc_3d_id:
                mesh_info = m
                break
        
        if mesh_info is None:
            print(f"[ERROR] 3D CC {cc_3d_id} mesh not found")
            return
        
        # === Step 4: Load prebuilt mesh ===
        label_color = _hex_to_rgb(app.data.label_colors[label_name])
        
        if mesh_info['outer_mesh']:
            outer_path = app.data.zarr_path.parent / mesh_info['outer_mesh']
            outer_mesh = pv.read(str(outer_path))
            app.morphology_view.outer_volume_actors[label_name] = app.morphology_view.plotter.add_mesh(
                outer_mesh,
                color=label_color,
                opacity=app.morphology_view.outer_opacity,
                smooth_shading=True,
                name=f'outer_{label_name}'
            )
            print(f"  ✓ Outer mesh: {outer_mesh.n_points:,} pts")
        
        if app.data.has_inner_mask and mesh_info.get('inner_mesh'):
            inner_path = app.data.zarr_path.parent / mesh_info['inner_mesh']
            inner_mesh = pv.read(str(inner_path))
            app.morphology_view.inner_mesh_actors[label_name] = app.morphology_view.plotter.add_mesh(
                inner_mesh,
                color=label_color,
                opacity=app.morphology_view.inner_opacity,
                smooth_shading=True,
                name=f'inner_{label_name}'
            )
            print(f"  ✓ Inner mesh: {inner_mesh.n_points:,} pts")
        
        # === Step 5: Guide plane + contour ===
        self._update_image_host_guide_plane()
        
        if app.show_plane_contour_chk.isChecked():
            app.show_plane_contour_chk.blockSignals(True)
            app.show_plane_contour_chk.setChecked(False)
            app.show_plane_contour_chk.setChecked(True)
            app.show_plane_contour_chk.blockSignals(False)
            self._update_plane_contour()
        
        self._zoom_to_mesh_bounds(mesh_info)
        app.morphology_view.plotter.render()
        
        print(f"[INSTANCE MODE] Complete!")

    def _render_image_label_mode(self):
        """Image Host Mode 1: Show all prebuilt meshes for current label (PARALLEL)"""
        app = self.app
        
        label_name = app.state.tracked_state['label_name']
        label_id = app.state.tracked_state['label_id']
        z_L2 = app.state.tracked_state['z_L2']
        cc_id = app.state.tracked_state['cc_id']
        
        print(f"[LABEL MODE] {label_name} @ z={z_L2}")
        
        if app.data.image_host_mesh_info is None:
            print("[ERROR] image_host_meshes.json not loaded")
            return
        
        if label_name not in app.data.image_host_mesh_info['labels']:
            print(f"[ERROR] {label_name} not in image_host_mesh_info")
            return
        
        # === Step 1: Build label-only path ===
        path_3d = app.cc_tracker._propagate_cc_path(
            label_id, z_L2, cc_id, cross_label=False
        )
        if path_3d:
            app.state.cc_path_3d = path_3d
            print(f"  Label path: z=[{min(path_3d.keys())}:{max(path_3d.keys())}]")
        
        # === Step 2: Store current label ===
        self._current_label_mode_label = label_name
        
        # === Step 3: Collect mesh paths ===
        label_info = app.data.image_host_mesh_info['labels'][label_name]
        label_color = _hex_to_rgb(app.data.label_colors[label_name])
        
        mesh_paths = []  # [(key, filepath), ...]
        mesh_types = {}  # {key: 'outer' or 'inner'}
        
        for mesh_info in label_info['meshes']:
            cc_3d_id = mesh_info['id']
            
            if mesh_info['outer_mesh']:
                key = f'outer_{label_name}_{cc_3d_id}'
                filepath = app.data.zarr_path.parent / mesh_info['outer_mesh']
                mesh_paths.append((key, filepath))
                mesh_types[key] = 'outer'
            
            if app.data.has_inner_mask and mesh_info.get('inner_mesh'):
                key = f'inner_{label_name}_{cc_3d_id}'
                filepath = app.data.zarr_path.parent / mesh_info['inner_mesh']
                mesh_paths.append((key, filepath))
                mesh_types[key] = 'inner'
        
        print(f"  Loading {len(mesh_paths)} meshes (parallel)...")
        
        # === Step 4: Parallel load ===
        import time
        t0 = time.time()
        loaded_meshes = _load_meshes_parallel(mesh_paths, max_workers=6)
        print(f"  ✓ Loaded {len(loaded_meshes)} meshes in {time.time()-t0:.2f}s")
        
        # === Step 5: Add to plotter ===
        plotter = app.morphology_view.plotter
        plotter.suppress_rendering = True  
        
        try:
            for key, mesh in loaded_meshes.items():
                mesh_type = mesh_types.get(key, 'outer')
                opacity = app.morphology_view.outer_opacity if mesh_type == 'outer' else app.morphology_view.inner_opacity
                
                actor = plotter.add_mesh(
                    mesh,
                    color=label_color,
                    opacity=opacity,
                    smooth_shading=True,
                    name=key
                )
                app.morphology_view.label_mode_actors[key] = actor
        finally:
            plotter.suppress_rendering = False  
        
        # === Step 6: Guide plane + contour ===
        self._update_image_host_guide_plane()
        
        if app.show_plane_contour_chk.isChecked():
            app.show_plane_contour_chk.blockSignals(True)
            app.show_plane_contour_chk.setChecked(False)
            app.show_plane_contour_chk.setChecked(True)
            app.show_plane_contour_chk.blockSignals(False)
            self._update_plane_contour()
        
        app.morphology_view.plotter.reset_camera()
        app.morphology_view.plotter.render()
        
        print(f"[LABEL MODE] Complete!")

    def _update_morphology_plane(self):
        """Update ONLY morphology plane position (no rebuild)"""
        app = self.app
        
        if app.morphology_view.plane_actor is None:
            return
        
        z_L2 = app.state.tracked_state['z_L2']
        z_world = z_L2 * app.data.voxel_size_L2[0]
        
        try:
            # Get current plane properties
            plane_mesh = app.morphology_view.plane_actor.mapper.GetInput()
            
            # Calculate Z offset
            current_center = plane_mesh.center
            z_offset = z_world - current_center[2]
            
            # Move plane
            plane_mesh.translate([0, 0, z_offset], inplace=True)
            
            # Move scale bar if exists
            if app.morphology_view.scale_line_actor:
                line_mesh = app.morphology_view.scale_line_actor.mapper.GetInput()
                line_mesh.translate([0, 0, z_offset], inplace=True)

            # Update contour if checkbox is checked
            if app.show_plane_contour_chk.isChecked():
                if _DEBUG_SLICE_UPDATE:
                    print(f"  [PLANE] Updating contour @ z={z_L2}")
                self._update_plane_contour()

            app.morphology_view.plotter.render()
            if _DEBUG_SLICE_UPDATE:
                print(f"  ✓ Plane moved to z={z_L2}")
            
        except Exception as e:
            if _DEBUG_SLICE_UPDATE:
                print(f"  ✗ Plane update failed: {e}")

    def _update_image_host_guide_plane(self):
        """Image Host: Update guide plane using CC bbox"""
        app = self.app
        
        # Remove old actors
        for attr in ['plane_actor', 'scale_line_actor', 'scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)
        
        if not app.show_plane_chk.isChecked():
            if app.show_plane_contour_chk.isChecked():
                self._update_plane_contour()
                app.morphology_view.plotter.render()
            return
        
        # Debug output (disabled for performance)
        # if _DEBUG_SLICE_UPDATE:
        #     print(f"  [DEBUG] _current_cc_mask_L0 exists: {app.state._current_cc_mask_L0 is not None}")
        #     if app.state._current_cc_mask_L0 is not None:
        #         print(f"  [DEBUG] _current_cc_mask_L0.any(): {app.state._current_cc_mask_L0.any()}")

        z_L0 = app.state.tracked_state['z_L0']
        label_id = app.state.tracked_state['label_id']
        
        if label_id is None:
            return
        
        # Use CC mask stored by _update_2d_view
        if app.state._current_cc_mask_L0 is not None and app.state._current_cc_mask_L0.any():
            cc_mask = app.state._current_cc_mask_L0
            
            # Calculate bbox from CC mask
            y_coords, x_coords = np.where(cc_mask)
            y_min_L0, y_max_L0 = y_coords.min(), y_coords.max()
            x_min_L0, x_max_L0 = x_coords.min(), x_coords.max()
            
            center_x = (x_min_L0 + x_max_L0) / 2 * app.data.voxel_size_L0[2]
            center_y = (y_min_L0 + y_max_L0) / 2 * app.data.voxel_size_L0[1]
            plane_width = (x_max_L0 - x_min_L0) * app.data.voxel_size_L0[2]
            plane_height = (y_max_L0 - y_min_L0) * app.data.voxel_size_L0[1]
            
            # Add margin
            margin = 50  # μm
            plane_width += margin * 2
            plane_height += margin * 2
            
            # Store for next use
            app.state._last_plane_center = (center_x, center_y)
            app.state._last_plane_size = (plane_width, plane_height)
        
        elif app.state._last_plane_size is not None:
            # No CC, use last size
            center_x, center_y = app.state._last_plane_center
            plane_width, plane_height = app.state._last_plane_size
        else:
            return
        
        # Level 0 coords → world coords
        z_world = z_L0 * app.data.voxel_size_L0[0]
        
        # Create plane
        plane = pv.Plane(
            center=(center_x, center_y, z_world),
            direction=(0, 0, 1),
            i_size=plane_width,
            j_size=plane_height
        )
        
        app.morphology_view.plane_actor = app.morphology_view.plotter.add_mesh(
            plane,
            color='black',
            opacity=0.3,
            name='guide_plane'
        )
        
        # Scale bar
        line_start = [center_x - plane_width/2, center_y + plane_height/2 + 20, z_world]
        line_end = [center_x + plane_width/2, center_y + plane_height/2 + 20, z_world]
        
        line = pv.Line(line_start, line_end)
        app.morphology_view.scale_line_actor = app.morphology_view.plotter.add_mesh(
            line,
            color='black',
            line_width=3,
            render_lines_as_tubes=True,
            name='scale_line'
        )
        
        scale_text = f"{plane_width:.0f} μm"
        app.morphology_view.scale_text_actor = app.morphology_view.plotter.add_text(
            scale_text,
            position='upper_right',
            font_size=10,
            color='black',
            name='scale_text'
        )
        
        # Contour (if enabled)
        if app.show_plane_contour_chk.isChecked():
            self._update_plane_contour()
        
        app.morphology_view.plotter.render()
        if _DEBUG_SLICE_UPDATE:
            print(f"  [PLANE] {plane_width:.0f} × {plane_height:.0f} μm @ z={z_world:.1f}")

    def _update_plane_contour(self):
        """Update plane contour - use result from _update_2d_view"""
        app = self.app
        
        # Remove old contour
        if app.morphology_view.contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.contour_actor)
            except:
                pass
            app.morphology_view.contour_actor = None
        
        # Check if current z is in path range
        z_L2 = app.state.tracked_state['z_L2']
        if app.state.cc_path_3d is not None:
            if z_L2 not in app.state.cc_path_3d:
                print(f"  [CONTOUR] z={z_L2} not in path, skip contour")
                return
        
        # Use CC mask stored by 2d view
        if app.state._current_cc_mask_L0 is None:
            return
        
        if app.state._current_cc_z_L0 is None:
            return
        
        cc_mask_L0 = app.state._current_cc_mask_L0
        z_L0 = app.state._current_cc_z_L0
        
        if not cc_mask_L0.any():
            return
        
        # Extract contours
        contours = measure.find_contours(cc_mask_L0.astype(float), 0.5)
        
        if not contours:
            return
        
        z_world = z_L0 * app.data.voxel_size_L0[0]
        
        for contour in contours:
            points_3d = np.zeros((len(contour), 3))
            points_3d[:, 0] = contour[:, 1] * app.data.voxel_size_L0[2]
            points_3d[:, 1] = contour[:, 0] * app.data.voxel_size_L0[1]
            points_3d[:, 2] = z_world
            
            points_closed = np.vstack([points_3d, points_3d[0:1]])
            
            lines = []
            for i in range(len(points_closed) - 1):
                lines.extend([2, i, i + 1])
            
            line_mesh = pv.PolyData(points_closed, lines=lines)
            
            app.morphology_view.contour_actor = app.morphology_view.plotter.add_mesh(
                line_mesh,
                color='black',
                line_width=4,
                render_lines_as_tubes=True,
                opacity=1.0,
                name='plane_contour'
            )
            break  # Only first contour
    
    def _zoom_to_mesh_bounds(self, mesh_info):
        """Zoom camera to mesh bounding box"""
        app = self.app
        
        z_min, z_max = mesh_info['z_range']
        z_center = (z_min + z_max) / 2 * app.data.voxel_size_L2[0]
        
        # Simple reset_camera, or can position more precisely
        app.morphology_view.plotter.reset_camera()