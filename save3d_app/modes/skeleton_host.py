"""
Skeleton Host Mode

Handles all operations when Skeleton 3D view is the primary host.

Morph Modes:
    0: Instance - show single skeleton instance prebuilt mesh
    1: Navigation - sphere navigates, shows local morphology
    2: Selection - draw on skeleton to select regions

OPTIMIZATIONS:
    - LRU cache for 2D CC labeling (shared with image_host)
    - GPU-accelerated CC labeling when available
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
from skimage import measure
from qtpy import QtCore, QtWidgets
import pyvista as pv

from ..controls import _center_on_component, _zoom_to_path_bbox, _clear_morphology_actors
from ..utils import _hex_to_rgb

# Import shared CC label cache from image_host
from .image_host import _cc_label_cache, _slice_data_cache, _GPU_CC_AVAILABLE


class SkeletonHost:
    """
    Skeleton Host mode - 3D Skeleton sphere drives navigation
    """
    
    def __init__(self, app):
        """
        Initialize with app reference
        """
        self.app = app
        
        # Morph mode: 0=Instance, 1=Navigation, 2=Selection
        self.morph_mode = 1  # Default to Navigation mode
        
        # Instance mode state
        self.current_skeleton_instance_id = -1
        
        # Selection mode state
        self.selection_drawing = False
        self.selected_skeleton_indices = set()
        self.last_drawing_idx = None
        
        # Selection mode caching
        self._selected_list = None
        self._selected_kdtree = None
        
        # Mode changing flag
        self._mode_changing = False
        self._skip_instance_clear = False
        self._selection_sphere_cleaned = False

        # Selection highlight throttle
        self._highlight_update_timer = QtCore.QTimer()
        self._highlight_update_timer.setSingleShot(True)
        self._highlight_update_timer.timeout.connect(self._update_selection_highlight)

        self._pending_sphere_pos = None
        self._sphere_update_timer = QtCore.QTimer()
        self._sphere_update_timer.setSingleShot(True)
        self._sphere_update_timer.timeout.connect(self._on_sphere_update_timeout)
    # =========================================================================
    # ENTRY POINTS - External callbacks
    # =========================================================================
    
    def _switch_to_skeleton_host(self):
        """Switch to Skeleton Host mode"""
        app = self.app
        
        print("\n[HOST MODE] → SKELETON HOST")
        app.image_morph_mode_widget.setVisible(False)
        app.morph_mode_widget.setVisible(True)
        app.state.host_mode = 'skeleton'
        
        # Force reset to Navigation mode
        self.morph_mode = 1
        app.morph_navigation_radio.setChecked(True)
        app.selection_buttons_widget.setVisible(False)

        # Clear selection state
        self._clear_selection_highlight()
        self.selected_skeleton_indices = set()
        self.selection_drawing = False
        if hasattr(app, 'draw_selection_btn'):
            app.draw_selection_btn.setChecked(False)
        self._selected_kdtree = None
        self._selected_list = None

        # Hide marker_actor
        if app.skeleton_view.black_marker_actor:
            app.skeleton_view.black_marker_actor.visibility = False

        # Check if we have tracking
        has_tracking = app.state.tracked_state['label_id'] is not None
        
        if has_tracking and app.skeleton_view.black_marker_actor:
            # Has tracking: sync marker → sphere
            try:
                marker_center = app.skeleton_view.black_marker_actor.mapper.GetInput().center
                # Snap to nearest skeleton point
                dist, idx = app.data.skeleton_kdtree.query(marker_center)
                nearest_pos = app.data.skeleton_coords[idx]
                app.state.sphere_position = nearest_pos.copy()

                if app.skeleton_view.sphere_widget:
                    app.skeleton_view.sphere_widget.SetCenter(*nearest_pos)
                
                print(f"  ✓ Sphere synced to marker: {marker_center}")
            except Exception as e:
                print(f"  ⚠ Sync failed: {e}")
                app.state.sphere_position = app.data.skeleton_coords[0].copy()
                if app.skeleton_view.sphere_widget:
                    app.skeleton_view.sphere_widget.SetCenter(*app.state.sphere_position)
                    
            # Show sphere widget
            if app.skeleton_view.sphere_widget:
                app.skeleton_view.sphere_widget.On()
                try:
                    app.skeleton_view.sphere_widget.GetSphereProperty().SetOpacity(0.5)
                except:
                    pass
        else:
            # No tracking: sphere at first skeleton point, wait for user click
            print("  [INFO] No tracking - waiting for user to click skeleton")
            app.state.sphere_position = app.data.skeleton_coords[0].copy()
            if app.skeleton_view.sphere_widget:
                app.skeleton_view.sphere_widget.SetCenter(*app.state.sphere_position)
        
        app.skeleton_view.plotter.render()

        _clear_morphology_actors(app)
        
        # === Initialize Skeleton Host morphology ===
        self._init_skeleton_host_morphology()

        print(f"  [DEBUG] global_morph_meshes: {len(app.morphology_view.global_morph_meshes)} meshes")
        print(f"  [DEBUG] global_morph_actors: {len(app.morphology_view.global_morph_actors)} actors")
        
        # === Initial morphology update (only if has tracking) ===
        if has_tracking and app.state.sphere_position is not None:
            self._update_skeleton_host_morphology(app.state.sphere_position)
            self._on_sphere_position_changed(app.state.sphere_position)

        # Adjust splitter ratio
        if hasattr(app, 'main_splitter'):
            total = sum(app.main_splitter.sizes())
            app.main_splitter.setSizes([int(total * 0.4), int(total * 0.6)])
        
        print("  ✓ Sphere controls active, marker_actor hidden")
        app.morph_mode_widget.setVisible(True)

        try:
            dims_widget = app.napari_view.viewer.window._qt_viewer.dims
            dims_widget.setEnabled(False)
        except:
            pass
    
    def _on_sphere_drag_lightweight(self, world_pos):
        app = self.app
        
        if app.state.host_mode != 'skeleton':
            return
        
        dist, idx = app.data.skeleton_kdtree.query(world_pos)
        label_id = int(app.data.skeleton_labels[idx])
        
        if label_id > 0:
            label_name = app.data.label_names[label_id - 1]
            z_world = world_pos[2]
            app.slice_info.setText(f"Z = {z_world:.1f} μm  |  {label_name} (dragging...)")
        
        self._pending_sphere_pos = world_pos.copy()
        self._sphere_update_timer.start(80)  

    def _on_sphere_update_timeout(self):
        if self._pending_sphere_pos is not None:
            self._on_sphere_position_changed(self._pending_sphere_pos)
            self._pending_sphere_pos = None

    def _on_sphere_position_changed(self, world_pos):
        """Handle sphere position change in Skeleton Host mode"""
        app = self.app
        
        if app.state.host_mode != 'skeleton':
            return
        
        print(f"\n[SKELETON HOST] Sphere moved to: {world_pos}")
        
        # Find nearest skeleton point
        dist, idx = app.data.skeleton_kdtree.query(world_pos)
        
        # Get label/CC info from preprocessing (no centroid matching needed)
        label_id = int(app.data.skeleton_labels[idx])
        cc_id = int(app.data.skeleton_cc_ids[idx])      # Already level 0
        z_L0 = int(app.data.skeleton_z_L0[idx])         # Already level 0
        
        if label_id == 0:
            print(f"  ⚠ No label at this position")
            return
        
        if cc_id < 0 or z_L0 < 0:
            print(f"  ⚠ No valid CC mapping (cc_id={cc_id}, z_L0={z_L0})")
            app.napari_view.boundary_layer.data = []
            app.state.tracked_state['cc_id'] = -1  # Mark invalid
            return
    
        label_name = app.data.label_names[label_id - 1]
        
        print(f"  Label: {label_name} (id={label_id})")
        print(f"  CC#{cc_id} @ z_L0={z_L0} (from preprocessing)")
        
        # z_L2 from voxel coords, for morphology view
        z_L2 = int(app.data.skeleton_coords_voxel[idx][0])
        
        # Update tracked state
        app.state.tracked_state['z_L0'] = z_L0
        app.state.tracked_state['z_L2'] = z_L2
        app.state.tracked_state['label_id'] = label_id
        app.state.tracked_state['label_name'] = label_name
        app.state.tracked_state['cc_id'] = cc_id
        
        # Update Napari slice
        self._update_napari_from_skeleton(z_L0)
        
        # Update 2D boundary directly (no centroid matching needed)
        self._update_2d_view_direct()
        
        # Propagate CC path
        app.state.cc_path_3d = app.cc_tracker._propagate_cc_path(
            label_id, z_L2, cc_id, cross_label=True
        )
        
        # Update slice info
        total_slices = app.data.lab_full.shape[0]
        z_world = world_pos[2]
        app.slice_info.setText(
            f"Slice: {z_L0} / {total_slices}  |  "
            f"Z = {z_world:.1f} μm  |  "
            f"Skeleton Host: {label_name} CC#{cc_id}"
        )
        
        # Update morphology based on current mode
        if self.morph_mode == 1:  # Navigation
            self._update_skeleton_host_morphology(world_pos)
        elif self.morph_mode == 0:  # Instance
            self._update_skeleton_host_instance_mode(world_pos, idx)
        elif self.morph_mode == 2:  # Selection
            # Selection mode: sphere only navigates, don't update morphology (fixed at draw end)
            self._update_skeleton_host_contour(world_pos)
            self._update_skeleton_host_guide_plane(world_pos)
            app.morphology_view.plotter.render()

        if app.state.auto_center_enabled:
            QtCore.QTimer.singleShot(50, lambda: _center_on_component(app, from_button=False))

    def _on_morph_mode_changed_skeleton(self, button):
        """Handle morphology mode change in Skeleton Host"""
        app = self.app
        
        mode_id = app.morph_mode_group.id(button)
        mode_names = {0: 'Instance', 1: 'Navigation', 2: 'Selection'}
        print(f"\n[MORPH MODE] Changed to: {mode_names[mode_id]}")

        self._mode_changing = True

        self.morph_mode = mode_id
        self.current_skeleton_instance_id = -1
        
        # Show/hide Selection buttons
        app.selection_buttons_widget.setVisible(mode_id == 2)
        
        # Clean up when leaving Selection mode
        if mode_id != 2:
            self._exit_selection_mode()

        _clear_morphology_actors(app)

        # Clear Skeleton Host global morph actors
        for mesh_name, data in list(app.morphology_view.global_morph_actors.items()):
            try:
                app.morphology_view.plotter.remove_actor(data['actor'])
            except:
                pass
        app.morphology_view.global_morph_actors = {}
        
        # Clear skeleton host contour
        if app.morphology_view.skeleton_host_contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
            except:
                pass
            app.morphology_view.skeleton_host_contour_actor = None

        app.napari_view.boundary_layer.data = []
        
        self._mode_changing = False
        
        if app.state.sphere_position is None:
            return
        
        if mode_id == 0:  # Instance
            self.current_skeleton_instance_id = -1  # Reset
            self._skip_instance_clear = True
            if app.state.sphere_position is not None:
                self._on_sphere_position_changed(app.state.sphere_position)
            self._skip_instance_clear = False
            
        elif mode_id == 1:  # Navigation
            # Re-initialize global morphology meshes (cleared above)
            self._init_skeleton_host_morphology()
            # Then update morphology view
            if app.state.sphere_position is not None:
                self._on_sphere_position_changed(app.state.sphere_position)
            
        elif mode_id == 2:  # Selection
            self._selection_sphere_cleaned = False
            _clear_morphology_actors(app)
            
            # Clear Navigation mode contour
            if app.morphology_view.skeleton_host_contour_actor:
                try:
                    app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
                except:
                    pass
                app.morphology_view.skeleton_host_contour_actor = None
            
            # Clear Navigation mode guide plane
            for attr in ['skeleton_host_plane_actor', 'skeleton_host_scale_actor', 'skeleton_host_scale_text_actor']:
                actor = getattr(app.morphology_view, attr, None)
                if actor:
                    try:
                        app.morphology_view.plotter.remove_actor(actor)
                    except:
                        pass
                    setattr(app.morphology_view, attr, None)
            
            # Also clear Image Host plane actors (just in case)
            for attr in ['plane_actor', 'scale_line_actor', 'scale_text_actor']:
                actor = getattr(app.morphology_view, attr, None)
                if actor:
                    try:
                        app.morphology_view.plotter.remove_actor(actor)
                    except:
                        pass
                    setattr(app.morphology_view, attr, None)
            
            app.morphology_view.plotter.render()
            print("  [Selection] Ready - click Draw to start")
    
    # =========================================================================
    # MODE 0: INSTANCE
    # =========================================================================
    
    def _update_skeleton_host_instance_mode(self, world_pos, skeleton_idx):
        """
        Instance Mode: Show prebuilt mesh for current skeleton instance
        """
        app = self.app
        
        # Find instance_id for this point
        if skeleton_idx >= len(app.data.skeleton_instance_ids):
            return
        
        instance_id = int(app.data.skeleton_instance_ids[skeleton_idx])
        
        if instance_id < 0:
            print(f"  [INSTANCE MODE] No instance at this point")
            return
        
        # If still same instance, only update contour and guide plane
        if instance_id == self.current_skeleton_instance_id:
            print(f"  [INSTANCE MODE] Same instance {instance_id}, updating contour/plane only")
            self._update_skeleton_host_contour(world_pos)
            self._update_skeleton_host_guide_plane(world_pos)
            app.morphology_view.plotter.render()
            return
        
        # Different instance, reload mesh
        print(f"  [INSTANCE MODE] Switching to instance {instance_id}")
        self.current_skeleton_instance_id = instance_id
        
        # Only clear if not during mode change (avoid double clear crash)
        if not self._skip_instance_clear:
            _clear_morphology_actors(app)
        
        # Find instance info
        inst_info = None
        for inst in app.data.skeleton_instances:
            if inst['id'] == instance_id:
                inst_info = inst
                break
        
        if inst_info is None:
            print(f"  [INSTANCE MODE] Instance {instance_id} info not found")
            return
        
        label_name = inst_info['label_name']
        label_color = _hex_to_rgb(app.data.label_colors[label_name])
        
        print(f"  [INSTANCE MODE] Loading {label_name} instance {instance_id}")
        print(f"    Z range: {inst_info.get('z_range_L2', 'N/A')}")
        print(f"    Points: {inst_info.get('point_count', 'N/A')}")
        
        base_path = app.data.zarr_path.parent
        
        # Load outer mesh
        outer_mesh_path = inst_info.get('outer_mesh')
        if outer_mesh_path:
            full_path = base_path / outer_mesh_path
            if full_path.exists():
                mesh = pv.read(str(full_path))
                app.morphology_view.outer_volume_actors[label_name] = app.morphology_view.plotter.add_mesh(
                    mesh,
                    color=label_color,
                    opacity=app.morphology_view.outer_opacity,
                    smooth_shading=True,
                    show_edges=False,
                    name=f'inst_outer_{label_name}'
                )
                print(f"    ✓ Outer mesh: {mesh.n_points:,} pts (opacity={app.morphology_view.outer_opacity})")
            else:
                print(f"    ✗ Outer mesh not found: {full_path}")
        
        # Load inner mesh (only if has_inner_mask)
        if app.data.has_inner_mask:
            inner_mesh_path = inst_info.get('inner_mesh')
            if inner_mesh_path:
                full_path = base_path / inner_mesh_path
                if full_path.exists():
                    mesh = pv.read(str(full_path))
                    app.morphology_view.inner_mesh_actors[label_name] = app.morphology_view.plotter.add_mesh(
                        mesh,
                        color=label_color,
                        opacity=app.morphology_view.inner_opacity,
                        smooth_shading=True,
                        show_edges=False,
                        name=f'inst_inner_{label_name}'
                    )
                    print(f"    ✓ Inner mesh: {mesh.n_points:,} pts (opacity={app.morphology_view.inner_opacity})")
                else:
                    print(f"    ✗ Inner mesh not found: {full_path}")
        
        # Calculate bounds from all meshes
        all_bounds = []
        for actor_dict in [app.morphology_view.outer_volume_actors, app.morphology_view.inner_mesh_actors]:
            for name, actor in actor_dict.items():
                if actor and hasattr(actor, 'bounds'):
                    all_bounds.append(actor.bounds)
        
        if all_bounds:
            # Merge bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            xmin = min(b[0] for b in all_bounds)
            xmax = max(b[1] for b in all_bounds)
            ymin = min(b[2] for b in all_bounds)
            ymax = max(b[3] for b in all_bounds)
            zmin = min(b[4] for b in all_bounds)
            zmax = max(b[5] for b in all_bounds)
            
            center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
            
            # Calculate appropriate camera distance
            diagonal = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
            distance = diagonal * 1.5
            
            self._update_morphology_camera_to_sphere(center, distance / 4.0)
            #app.morphology_view.plotter.camera.focal_point = center
            #app.morphology_view.plotter.camera.position = (center[0], center[1], center[2] - distance)

        # Update contour and guide plane
        self._update_skeleton_host_contour(world_pos)
        self._update_skeleton_host_guide_plane(world_pos)
        
        app.morphology_view.plotter.render()
        print(f"  [INSTANCE MODE] ✓ Instance {instance_id} loaded")

    # =========================================================================
    # MODE 1: NAVIGATION
    # =========================================================================
    
    def _init_skeleton_host_morphology(self):
        """
        Initialize morphology view for Skeleton Host Navigation mode
        """
        app = self.app
        
        if not app.morphology_view.morph_meshes_loaded:
            app.morphology_view._load_global_morphology_meshes()
        
        if not app.morphology_view.global_morph_meshes:
            print("[WARN] No morphology meshes available")
            return
        
        print("\n=== Initializing Skeleton Host Morphology View ===")
        
        # Clear any existing actors
        for mesh_name in list(app.morphology_view.global_morph_actors.keys()):
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.global_morph_actors[mesh_name]['actor'])
            except:
                pass
        app.morphology_view.global_morph_actors = {}
        
        for mesh_name, mesh in app.morphology_view.global_morph_meshes.items():
            label_name = mesh_name.split('_', 1)[1]
            color = _hex_to_rgb(app.data.label_colors[label_name])
            
            is_outer = mesh_name.startswith('outer_')
            if is_outer:
                base_opacity = 0.3 if app.data.has_inner_mask else 0.8  # No inner → solid outer
            else:
                base_opacity = 0.8
            
            # Initialize RGBA (fully transparent)
            n_points = mesh.n_points
            rgba = np.zeros((n_points, 4), dtype=np.uint8)
            rgba[:, 0] = int(color[0] * 255)
            rgba[:, 1] = int(color[1] * 255)
            rgba[:, 2] = int(color[2] * 255)
            rgba[:, 3] = 0  # Fully transparent
            
            mesh.point_data['rgba'] = rgba
            
            actor = app.morphology_view.plotter.add_mesh(
                mesh,
                scalars='rgba',
                rgba=True,
                smooth_shading=True,
                show_edges=False,
                name=f'global_{mesh_name}'
            )
            
            app.morphology_view.global_morph_actors[mesh_name] = {
                'actor': actor,
                'mesh': mesh,
                'color': color,
                'base_opacity': base_opacity
            }
            
            print(f"  ✓ {mesh_name}: {n_points:,} vertices")
        
        app.morphology_view.plotter.render()
        print("✓ Skeleton Host morphology initialized")

    def _update_skeleton_host_morphology(self, sphere_center):
        """
        Sphere-based morphology view with dynamic radius based on current CC size
        """
        import time
        t_total = time.time()

        app = self.app
        
        if not app.morphology_view.global_morph_actors:
            return
        
        sphere_center = np.array(sphere_center)
        
        # Step 1: Find current CC and calculate dynamic radius
        t0 = time.time()
        range_um = self._get_dynamic_sphere_radius(sphere_center)
        range_um_sq = range_um * range_um
        print(f"  [TIMING] _get_dynamic_sphere_radius: {time.time() - t0:.3f}s")
        
        # Step 2: Calculate per-vertex opacity (sphere-based)
        t0 = time.time()
        for mesh_name, data in app.morphology_view.global_morph_actors.items():
            mesh = data['mesh']
            base_opacity = data['base_opacity']
            color = data['color']
            
            vertices = mesh.points
            n_points = len(vertices)
            
            '''
            # Distance to sphere center
            distances = np.linalg.norm(vertices - sphere_center, axis=1)
            # Inside range → show, outside range → hide
            opacity = np.where(distances <= range_um, base_opacity, 0.0)'''

            diff = vertices - sphere_center
            distances_sq = np.einsum('ij,ij->i', diff, diff)
            
            # Build RGBA
            rgba = np.zeros((n_points, 4), dtype=np.uint8)
            rgba[:, 0] = int(color[0] * 255)
            rgba[:, 1] = int(color[1] * 255)
            rgba[:, 2] = int(color[2] * 255)

            in_range = distances_sq <= range_um_sq
            rgba[in_range, 3] = int(base_opacity * 255)

            #rgba[:, 3] = (opacity * 255).astype(np.uint8)
            
            mesh.point_data['rgba'] = rgba
            mesh.Modified()
        
        print(f"  [TIMING] vertex opacity : {time.time() - t0:.3f}s")

        t0 = time.time()
        self._update_skeleton_host_contour(sphere_center)
        print(f"  [TIMING] _update_skeleton_host_contour: {time.time() - t0:.3f}s")

        t0 = time.time()
        self._update_morphology_camera_to_sphere(sphere_center, range_um)
        print(f"  [TIMING] _update_morphology_camera: {time.time() - t0:.3f}s")
    
        # Step 4: Update guide plane
        t0 = time.time()
        self._update_skeleton_host_guide_plane(sphere_center)
        print(f"  [TIMING] _update_skeleton_host_guide_plane: {time.time() - t0:.3f}s")

        t0 = time.time()
        app.morphology_view.plotter.render()
        print(f"  [TIMING] plotter.render() #2: {time.time() - t0:.3f}s")

        print(f"  [TIMING] _update_skeleton_host_morphology TOTAL: {time.time() - t_total:.3f}s")

    def _get_dynamic_sphere_radius(self, sphere_center):
        """
        Calculate dynamic sphere radius based on current CC size
        
        Returns radius in μm
        """
        app = self.app
        
        # Find nearest skeleton point
        dist, idx = app.data.skeleton_kdtree.query(sphere_center)
        label_id = int(app.data.skeleton_labels[idx])
        
        # Calculate z_L2
        z_L2 = int(sphere_center[2] / app.data.voxel_size_L2[0])
        
        # Find nearest CC
        label_str = str(label_id)
        
        if label_str not in app.data.cc_metadata['labels']:
            return app.morphology_view.morph_range_um  # fallback to default
        
        label_data = app.data.cc_metadata['labels'][label_str]
        
        if str(z_L2) not in label_data['layers']:
            return app.morphology_view.morph_range_um
        
        layer_data = label_data['layers'][str(z_L2)]
        
        # Find nearest CC by skeleton_marker
        min_dist = float('inf')
        best_cc_bbox = None
        
        for cc_info in layer_data['outer']:
            marker_info = cc_info.get('skeleton_marker', {})
            marker_pos = marker_info.get('position')
            
            if marker_pos is None:
                continue
            
            d = np.linalg.norm(np.array(marker_pos) - sphere_center)
            if d < min_dist:
                min_dist = d
                best_cc_bbox = cc_info['bbox']
        
        if best_cc_bbox is None:
            return app.morphology_view.morph_range_um
        
        # Calculate radius from bbox
        y_min, y_max, x_min, x_max = best_cc_bbox
        height_um = (y_max - y_min) * app.data.voxel_size_L2[1]
        width_um = (x_max - x_min) * app.data.voxel_size_L2[2]
        
        margin_um = 50.0
        radius = max(height_um, width_um) / 2 + margin_um
        
        # Clamp
        radius = max(100.0, min(radius, 500.0))
        
        print(f"  [RADIUS] CC size: {width_um:.0f}×{height_um:.0f} μm → radius: {radius:.0f} μm")
        
        return radius
    
    def _update_skeleton_host_contour(self, sphere_center):
        """
        Show current 2D CC black contour on morphology view
        Uses preprocessing cc_id and z_L0
        """
        import time
        t_start = time.time()

        app = self.app
        
        # Remove old contour
        if app.morphology_view.skeleton_host_contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
            except:
                pass
            app.morphology_view.skeleton_host_contour_actor = None
        
        if not app.show_plane_contour_chk.isChecked():
            return
        
        # Use tracked state (already level 0 values)
        z_L0 = app.state.tracked_state['z_L0']
        cc_id = app.state.tracked_state['cc_id']
        label_id = app.state.tracked_state['label_id']
        
        if cc_id < 0 or label_id is None:
            print(f"  [CONTOUR] Invalid state")
            return
        
        # Extract contour from level 0 labels
        t0 = time.time()
        labels_slice = app.data.lab_full[z_L0]
        if hasattr(labels_slice, 'compute'):
            labels_slice = labels_slice.compute()
        print(f"    [CONTOUR TIMING] .compute(): {time.time() - t0:.3f}s")

        t0 = time.time()
        label_mask = (labels_slice == label_id)
        
        if not label_mask.any():
            return
        
        labeled, _ = ndimage.label(label_mask)
        cc_mask = (labeled == (cc_id + 1))
        print(f"    [CONTOUR TIMING] ndimage.label: {time.time() - t0:.3f}s")

        if not cc_mask.any():
            return
        
        t0 = time.time()
        contours = measure.find_contours(cc_mask, 0.5)
        print(f"    [CONTOUR TIMING] find_contours: {time.time() - t0:.3f}s")
        
        if not contours:
            return
        
        # Calculate z_world from z_L0
        z_world = z_L0 * app.data.voxel_size_L0[0]

        t0 = time.time()
        for contour in contours:
            # L0 coords → world coords
            points_3d = np.zeros((len(contour), 3))
            points_3d[:, 0] = contour[:, 1] * app.data.voxel_size_L0[2]  # X world
            points_3d[:, 1] = contour[:, 0] * app.data.voxel_size_L0[1]  # Y world
            points_3d[:, 2] = z_world  # Z world
            
            # Close loop
            points_closed = np.vstack([points_3d, points_3d[0:1]])
            
            # Create polyline
            n_pts = len(points_closed)
            lines = []
            for i in range(n_pts - 1):
                lines.extend([2, i, i + 1])
            
            line_mesh = pv.PolyData(points_closed, lines=lines)
            
            app.morphology_view.skeleton_host_contour_actor = app.morphology_view.plotter.add_mesh(
                line_mesh,
                color='black',
                line_width=5,
                render_lines_as_tubes=True,
                opacity=1.0,
                name='skeleton_host_contour'
            )
            
            # Depth offset
            if hasattr(app.morphology_view.skeleton_host_contour_actor, 'GetMapper'):
                mapper = app.morphology_view.skeleton_host_contour_actor.GetMapper()
                if mapper:
                    mapper.SetResolveCoincidentTopologyToPolygonOffset()
                    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -2)
            
            print(f"  [CONTOUR] {len(points_3d)} pts @ z_world={z_world:.1f} μm")
            break  # Only first contour
        
        print(f"    [CONTOUR TIMING] add_mesh: {time.time() - t0:.3f}s")
        print(f"    [CONTOUR TIMING] TOTAL: {time.time() - t_start:.3f}s")
    # =========================================================================
    # MODE 2: SELECTION
    # =========================================================================
    
    def _on_draw_selection_toggle(self):
        """Toggle selection drawing mode"""
        app = self.app
        
        if app.draw_selection_btn.isChecked():
            self._start_selection_drawing()
        else:
            self._finish_selection_drawing()
        
    def _start_selection_drawing(self):
        """Start drawing selection on skeleton"""
        app = self.app
        
        print("\n[SELECTION] Drawing started")
        self.selection_drawing = True
        self.last_drawing_idx = None
        self._allow_jump = True
        
        # Check Ctrl key
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl_pressed = bool(modifiers & QtCore.Qt.ControlModifier)

        self.selection_accumulate_mode = ctrl_pressed
        
        if ctrl_pressed:
            # Ctrl+Draw → accumulate mode, keep previous selection
            print("  [ACCUMULATE MODE] Keeping previous selection")
        else:
            # Normal Draw → replace mode, clear previous selection
            self.selected_skeleton_indices = set()
            self._clear_selection_highlight()
        
        # Hide navigation sphere
        if app.skeleton_view.sphere_widget:
            try:
                app.skeleton_view.plotter.clear_sphere_widgets()
            except:
                pass
            app.skeleton_view.sphere_widget = None
        
        # Clear old highlight
        # Clear old highlight (only in replace mode)
        if not ctrl_pressed:
            self._clear_selection_highlight()
        #self._clear_selection_highlight()
        
        # Create brush sphere widget
        skeleton_radius_voxels = 3
        skeleton_radius_um = skeleton_radius_voxels * app.data.voxel_size_L2[0]
        brush_radius = skeleton_radius_um * 4.0
        
        initial_center = app.state.sphere_position if app.state.sphere_position is not None else app.data.skeleton_coords[0]
        
        self.selection_sphere_widget = app.skeleton_view.plotter.add_sphere_widget(
            callback=self._on_selection_brush_moved,
            center=initial_center,
            radius=brush_radius,
            color='gray',
            interaction_event='always',
            pass_widget=True,
        )
        self.selection_sphere_widget.ScaleOff()

        try:
            prop = self.selection_sphere_widget.GetSphereProperty()
            if prop:
                prop.SetOpacity(0.5)
        except:
            pass
    
        # Set semi-transparent
        if hasattr(self.selection_sphere_widget, 'GetRepresentation'):
            rep = self.selection_sphere_widget.GetRepresentation()
            if rep:
                prop = rep.GetProperty()
                if prop:
                    prop.SetOpacity(0.5)
        
        self._first_brush_move = True
        app.skeleton_view.plotter.render()
        print("  ✓ Brush sphere created. Drag along skeleton to select.")
    
    def _on_selection_brush_moved(self, center, widget):
        """Callback when selection brush is dragged"""
        app = self.app
        
        # Snap to nearest skeleton point
        dist, idx = app.data.skeleton_kdtree.query(center)
        nearest_pos = app.data.skeleton_coords[idx]

        # Positioning phase: allow jump to any position until connected move
        if getattr(self, '_allow_jump', False):
            if self.last_drawing_idx is not None and self.last_drawing_idx != idx:
                last_pos = app.data.skeleton_coords[self.last_drawing_idx]
                move_distance = np.linalg.norm(nearest_pos - last_pos)
                max_connected_distance = app.data.voxel_size_L2[0] * 20
                
                if move_distance <= max_connected_distance:
                    # Connected move = start drawing, disable jump
                    self._allow_jump = False
                    # Fall through to normal drawing logic
                else:
                    # Jump = still positioning, don't add to selection
                    widget.SetCenter(*nearest_pos)
                    self.last_drawing_idx = idx
                    return
            else:
                # First callback after widget created
                widget.SetCenter(*nearest_pos)
                self.last_drawing_idx = idx
                return

        # === Drawing phase: prevent jumping ===
        max_jump_distance = app.data.voxel_size_L2[0] * 20
        
        if self.last_drawing_idx is not None and self.last_drawing_idx != idx:
            last_pos = app.data.skeleton_coords[self.last_drawing_idx]
            jump_distance = np.linalg.norm(nearest_pos - last_pos)
            
            if jump_distance > max_jump_distance:
                old_pos = app.data.skeleton_coords[self.last_drawing_idx]
                widget.SetCenter(*old_pos)
                print(f"  [BRUSH] Jump rejected: {jump_distance:.1f} μm > {max_jump_distance:.1f} μm")
                return
            
        widget.SetCenter(*nearest_pos)
        
        # Fill path from last_drawing_idx to idx
        if self.last_drawing_idx is not None and self.last_drawing_idx != idx:
            valid = self._fill_selection_path(self.last_drawing_idx, idx)
            if not valid:
                return
        
        # Add current point
        self.selected_skeleton_indices.add(idx)
        self.last_drawing_idx = idx
        
        # Update visualization
        self._highlight_update_timer.start(50)
    '''
    def _on_selection_brush_moved(self, center, widget):
        """Callback when selection brush is dragged"""
        app = self.app
        
        # First move after Draw clicked: allow jump to any position
        if getattr(self, '_first_brush_move', False):
            self._first_brush_move = False
            widget.SetCenter(*nearest_pos)
            self.selected_skeleton_indices.add(idx)
            self.last_drawing_idx = idx
            self._update_selection_highlight()
            return
    
        # Snap to nearest skeleton point
        dist, idx = app.data.skeleton_kdtree.query(center)
        nearest_pos = app.data.skeleton_coords[idx]

        # === Prevent jumping: check distance from last position ===
        max_jump_distance = app.data.voxel_size_L2[0] * 20  # Max allowed jump (~20 voxels)
        
        if self.last_drawing_idx is not None and self.last_drawing_idx != idx:
            last_pos = app.data.skeleton_coords[self.last_drawing_idx]
            jump_distance = np.linalg.norm(nearest_pos - last_pos)
            
            if jump_distance > max_jump_distance:
                # Jump too far, reject move, keep original position
                old_pos = app.data.skeleton_coords[self.last_drawing_idx]
                widget.SetCenter(*old_pos)
                print(f"  [BRUSH] Jump rejected: {jump_distance:.1f} μm > {max_jump_distance:.1f} μm")
                return
            
        widget.SetCenter(*nearest_pos)
        
        # Fill path from last_drawing_idx to idx
        if self.last_drawing_idx is not None and self.last_drawing_idx != idx:
            valid = self._fill_selection_path(self.last_drawing_idx, idx)

            if not valid:
                # Path invalid, don't add this point
                return
        
        # Add current point
        self.selected_skeleton_indices.add(idx)
        self.last_drawing_idx = idx
        
        # Update visualization
        self._update_selection_highlight()
    '''
    def _fill_selection_path(self, from_idx, to_idx):
        """Use neighbors BFS to find shortest path on skeleton and fill"""
        app = self.app
        
        if from_idx == to_idx:
            return True
        
        # Check if neighbors data exists
        if not app.data.skeleton_neighbors:
            print("  [FILL] No neighbors data, skipping")
            return False
        
        # BFS for shortest path
        max_allowed_steps = 30

        visited = {from_idx}
        parent = {from_idx: None}
        queue = [from_idx]
        steps = 0
        found = False
        
        while queue and not found and steps < max_allowed_steps:
            current = queue.pop(0)
            steps += 1

            for neighbor in app.data.skeleton_neighbors[current]:
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                parent[neighbor] = current
                
                if neighbor == to_idx:
                    found = True
                    break
                
                queue.append(neighbor)
        
        if not found:
            # Path not found, skip
            print(f"  [FILL] Skipped: path not found within {max_allowed_steps} steps")
            return False
        
        # Trace back path, add to selection
        current = to_idx
        while current is not None and current != from_idx:
            self.selected_skeleton_indices.add(current)
            current = parent[current]
        return True    

    
    def _update_selection_highlight(self):
        """Update visual highlight for selected skeleton region"""
        app = self.app
        
        # Clear old
        self._clear_selection_highlight()
        
        if not self.selected_skeleton_indices:
            return
        
        # Create highlight mesh - use thick spheres to mark selected points
        selected_coords = app.data.skeleton_coords[list(self.selected_skeleton_indices)]
        
        # Use PolyData points + glyph
        points = pv.PolyData(selected_coords)
        
        skeleton_radius_voxels = 3
        skeleton_radius_um = skeleton_radius_voxels * app.data.voxel_size_L2[0]
        highlight_radius = skeleton_radius_um * 3.0
        
        sphere_glyph = pv.Sphere(radius=highlight_radius)
        glyphs = points.glyph(geom=sphere_glyph, scale=False)
        
        self.selection_highlight_actor = app.skeleton_view.plotter.add_mesh(
            glyphs,
            color='gray',
            opacity=0.1,
            name='selection_highlight'
        )
        
        app.skeleton_view.plotter.render()
    
    def _clear_selection_highlight(self):
        """Clear selection highlight actor"""
        app = self.app
        
        if hasattr(self, 'selection_highlight_actor') and self.selection_highlight_actor:
            try:
                app.skeleton_view.plotter.remove_actor(self.selection_highlight_actor)
            except:
                pass
            self.selection_highlight_actor = None
    
    def _finish_selection_drawing(self):
        """Finish drawing and apply selection"""
        app = self.app
        
        print(f"\n[SELECTION] Drawing finished. {len(self.selected_skeleton_indices)} points selected.")
        
        self.selection_drawing = False
        
        # Remove brush sphere
        if hasattr(self, 'selection_sphere_widget') and self.selection_sphere_widget:
            try:
                app.skeleton_view.plotter.clear_sphere_widgets()
            except:
                pass
            self.selection_sphere_widget = None
        
        app.show_plane_contour_chk.setChecked(False)
        app.show_plane_chk.setChecked(False)

        # If has selection, move navigation sphere to selection center
        if self.selected_skeleton_indices:
            # Find selection center
            selected_coords = app.data.skeleton_coords[list(self.selected_skeleton_indices)]
            center_pos = np.mean(selected_coords, axis=0)
            _, center_idx = app.data.skeleton_kdtree.query(center_pos)
            
            # Ensure center is in selection
            if center_idx not in self.selected_skeleton_indices:
                center_idx = list(self.selected_skeleton_indices)[0]
            
            app.state.sphere_position = app.data.skeleton_coords[center_idx].copy()
            
            # Rebuild navigation sphere (with restricted callback)
            self._rebuild_navigation_sphere_restricted()
            
            # Trigger update
            dist, idx = app.data.skeleton_kdtree.query(app.state.sphere_position)
            label_id = int(app.data.skeleton_labels[idx])
            cc_id = int(app.data.skeleton_cc_ids[idx])
            z_L0 = int(app.data.skeleton_z_L0[idx])
            z_L2 = int(app.data.skeleton_coords_voxel[idx, 0])

            if label_id > 0 and cc_id >= 0:
                label_name = app.data.label_names[label_id - 1]
                app.state.tracked_state['z_L0'] = z_L0
                app.state.tracked_state['z_L2'] = z_L2
                app.state.tracked_state['label_id'] = label_id
                app.state.tracked_state['label_name'] = label_name
                app.state.tracked_state['cc_id'] = cc_id

            # Only update Napari slice, not boundary
            self._update_napari_from_skeleton(z_L0)
            
            print(f"  ✓ Navigation sphere moved to selection center")
        else:
            # No selection, restore original sphere
            if app.skeleton_view.sphere_widget:
                app.skeleton_view.sphere_widget.On()
                try:
                    app.skeleton_view.sphere_widget.GetSphereProperty().SetOpacity(0.5)
                except:
                    pass

        self._update_selection_morphology()
        app.skeleton_view.plotter.render()

        app.show_plane_contour_chk.setChecked(True)

        # Ensure correct boundary is shown after all setup
        if app.state.tracked_state['label_id'] is not None:
            self._update_boundary_from_napari_position()

    def _update_selection_morphology(self):
        """
        Generate morphology view for selected skeleton region
        Called once after drawing finishes - static display
        """
        import time
        t_total = time.time()

        app = self.app
        
        if not self.selected_skeleton_indices:
            print("[SELECTION MORPH] No selection")
            return
        
        print(f"\n{'='*60}")
        print(f"[SELECTION MORPH] Building from {len(self.selected_skeleton_indices)} skeleton points")
        print(f"{'='*60}")
        
        # === Step 1: Convert selection to path format (BATCH READ) ===
        t0 = time.time()
        
        # First pass: collect unique (label_name, z) pairs
        from collections import defaultdict
        
        slices_needed = set()  # {(label_name, z_L2), ...}
        points_data = []  # [(z_L2, y_L2, x_L2, label_id, label_name), ...]
        
        for idx in self.selected_skeleton_indices:
            z_L2 = int(app.data.skeleton_coords_voxel[idx, 0])
            y_L2 = int(app.data.skeleton_coords_voxel[idx, 1])
            x_L2 = int(app.data.skeleton_coords_voxel[idx, 2])
            label_id = int(app.data.skeleton_labels[idx])
            
            if label_id == 0:
                continue
            
            label_name = app.data.label_names[label_id - 1]
            slices_needed.add((label_name, z_L2))
            points_data.append((z_L2, y_L2, x_L2, label_id, label_name))
        
        # Group by label for batch read
        label_z_map = defaultdict(set)  # {label_name: {z1, z2, ...}}
        for label_name, z_L2 in slices_needed:
            label_z_map[label_name].add(z_L2)
        
        # Batch read and CC label
        t1 = time.time()
        labeled_cache = {}  # {(label_id, z_L2): labeled_array}
        
        for label_name, z_set in label_z_map.items():
            z_list = sorted(z_set)
            z_min, z_max = min(z_list), max(z_list) + 1
            
            # Batch read entire Z range
            block = app.data.outer_masks_L2[label_name][z_min:z_max].compute()
            
            label_id = app.data.label_names.index(label_name) + 1
            
            for z in z_list:
                z_rel = z - z_min
                slice_mask = block[z_rel] > 0
                if slice_mask.any():
                    labeled, _ = ndimage.label(slice_mask)
                    labeled_cache[(label_id, z)] = labeled
                else:
                    labeled_cache[(label_id, z)] = None
        
        print(f"  [TIMING] Step 1 batch read + label: {time.time() - t1:.3f}s")
        
        # Second pass: build path_3d
        path_3d = {}
        seen_ccs = set()
        
        for z_L2, y_L2, x_L2, label_id, label_name in points_data:
            labeled = labeled_cache.get((label_id, z_L2))
            if labeled is None:
                continue
            
            cc_id_1indexed = labeled[y_L2, x_L2]
            if cc_id_1indexed == 0:
                continue
            cc_id = cc_id_1indexed - 1
            
            key = (z_L2, label_id, cc_id)
            if key in seen_ccs:
                continue
            seen_ccs.add(key)
            
            if z_L2 not in path_3d:
                path_3d[z_L2] = []
            path_3d[z_L2].append((label_id, cc_id))
        
        print(f"  [TIMING] Step 1 TOTAL: {time.time() - t0:.3f}s")
        
        if not path_3d:
            print("  ✗ No valid CCs found")
            return
        
        print(f"  Found {len(seen_ccs)} unique CCs across {len(path_3d)} Z slices")
        print(f"  Z range: [{min(path_3d.keys())}:{max(path_3d.keys())}]")
        
        # === Step 2: Calculate union bbox ===
        t0 = time.time()
        xy_bbox = app.cc_tracker._calculate_union_xy_bbox_from_path(path_3d)
        print(f"  [TIMING] Step 2 (bbox): {time.time() - t0:.3f}s")

        if xy_bbox is None:
            print("  ✗ Failed to calculate bbox")
            return
        
        # === Step 3: Extract masks ===
        t0 = time.time()
        label_masks = app.cc_tracker._extract_path_ccs_in_bbox(path_3d, xy_bbox)
        print(f"  [TIMING] Step 3 (extract masks): {time.time() - t0:.3f}s")

        # === Step 4: Clear old morphology actors ===
        _clear_morphology_actors(app)
        
        # === Step 5: Create surfaces per label ===
        t0 = time.time()
        z_min = min(path_3d.keys())
        
        for label_name, masks in label_masks.items():
            if np.sum(masks['outer']) == 0:
                continue
            
            label_color = _hex_to_rgb(app.data.label_colors[label_name])
            
            # Outer mesh
            outer_mesh = app.mesh_builder._create_surface(masks['outer'], xy_bbox, z_min)
            
            if outer_mesh and outer_mesh.n_points > 0:
                app.morphology_view.outer_volume_actors[label_name] = app.morphology_view.plotter.add_mesh(
                    outer_mesh,
                    color=label_color,
                    opacity=app.morphology_view.outer_opacity,
                    smooth_shading=True,
                    show_edges=False,
                    name=f'sel_outer_{label_name}'
                )
                print(f"    ✓ {label_name} outer: {outer_mesh.n_points:,} pts")
            
            # Inner mesh (only if has_inner_mask)
            if app.data.has_inner_mask and np.sum(masks['inner']) > 0:
                inner_mesh = app.mesh_builder._create_surface(masks['inner'], xy_bbox, z_min)
                
                if inner_mesh and inner_mesh.n_points > 0:
                    app.morphology_view.inner_mesh_actors[label_name] = app.morphology_view.plotter.add_mesh(
                        inner_mesh,
                        color=label_color,
                        opacity=app.morphology_view.inner_opacity,
                        smooth_shading=True,
                        show_edges=False,
                        name=f'sel_inner_{label_name}'
                    )
                    print(f"    ✓ {label_name} inner: {inner_mesh.n_points:,} pts")
        
        print(f"  [TIMING] Step 5 (create surfaces): {time.time() - t0:.3f}s")

        # === Step 6: Zoom to selection ===
        z_center = (min(path_3d.keys()) + max(path_3d.keys())) // 2
        _zoom_to_path_bbox(app, xy_bbox, z_center)
        
        # === Step 7: Guide plane (at center Z) ===
        self._update_guide_plane_for_path(z_center, xy_bbox)
        
        app.morphology_view.plotter.render()
        
        print(f"\n[SELECTION MORPH] Complete!")
        print(f"{'='*60}\n")
    
    def _propagate_selection_path(self, seed_path):
        """
        Fill Z gaps - only fill missing Z layers from overlap table
        No BFS expansion, just fill gaps
        """
        app = self.app
        
        if not seed_path:
            return seed_path
        
        z_min = min(seed_path.keys())
        z_max = max(seed_path.keys())
        
        print(f"\n[PROPAGATE] Filling gaps...")
        print(f"  Seed: {len(seed_path)} Z slices, range=[{z_min}:{z_max}]")
        
        full_path = {z: list(ccs) for z, ccs in seed_path.items()}
        
        # Find gaps
        all_z = set(range(z_min, z_max + 1))
        existing_z = set(seed_path.keys())
        missing_z = sorted(all_z - existing_z)
        
        if not missing_z:
            print(f"  No gaps to fill")
            return full_path
        
        print(f"  Missing: {len(missing_z)} Z slices")
        
        # Fill each gap (lookup from previous layer)
        filled = 0
        for z in missing_z:
            ccs_to_add = []
            
            # From z-1 lookup overlaps_next
            if z - 1 in full_path:
                for label_id, cc_id in full_path[z - 1]:
                    next_cc = app.cc_tracker._get_best_overlap(z - 1, label_id, cc_id, direction='next')
                    if next_cc and next_cc not in ccs_to_add:
                        ccs_to_add.append(next_cc)
            
            # If z-1 has no data, from z+1 lookup overlaps_prev
            if not ccs_to_add and z + 1 in full_path:
                for label_id, cc_id in full_path[z + 1]:
                    prev_cc = app.cc_tracker._get_best_overlap(z + 1, label_id, cc_id, direction='prev')
                    if prev_cc and prev_cc not in ccs_to_add:
                        ccs_to_add.append(prev_cc)
            
            if ccs_to_add:
                full_path[z] = ccs_to_add
                filled += 1
        
        total_ccs = sum(len(ccs) for ccs in full_path.values())
        print(f"  Filled: {filled} Z slices")
        print(f"  After: {len(full_path)} Z slices, {total_ccs} CCs")
        
        return full_path

    def _on_clear_selection(self):
        """Clear selection and restore full navigation"""
        app = self.app
        
        print("\n[SELECTION] Cleared")
        
        self._selection_sphere_cleaned = False 
        self.selected_skeleton_indices = set()
        self.last_drawing_idx = None
        self.selection_drawing = False
        app.draw_selection_btn.setChecked(False)
        
        self._clear_selection_highlight()
        
        # Don't clear sphere_widget! Just swap restricted KDTree to full one
        # This way existing sphere callback automatically becomes unrestricted
        self._selected_kdtree = app.data.skeleton_kdtree
        self._selected_list = list(range(len(app.data.skeleton_coords)))
        
        _clear_morphology_actors(app)
        
        # Clear Navigation mode contour
        if app.morphology_view.skeleton_host_contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
            except:
                pass
            app.morphology_view.skeleton_host_contour_actor = None
        
        # Clear Navigation mode guide plane
        for attr in ['skeleton_host_plane_actor', 'skeleton_host_scale_actor', 'skeleton_host_scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)
        
        # Clear Image Host plane actors
        for attr in ['plane_actor', 'scale_line_actor', 'scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)
        
        # Clear 2D boundary layer
        if app.napari_view.boundary_layer is not None:
            app.napari_view.boundary_layer.data = np.zeros((1, 1), dtype=np.uint8)
        
        # Rebuild unrestricted navigation sphere
        self._rebuild_navigation_sphere_unrestricted()

        app.skeleton_view.plotter.render()
        app.morphology_view.plotter.render()

    def _exit_selection_mode(self):
        """Clean up when exiting Selection mode"""
        app = self.app
        
        self.selection_drawing = False
        app.draw_selection_btn.setChecked(False)
        
        # Clear highlight
        self._clear_selection_highlight()
        
        # Clear selection (restore free navigation)
        self.selected_skeleton_indices = set()
        
        # Remove brush sphere
        if hasattr(self, 'selection_sphere_widget') and self.selection_sphere_widget:
            try:
                app.skeleton_view.plotter.clear_sphere_widgets()
            except:
                pass
            self.selection_sphere_widget = None
        
        # Rebuild unrestricted navigation sphere
        self._rebuild_navigation_sphere_unrestricted()

    # =========================================================================
    # NAPARI OPS - 2D boundary updates (Skeleton Host specific)
    # =========================================================================
    
    def _update_2d_view_direct(self):
        """
        Direct update using preprocessing cc_id to show 2D boundary
        No centroid matching needed, direct lookup
        """
        app = self.app
        
        label_id = app.state.tracked_state['label_id']
        label_name = app.state.tracked_state['label_name']
        z_L0 = app.state.tracked_state['z_L0']      # This is z_L0
        cc_id = app.state.tracked_state['cc_id']    # This is level 0 CC ID
        
        print(f"\n[2D DIRECT] z_L0={z_L0}, {label_name} CC#{cc_id}")
        
        if cc_id < 0:
            print(f"  ⚠ Invalid CC ID, clearing boundary")
            app.napari_view.boundary_layer.data = []
            return
        
        # Extract from level 0 labels
        # Extract from level 0 labels (with cache)
        labels_slice = _slice_data_cache.get_label_slice(app.data.lab_full, z_L0)

        # Create binary mask for this label
        label_mask = (labels_slice == label_id)

        if not label_mask.any():
            print(f"  ⚠ Label {label_id} not found at z={z_L0}")
            app.napari_view.boundary_layer.data = []
            return

        # Label CCs (with cache)
        labeled, num_cc = _cc_label_cache.get(label_id + 2000, z_L0, label_mask)
        
        # Directly use preprocessing cc_id (0-indexed → 1-indexed for labeled)
        cc_mask = (labeled == (cc_id + 1))
        
        if not cc_mask.any():
            print(f"  ⚠ CC#{cc_id} not found at z={z_L0} (num_cc={num_cc})")
            # Fallback: show largest CC
            if num_cc > 0:
                cc_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_cc + 1)]
                largest_cc = max(cc_sizes, key=lambda x: x[1])[0]
                cc_mask = (labeled == largest_cc)
                print(f"  → Fallback to largest CC: {largest_cc}")
            else:
                app.napari_view.boundary_layer.data = []
                return
        
        # Extract contour
        contours = measure.find_contours(cc_mask, 0.5)
        
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
            print(f"  ✓ Boundary: {len(contours)} contours")
        else:
            print(f"  ⚠ No contours found")
        
        app.napari_view.boundary_layer.refresh()

        if app.state.auto_center_enabled:
            QtCore.QTimer.singleShot(50, lambda: _center_on_component(app, from_button=False))

    def _update_boundary_from_napari_position(self):
        """
        Selection mode specific: Find CC from Napari current slice + sphere position
        Ensures boundary matches Napari display
        """
        app = self.app
        
        if app.state.sphere_position is None:
            app.napari_view.boundary_layer.data = []
            return
        
        # 1. Use Napari current z slice (this is correct)
        z_L0 = int(app.napari_view.viewer.dims.current_step[0])
        
        # 2. Convert sphere world coords to L0 pixel coords
        x_L0 = int(app.state.sphere_position[0] / app.data.voxel_size_L0[2])
        y_L0 = int(app.state.sphere_position[1] / app.data.voxel_size_L0[1])
        
        print(f"\n[2D FROM NAPARI] z={z_L0}, sphere pixel=({y_L0}, {x_L0})")
        
        # 3. Read label from Napari current slice
        label_id = app.state.tracked_state['label_id']
        if label_id is None:
            app.napari_view.boundary_layer.data = []
            return
        
        '''
        labels_slice = app.data.lab_full[z_L0]
        if hasattr(labels_slice, 'compute'):
            labels_slice = labels_slice.compute()'''
        labels_slice = _slice_data_cache.get_label_slice(app.data.lab_full, z_L0)
        
        label_mask = (labels_slice == label_id)
        
        if not label_mask.any():
            print(f"  ⚠ Label {label_id} not found at z={z_L0}")
            app.napari_view.boundary_layer.data = []
            return
        
        # 4. Label CCs and find CC containing sphere position
        labeled, num_cc = ndimage.label(label_mask)
        
        # Clamp to bounds
        y_L0 = max(0, min(y_L0, labeled.shape[0] - 1))
        x_L0 = max(0, min(x_L0, labeled.shape[1] - 1))
        
        target_cc_label = labeled[y_L0, x_L0]
        
        if target_cc_label == 0:
            # Sphere position not on label, search nearby
            print(f"  Sphere position not on label, searching nearby...")
            search_radius = 20
            found = False
            for r in range(1, search_radius + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dy) != r and abs(dx) != r:
                            continue
                        ny, nx = y_L0 + dy, x_L0 + dx
                        if 0 <= ny < labeled.shape[0] and 0 <= nx < labeled.shape[1]:
                            if labeled[ny, nx] > 0:
                                target_cc_label = labeled[ny, nx]
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                print(f"  ⚠ No CC found near sphere position")
                app.napari_view.boundary_layer.data = []
                return
        
        cc_mask = (labeled == target_cc_label)
        print(f"  ✓ Found CC label={target_cc_label} at ({y_L0}, {x_L0})")
        
        # 5. Update tracked_state z_L0 (for later lookup when moving)
        # Note: cc_id here is level 0 numbering, may differ from preprocessing
        # But that's OK, movement will use _on_sphere_position_changed() to re-lookup
        app.state.tracked_state['z_L0'] = z_L0
        
        # 6. Extract contour
        contours = measure.find_contours(cc_mask, 0.5)
        
        app.napari_view.boundary_layer.data = []
        
        if contours:
            for contour in contours:
                polygon = np.column_stack([
                    np.full(len(contour), z_L0),
                    contour[:, 0],
                    contour[:, 1]
                ])
                app.napari_view.boundary_layer.add_polygons(polygon)
            print(f"  ✓ Boundary: {len(contours)} contours")
        
        app.napari_view.boundary_layer.refresh()

    def _update_napari_from_skeleton(self, z_L0):
        """
        Update Napari slice from Skeleton Host mode
        
        Temporarily disconnects callback to avoid feedback loop
        """
        app = self.app
        
        # Clamp to valid range
        max_z = app.data.lab_full.shape[0] - 1
        z_L0 = max(0, min(z_L0, max_z))
        
        # Temporarily disconnect to avoid feedback loop
        try:
            app.napari_view.viewer.dims.events.current_step.disconnect(app.image_host._on_slice_changed)
        except:
            pass
        
        # Set slice
        app.napari_view.viewer.dims.set_point(0, z_L0 * app.data.voxel_size_L0[0])
        
        # Reconnect
        app.napari_view.viewer.dims.events.current_step.connect(app.image_host._on_slice_changed)
        
        print(f"  ✓ Napari slice updated to {z_L0}")
    
    # =========================================================================
    # SKELETON OPS - Sphere widget management
    # =========================================================================
    
    def _rebuild_navigation_sphere_unrestricted(self):
        """Rebuild navigation sphere with no restrictions"""
        app = self.app
        
        try:
            app.skeleton_view.plotter.clear_sphere_widgets()
        except:
            pass
        
        skeleton_radius_voxels = 3
        skeleton_radius_um = skeleton_radius_voxels * app.data.voxel_size_L2[0]
        sphere_radius = skeleton_radius_um * 3.0
        
        initial_center = app.state.sphere_position if app.state.sphere_position is not None else app.data.skeleton_coords[0]
        
        # Store reference to self for nested function
        skeleton_host = self
        
        def on_sphere_drag(center, widget):
            if getattr(skeleton_host, '_mode_changing', False):
                return
            dist, idx = app.data.skeleton_kdtree.query(center)
            nearest_pos = app.data.skeleton_coords[idx]
            widget.SetCenter(*nearest_pos)
            app.state.sphere_position = nearest_pos.copy()
            
            if app.state.host_mode == 'skeleton':
                skeleton_host._on_sphere_drag_lightweight(nearest_pos)
        
        app.skeleton_view.sphere_widget = app.skeleton_view.plotter.add_sphere_widget(
            callback=on_sphere_drag,
            center=initial_center,
            radius=sphere_radius,
            color='black',
            interaction_event='always',
            pass_widget=True,
        )
        self.selection_sphere_widget.ScaleOff()

        try:
            prop = app.skeleton_view.sphere_widget.GetSphereProperty()
            if prop:
                prop.SetOpacity(0.5)
        except:
            pass
        
        app.skeleton_view.sphere_widget.On()
    
    def _rebuild_navigation_sphere_restricted(self):
        """Rebuild navigation sphere that's restricted to selected region"""
        app = self.app
        
        # Clear all sphere widgets first
        try:
            app.skeleton_view.plotter.clear_sphere_widgets()
        except:
            pass

        app.skeleton_view.sphere_widget = None
        self.selection_sphere_widget = None
        
        # Force render to ensure clear takes effect
        app.skeleton_view.plotter.render()

        skeleton_radius_voxels = 3
        skeleton_radius_um = skeleton_radius_voxels * app.data.voxel_size_L2[0]
        sphere_radius = skeleton_radius_um * 3.0

        # Build KDTree for selected region
        selected_list = list(self.selected_skeleton_indices)
        selected_coords = app.data.skeleton_coords[selected_list]
        self._selected_kdtree = KDTree(selected_coords)
        self._selected_list = selected_list
        
        # Store reference to self for nested function
        skeleton_host = self
        
        def on_restricted_sphere_drag(center, widget):
            # Only find nearest point in selected region
            # This way the sphere can only move within the highlighted area
            dist, local_idx = skeleton_host._selected_kdtree.query(center)
            global_idx = skeleton_host._selected_list[local_idx]
            nearest_pos = app.data.skeleton_coords[global_idx]
            
            widget.SetCenter(*nearest_pos)
            app.state.sphere_position = nearest_pos.copy()
            
            if app.state.host_mode == 'skeleton':
                skeleton_host._on_sphere_drag_lightweight(nearest_pos)
        
        app.skeleton_view.sphere_widget = app.skeleton_view.plotter.add_sphere_widget(
            callback=on_restricted_sphere_drag,
            center=app.state.sphere_position,
            radius=sphere_radius,
            color='black',
            interaction_event='always',
            pass_widget=True,
        )
        self.selection_sphere_widget.ScaleOff()

        try:
            prop = app.skeleton_view.sphere_widget.GetSphereProperty()
            if prop:
                prop.SetOpacity(0.5)
        except:
            pass

    # =========================================================================
    # MORPHOLOGY OPS - 3D rendering (Skeleton Host specific)
    # =========================================================================
    
    def _update_skeleton_host_guide_plane(self, sphere_center):
        """
        Use level 0 CC mask directly to calculate bbox, no cc_metadata lookup
        """
        app = self.app
        
        # Remove old (including Selection mode created plane_actor)
        for attr in ['skeleton_host_plane_actor', 'skeleton_host_scale_actor', 'skeleton_host_scale_text_actor',
                    'plane_actor', 'scale_line_actor', 'scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)
        
        if not app.show_plane_chk.isChecked():
            return
        
        # Use tracked_state (already level 0)
        z_L0 = app.state.tracked_state['z_L0']
        cc_id = app.state.tracked_state['cc_id']
        label_id = app.state.tracked_state['label_id']
        
        if label_id is None or cc_id < 0:
            return
        
        # Get CC mask from level 0 labels
        labels_slice = _slice_data_cache.get_label_slice(app.data.lab_full, z_L0)
        '''
        labels_slice = app.data.lab_full[z_L0]
        if hasattr(labels_slice, 'compute'):
            labels_slice = labels_slice.compute()'''
        
        label_mask = (labels_slice == label_id)
        if not label_mask.any():
            return
        
        labeled, _ = ndimage.label(label_mask)
        cc_mask = (labeled == (cc_id + 1))
        
        if not cc_mask.any():
            return
        
        # Directly calculate bbox from level 0 CC mask
        y_coords, x_coords = np.where(cc_mask)
        y_min_L0, y_max_L0 = y_coords.min(), y_coords.max()
        x_min_L0, x_max_L0 = x_coords.min(), x_coords.max()
        
        # Level 0 coords → world coords
        z_world = z_L0 * app.data.voxel_size_L0[0]
        center_x = (x_min_L0 + x_max_L0) / 2 * app.data.voxel_size_L0[2]
        center_y = (y_min_L0 + y_max_L0) / 2 * app.data.voxel_size_L0[1]
        plane_width = (x_max_L0 - x_min_L0) * app.data.voxel_size_L0[2]
        plane_height = (y_max_L0 - y_min_L0) * app.data.voxel_size_L0[1]
        
        # Add margin
        margin = 50  # μm
        plane_width += margin * 2
        plane_height += margin * 2
        
        # Create plane
        plane = pv.Plane(
            center=(center_x, center_y, z_world),
            direction=(0, 0, 1),
            i_size=plane_width,
            j_size=plane_height
        )
        
        app.morphology_view.skeleton_host_plane_actor = app.morphology_view.plotter.add_mesh(
            plane,
            color='black',
            opacity=0.3,
            name='skeleton_host_plane'
        )
        
        # Scale bar
        line_start = [center_x - plane_width/2, center_y + plane_height/2 + 20, z_world]
        line_end = [center_x + plane_width/2, center_y + plane_height/2 + 20, z_world]
        
        line = pv.Line(line_start, line_end)
        app.morphology_view.skeleton_host_scale_actor = app.morphology_view.plotter.add_mesh(
            line,
            color='black',
            line_width=3,
            render_lines_as_tubes=True,
            name='skeleton_host_scale'
        )
        
        # Scale text
        scale_text = f"{plane_width:.0f} μm"
        app.morphology_view.skeleton_host_scale_text_actor = app.morphology_view.plotter.add_text(
            scale_text,
            position='upper_right',
            font_size=10,
            color='black',
            name='skeleton_host_scale_text'
        )
        
        print(f"  [PLANE] {plane_width:.0f} × {plane_height:.0f} μm @ z={z_world:.1f}")

    def _update_guide_plane_for_path(self, z_L2, xy_bbox):
        """Add guide plane for path - always create, control with visibility"""
        app = self.app
        
        z_world = z_L2 * app.data.voxel_size_L2[0]
        
        center_x = (xy_bbox['x_min'] + xy_bbox['x_max']) / 2 * app.data.voxel_size_L2[2]
        center_y = (xy_bbox['y_min'] + xy_bbox['y_max']) / 2 * app.data.voxel_size_L2[1]
        plane_width = (xy_bbox['x_max'] - xy_bbox['x_min']) * app.data.voxel_size_L2[2]
        plane_height = (xy_bbox['y_max'] - xy_bbox['y_min']) * app.data.voxel_size_L2[1]
        
        plane = pv.Plane(
            center=(center_x, center_y, z_world),
            direction=(0, 0, 1),
            i_size=plane_width,
            j_size=plane_height
        )
        
        # Set visibility based on checkbox
        initial_visibility = app.show_plane_chk.isChecked()
        
        app.morphology_view.plane_actor = app.morphology_view.plotter.add_mesh(
            plane,
            color='black',
            opacity=0.3,
            name='guide_plane'
        )
        
        # Apply visibility
        app.morphology_view.plane_actor.visibility = initial_visibility
        
        # === Scale bar ===
        line_start = [center_x - plane_width/2, center_y + plane_height/2, z_world + 1]
        line_end = [center_x + plane_width/2, center_y + plane_height/2, z_world + 1]
        
        line = pv.Line(line_start, line_end)
        
        app.morphology_view.scale_line_actor = app.morphology_view.plotter.add_mesh(
            line,
            color='black',
            line_width=3,
            name='scale_line',
            render=False  # Don't render yet
        )
        
        # Apply visibility to scale line
        if app.morphology_view.scale_line_actor:
            app.morphology_view.scale_line_actor.visibility = initial_visibility
        
        # === Scale text ===
        scale_text = f"{plane_width:.0f} μm"
        app.morphology_view.scale_text_actor = app.morphology_view.plotter.add_text(
            scale_text,
            position='upper_right',
            font_size=10,
            color='black',
            name='scale_text'
        )
        
        # Text actor visibility
        if hasattr(app.morphology_view.scale_text_actor, 'SetVisibility'):
            app.morphology_view.scale_text_actor.SetVisibility(initial_visibility)
        
        # Only add contour if checkbox is checked
        if app.show_plane_contour_chk.isChecked():
            print(f"  [PLANE] Adding contour (checkbox is checked)")
            self._add_cc_contour_on_plane(z_L2, xy_bbox)
        else:
            print(f"  [PLANE] Skipping contour (checkbox is unchecked)")
        
        status = "visible" if initial_visibility else "hidden"
        print(f"  ✓ Guide plane + scale bar added ({status})")

    def _add_cc_contour_on_plane(self, z_L2, xy_bbox):
        """
        Add current CC contour on guide plane (in BLACK)
        
        Properly handles coordinate transformation relative to bbox
        """
        app = self.app
        
        # Check if contour should be visible
        if not app.show_plane_contour_chk.isChecked():
            print(f"    [CONTOUR] Skipped (checkbox disabled)")
            return

        label_name = app.state.tracked_state['label_name']
        cc_id = app.state.tracked_state['cc_id']
        
        if label_name not in app.data.outer_masks_L2:
            return
        
        outer_mask = app.data.outer_masks_L2[label_name]
        
        # Check bounds
        if z_L2 >= outer_mask.shape[0]:
            return
        
        # Load FULL slice @ L2
        slice_mask = outer_mask[z_L2].compute()
        
        # Label CCs in FULL slice
        labeled, _ = ndimage.label(slice_mask > 0)
        
        # Get current CC mask in FULL coordinates
        cc_mask = (labeled == (cc_id + 1))
        
        if not cc_mask.any():
            print(f"    [CONTOUR] CC#{cc_id} not found at z={z_L2}")
            return
        
        # Extract contours (coordinates are in FULL image space)
        contours = measure.find_contours(cc_mask, 0.5)
        
        if not contours:
            print(f"    [CONTOUR] No contours found")
            return
        
        z_world = z_L2 * app.data.voxel_size_L2[0]
        
        # Convert from full image coords to world coords
        # contour[:, 0] = Y in full image @ L2
        # contour[:, 1] = X in full image @ L2
        
        for contour in contours:
            # Convert to world coordinates (μm)
            points_3d = np.zeros((len(contour), 3))
            points_3d[:, 0] = contour[:, 1] * app.data.voxel_size_L2[2]  # X world
            points_3d[:, 1] = contour[:, 0] * app.data.voxel_size_L2[1]  # Y world
            points_3d[:, 2] = z_world  # Z world
            
            # Debug: Check if contour is within bbox
            x_min_world = xy_bbox['x_min'] * app.data.voxel_size_L2[2]
            x_max_world = xy_bbox['x_max'] * app.data.voxel_size_L2[2]
            y_min_world = xy_bbox['y_min'] * app.data.voxel_size_L2[1]
            y_max_world = xy_bbox['y_max'] * app.data.voxel_size_L2[1]
            
            contour_x_range = (points_3d[:, 0].min(), points_3d[:, 0].max())
            contour_y_range = (points_3d[:, 1].min(), points_3d[:, 1].max())
            
            print(f"    [CONTOUR] Bbox X: [{x_min_world:.1f}, {x_max_world:.1f}] μm")
            print(f"    [CONTOUR] Bbox Y: [{y_min_world:.1f}, {y_max_world:.1f}] μm")
            print(f"    [CONTOUR] Contour X: [{contour_x_range[0]:.1f}, {contour_x_range[1]:.1f}] μm")
            print(f"    [CONTOUR] Contour Y: [{contour_y_range[0]:.1f}, {contour_y_range[1]:.1f}] μm")
            
            # Check if contour overlaps with bbox
            if (contour_x_range[1] < x_min_world or contour_x_range[0] > x_max_world or
                contour_y_range[1] < y_min_world or contour_y_range[0] > y_max_world):
                print(f"    ⚠️  [CONTOUR] Outside bbox, skipping")
                continue
            
            # Close the loop
            points_closed = np.vstack([points_3d, points_3d[0:1]])
            
            # Create polyline
            lines = []
            for i in range(len(points_closed) - 1):
                lines.extend([2, i, i + 1])
            
            line_mesh = pv.PolyData(points_closed, lines=lines)
            
            # BLACK contour on plane
            app.morphology_view.contour_actor = app.morphology_view.plotter.add_mesh(
                line_mesh,
                color='black',
                line_width=4,
                render_lines_as_tubes=True,
                opacity=1.0,
                name='plane_contour'
            )
            
            print(f"    ✓ CC contour added on plane: {len(points_3d)} points")
            break  # Only first contour

    def _update_morphology_camera_to_sphere(self, sphere_center, range_um):
        """
        Update morphology view camera to center on sphere region
        Preserves current view direction (rotation angle)
        """
        app = self.app
        
        try:
            cam = app.morphology_view.plotter.camera
            
            # Get current camera view direction
            current_pos = np.array(cam.position)
            current_focal = np.array(cam.focal_point)
            view_vector = current_pos - current_focal
            view_distance = np.linalg.norm(view_vector)
            view_direction = view_vector / (view_distance + 1e-10)
            
            # Calculate appropriate camera distance based on range_um
            # Make visible region just fill the window (with some margin)
            camera_distance = range_um * 4.0  # Adjustable multiplier
            
            # New focal point = sphere center
            new_focal = sphere_center
            
            # New camera position = focal point + view_direction * distance
            new_pos = new_focal + view_direction * camera_distance
            
            # Set camera
            cam.focal_point = tuple(new_focal)
            cam.position = tuple(new_pos)
            
            # Update clipping range
            app.morphology_view.plotter.reset_camera_clipping_range()
            
        except Exception as e:
            print(f"  [CAMERA] Update failed: {e}")
    
