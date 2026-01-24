"""
UI Controls

Contains:
1. UI callbacks (toggles, sliders, etc.)
2. Shared operations used by both modes
"""

from qtpy import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor 
import numpy as np

# =============================================================================
# UI CALLBACKS
# =============================================================================
    
def _on_host_mode_changed(app, button):
    """Handle host mode toggle"""
    if button == app.image_host_radio:
        app.image_host._switch_to_image_host()
    else:
        app.skeleton_host._switch_to_skeleton_host()


def _on_plane_toggle(app):
    """Handle guide plane checkbox toggle - separate from _update_visibility to avoid ghost widgets"""
    plane_visible = app.show_plane_chk.isChecked()
    
    # Update existing plane actors visibility
    if app.morphology_view.plane_actor:
        app.morphology_view.plane_actor.visibility = plane_visible
    if app.morphology_view.skeleton_host_plane_actor:
        app.morphology_view.skeleton_host_plane_actor.visibility = plane_visible
    if app.morphology_view.scale_line_actor:
        app.morphology_view.scale_line_actor.visibility = plane_visible
    if app.morphology_view.skeleton_host_scale_actor:
        app.morphology_view.skeleton_host_scale_actor.visibility = plane_visible
    
    # Selection mode: need to create plane but avoid triggering ghost sphere
    if (app.state.host_mode == 'skeleton' and 
        app.skeleton_host.morph_mode == 2 and
        plane_visible and 
        app.state.sphere_position is not None):
        
        # Remove large plane
        for attr in ['plane_actor', 'scale_line_actor', 'scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)
        
        # Create correct plane
        has_plane = app.morphology_view.skeleton_host_plane_actor is not None
        if not has_plane:
            app.skeleton_host._update_skeleton_host_guide_plane(app.state.sphere_position)
    
    # Navigation mode
    elif (app.state.host_mode == 'skeleton' and 
        app.skeleton_host.morph_mode == 1 and
        plane_visible and 
        app.state.sphere_position is not None):
        if app.morphology_view.skeleton_host_plane_actor is None:
            app.skeleton_host._update_skeleton_host_guide_plane(app.state.sphere_position)
    
    # Instance mode (mode 0)
    elif (app.state.host_mode == 'skeleton' and 
        app.skeleton_host.morph_mode == 0 and
        plane_visible and 
        app.state.sphere_position is not None):
        if app.morphology_view.skeleton_host_plane_actor is None:
            app.skeleton_host._update_skeleton_host_guide_plane(app.state.sphere_position)
        
    elif (app.state.host_mode == 'image' and 
        plane_visible and 
        app.state.tracked_state['label_id'] is not None):
        if app.morphology_view.plane_actor is None:
            app.image_host._update_image_host_guide_plane()
    
    # Only render morphology_plotter, don't touch skeleton_plotter
    app.morphology_view.plotter.render()


def _on_contour_toggle(app):
    """Handle contour checkbox toggle"""
    print(f"\n[CONTOUR TOGGLE] Checkbox: {app.show_plane_contour_chk.isChecked()}")
    
    # If no CC is tracked, do nothing
    if app.state.tracked_state['label_id'] is None:
        print(f"  [CONTOUR] No CC tracked")
        app.show_plane_contour_chk.setChecked(False)
        return
    
    if app.show_plane_contour_chk.isChecked():
        print(f"  [CONTOUR] Creating contour...")
        
        # Selection mode (mode 2): if contour doesn't exist, create it
        if app.state.host_mode == 'skeleton' and app.skeleton_host.morph_mode == 2:
            print(f"  [CONTOUR] Selection mode")
            has_contour = (app.morphology_view.contour_actor is not None or 
                          app.morphology_view.skeleton_host_contour_actor is not None)
            
            if has_contour:
                # Has existing contour, just toggle visibility
                if app.morphology_view.contour_actor:
                    app.morphology_view.contour_actor.visibility = True
                if app.morphology_view.skeleton_host_contour_actor:
                    app.morphology_view.skeleton_host_contour_actor.visibility = True
            else:
                # No contour, create one
                if app.state.sphere_position is not None:
                    app.skeleton_host._update_skeleton_host_contour(app.state.sphere_position)
            
            app.morphology_view.plotter.render()
            return
        
        # Image Host mode
        if app.state.host_mode == 'image':
            # Use _update_plane_contour (it uses _current_cc_mask_L0)
            if app.state._current_cc_mask_L0 is not None:
                app.image_host._update_plane_contour()
                app.morphology_view.plotter.render()
            else:
                print(f"  [CONTOUR] No CC mask available")
                app.show_plane_contour_chk.setChecked(False)
            return
    
        # Navigation mode (mode 1): dynamic creation
        if app.state.cc_path_3d is None:
            print(f"  [CONTOUR] No path available, cannot create contour")
            app.show_plane_contour_chk.setChecked(False)
            return
        
        z_L2 = app.state.tracked_state['z_L2']
        xy_bbox = app.cc_tracker._calculate_union_xy_bbox_from_path(app.state.cc_path_3d)
        
        if xy_bbox is None:
            print(f"  [CONTOUR] Cannot calculate bbox")
            app.show_plane_contour_chk.setChecked(False)
            return
        
        app.skeleton_host._add_cc_contour_on_plane(z_L2, xy_bbox)
        
    else:
        # Hide contour
        print(f"  [CONTOUR] Removing contour...")
        
        if app.morphology_view.contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.contour_actor)
                print(f"    ✓ Removed contour_actor")
            except:
                pass
            app.morphology_view.contour_actor = None
        
        if app.morphology_view.skeleton_host_contour_actor:
            try:
                app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
                print(f"    ✓ Removed skeleton_host_contour_actor")
            except:
                pass
            app.morphology_view.skeleton_host_contour_actor = None
        
        app.morphology_view.plotter.render()


def _on_opacity_changed(app, value):
    """Update labels opacity"""
    opacity = value / 100.0
    app.napari_view.lab_layer.opacity = opacity


def _on_camera_sync_toggle(app):
    """Handle camera sync checkbox toggle"""
    app.state.camera_sync_enabled = app.sync_camera_chk.isChecked()
    
    if app.state.camera_sync_enabled:
        print(f"[SYNC] Camera sync ENABLED (bidirectional)")
        
        # Reset stored states
        for attr in ['_last_skel_view', '_last_skel_up', '_last_morph_view', '_last_morph_up']:
            if hasattr(app, attr):
                delattr(app, attr)
        
        # Start sync timer (check every 50ms for smoother sync)
        app._camera_sync_timer.start(50)
        
        # Initial sync: copy skeleton orientation to morphology
        try:
            skel_cam = app.skeleton_view.plotter.camera
            morph_cam = app.morphology_view.plotter.camera
            
            skel_pos = np.array(skel_cam.position)
            skel_focal = np.array(skel_cam.focal_point)
            skel_view = skel_pos - skel_focal
            skel_view_norm = skel_view / (np.linalg.norm(skel_view) + 1e-10)
            
            morph_focal = np.array(morph_cam.focal_point)
            morph_pos = np.array(morph_cam.position)
            morph_distance = np.linalg.norm(morph_pos - morph_focal)
            
            new_morph_pos = morph_focal + skel_view_norm * morph_distance
            morph_cam.position = tuple(new_morph_pos)
            morph_cam.up = skel_cam.up
            
            app.morphology_view.plotter.render()
            print(f"  Initial sync: morphology → skeleton orientation")
        except:
            pass
        
    else:
        print(f"[SYNC] Camera sync DISABLED")
        # Stop sync timer
        app._camera_sync_timer.stop()
        
        # Clear stored states
        for attr in ['_last_skel_view', '_last_skel_up', '_last_morph_view', '_last_morph_up']:
            if hasattr(app, attr):
                delattr(app, attr)


def _update_visibility(app):
    """Toggle visibility of various elements"""
    
    if app.state.host_mode == 'skeleton':
        renderer = app.skeleton_view.plotter.renderer
        actors = renderer.GetActors()
        actors.InitTraversal()
        print(f"[DEBUG] Total actors in skeleton_view renderer: {actors.GetNumberOfItems()}")
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            bounds = actor.GetBounds() if actor else None
            vis = actor.GetVisibility() if actor else None
            print(f"  Actor {i}: visible={vis}, bounds={bounds}")

    # Marker visibility
    if app.skeleton_view.black_marker_actor:
        if app.state.host_mode == 'image':
            app.skeleton_view.black_marker_actor.visibility = app.show_marker_chk.isChecked()
            app.skeleton_view.plotter.render()
        else:
            # Skeleton Host mode: marker always hidden
            app.skeleton_view.black_marker_actor.visibility = False
    
    plane_visible = app.show_plane_chk.isChecked()

    # Image Host mode actors
    if app.morphology_view.plane_actor:
        app.morphology_view.plane_actor.visibility = plane_visible
    
    if app.morphology_view.scale_line_actor:
        app.morphology_view.scale_line_actor.visibility = plane_visible
    
    if app.morphology_view.scale_text_actor:
        if hasattr(app.morphology_view.scale_text_actor, 'SetVisibility'):
            app.morphology_view.scale_text_actor.SetVisibility(plane_visible)
        elif hasattr(app.morphology_view.scale_text_actor, 'visibility'):
            app.morphology_view.scale_text_actor.visibility = plane_visible
    
    # Skeleton Host mode actors
    if app.morphology_view.skeleton_host_plane_actor:
        app.morphology_view.skeleton_host_plane_actor.visibility = plane_visible
    
    if app.morphology_view.skeleton_host_scale_actor:
        app.morphology_view.skeleton_host_scale_actor.visibility = plane_visible
    
    if app.morphology_view.skeleton_host_scale_text_actor:
        app.morphology_view.skeleton_host_scale_text_actor.visibility = plane_visible
    
    # Selection mode (mode 2): remove union bbox plane, create plane at sphere position
    if (app.state.host_mode == 'skeleton' and 
        app.skeleton_host.morph_mode == 2 and
        plane_visible and 
        app.state.sphere_position is not None):

        # Remove plane created by _update_guide_plane_for_path
        for attr in ['plane_actor', 'scale_line_actor', 'scale_text_actor']:
            actor = getattr(app.morphology_view, attr, None)
            if actor:
                try:
                    app.morphology_view.plotter.remove_actor(actor)
                except:
                    pass
                setattr(app.morphology_view, attr, None)
        
        # If no skeleton_host_plane_actor, create one
        has_plane = app.morphology_view.skeleton_host_plane_actor is not None
        if not has_plane:
            app.skeleton_host._update_skeleton_host_guide_plane(app.state.sphere_position)

    # Navigation mode (mode 1): dynamic guide plane creation
    elif (app.state.host_mode == 'skeleton' and 
        app.skeleton_host.morph_mode == 1 and
        plane_visible and 
        app.state.sphere_position is not None):
        if app.morphology_view.skeleton_host_plane_actor is None:
            app.skeleton_host._update_skeleton_host_guide_plane(app.state.sphere_position)

    # Instance mode (mode 0)
    elif (app.state.host_mode == 'skeleton' and 
        app.skeleton_host.morph_mode == 0 and
        plane_visible and 
        app.state.sphere_position is not None):
        if app.morphology_view.skeleton_host_plane_actor is None:
            app.skeleton_host._update_skeleton_host_guide_plane(app.state.sphere_position)

    # Contour visibility
    if app.morphology_view.contour_actor:
        app.morphology_view.contour_actor.visibility = app.show_plane_contour_chk.isChecked()
    
    if app.morphology_view.skeleton_host_contour_actor:
        app.morphology_view.skeleton_host_contour_actor.visibility = app.show_plane_contour_chk.isChecked()
    
    app.morphology_view.plotter.render()
    #app.skeleton_view.plotter.render()

# =============================================================================
# SHARED VIEW OPERATIONS (used by both modes)
# =============================================================================

def _reset_2d_view(app):
    """Reset Napari 2D view to fit all data"""
    app.state.auto_center_enabled = False
    app.center_btn.setText('Center on 2D CC')
    app.napari_view.viewer.reset_view()


def _center_on_component(app, from_button=True):
    """Center napari view on CURRENT tracked CC in current slice"""
    if app.state.tracked_state['label_id'] is None:
        print("[INFO] No component tracked")
        return
    
    if from_button:
        if not app.state.auto_center_enabled:
            app.state.auto_center_enabled = True
            app.center_btn.setText('Center on 2D CC ●')
            print("[AUTO CENTER] Enabled")
        else:
            # Already enabled, this click disables it
            app.state.auto_center_enabled = False
            app.center_btn.setText('Center on 2D CC')
            print("[AUTO CENTER] Disabled")
            return
    
    try:
        label_name = app.state.tracked_state['label_name']
        label_id = app.state.tracked_state['label_id']
        current_slice = int(app.napari_view.viewer.dims.current_step[0])
        z_L2 = current_slice // 4  # Convert to L2 coords
        
        print(f"[CENTER] Attempting to center on {label_name} @ slice {current_slice} (z_L2={z_L2})")
        
        if app.state.host_mode == 'skeleton':
            # Skeleton Host: use L0 directly (consistent with _update_2d_view_direct)
            z_L0 = app.state.tracked_state['z_L0']
            cc_id = app.state.tracked_state['cc_id']
            
            if z_L0 is None or cc_id is None or cc_id < 0:
                print(f"  [Skeleton Host] No valid tracking state")
                return
            
            print(f"  [Skeleton Host] Using L0: z={z_L0}, CC#{cc_id}")
            
            # Get CC from L0 lab_full
            labels_slice = app.data.lab_full[z_L0]
            if hasattr(labels_slice, 'compute'):
                labels_slice = labels_slice.compute()
            
            label_mask = (labels_slice == label_id)
            
            if not label_mask.any():
                print(f"  [Skeleton Host] Label {label_id} not found at z={z_L0}")
                return
            
            from scipy import ndimage
            labeled, _ = ndimage.label(label_mask)
            cc_mask = (labeled == (cc_id + 1))
            
            if not cc_mask.any():
                print(f"  [Skeleton Host] CC#{cc_id} not found at z={z_L0}")
                return
            
            # Calculate center (already L0 coordinates)
            y_coords, x_coords = np.where(cc_mask)
            center_y_L0 = (y_coords.min() + y_coords.max()) / 2
            center_x_L0 = (x_coords.min() + x_coords.max()) / 2
            height_L0 = y_coords.max() - y_coords.min() + 1
            width_L0 = x_coords.max() - x_coords.min() + 1
            
            print(f"  [Skeleton Host] CC at L0: center=({center_y_L0:.1f}, {center_x_L0:.1f}), size={width_L0}×{height_L0}")
            
            # Set camera center
            app.napari_view.viewer.camera.center = (
                z_L0 * app.data.voxel_size_L0[0],
                center_y_L0 * app.data.voxel_size_L0[1],
                center_x_L0 * app.data.voxel_size_L0[2]
            )
            
            # Calculate zoom
            try:
                canvas_height, canvas_width = app.napari_view.viewer.window.qt_viewer.canvas.size
            except:
                canvas_height, canvas_width = 800, 800
            
            padding = 100

            height_world = height_L0 * app.data.voxel_size_L0[1]
            width_world = width_L0 * app.data.voxel_size_L0[2]

            zoom_y = (canvas_height - 2*padding) / height_world if height_world > 0 else 1
            zoom_x = (canvas_width - 2*padding) / width_world if width_world > 0 else 1
            target_zoom = min(zoom_y, zoom_x)
            final_zoom = max(0.2, min(target_zoom * 1.2, 50.0))
            
            app.napari_view.viewer.camera.zoom = final_zoom
            
            print(f"  [Skeleton Host] ✓ Centered! Zoom: {final_zoom:.2f}")
            return  # Return directly, don't run Image Host logic below

        # Image Host mode
        # Check if we're tracking a CC at this Z level
        if app.state.cc_path_3d is None:
            print(f"  No path available - jumping to original click position")
            z_target = app.state.tracked_state['z_L0']
            if z_target is not None and z_target != current_slice:
                app.napari_view.viewer.dims.set_point(0, z_target * app.data.voxel_size_L0[0])
                QtCore.QTimer.singleShot(100, lambda: _center_on_component(app, from_button=False))
            return
        
        # Check if current z_L2 is in the path
        if z_L2 not in app.state.cc_path_3d:
            print(f"  z_L2={z_L2} not in path - jumping to tracked position")
            z_target = app.state.tracked_state['z_L0']
            if z_target is not None and z_target != current_slice:
                app.napari_view.viewer.dims.set_point(0, z_target * app.data.voxel_size_L0[0])
                QtCore.QTimer.singleShot(100, lambda: _center_on_component(app, from_button=False))
            return
        
        # Get the CC at this Z from path
        ccs_at_z = app.state.cc_path_3d[z_L2]
        if len(ccs_at_z) == 0:
            print(f"  No CCs at z_L2={z_L2}")
            return
        
        cc_label_id, cc_id = ccs_at_z[0]
        cc_label_name = app.data.label_names[cc_label_id - 1]
        
        print(f"  Centering on {cc_label_name} CC#{cc_id}")
        
        # Load outer mask at L2
        if cc_label_name not in app.data.outer_masks_L2:
            print(f"  No mask for {cc_label_name}")
            return
        
        outer_mask = app.data.outer_masks_L2[cc_label_name]
        
        if z_L2 >= outer_mask.shape[0]:
            print(f"  z_L2={z_L2} out of bounds")
            return
        
        # Get slice at L2
        slice_mask = outer_mask[z_L2].compute()
        
        # Label CCs
        from scipy import ndimage
        labeled, _ = ndimage.label(slice_mask > 0)
        
        # Get the specific CC (cc_id is 0-indexed, labels are 1-indexed)
        cc_mask = (labeled == (cc_id + 1))
        
        if not cc_mask.any():
            print(f"  CC#{cc_id} not found at z_L2={z_L2}")
            return
        
        # Find bounding box of THIS CC at L2
        y_coords_L2, x_coords_L2 = np.where(cc_mask)
        
        y_min_L2 = y_coords_L2.min()
        y_max_L2 = y_coords_L2.max()
        x_min_L2 = x_coords_L2.min()
        x_max_L2 = x_coords_L2.max()
        
        # Calculate center at L2
        center_y_L2 = (y_min_L2 + y_max_L2) / 2
        center_x_L2 = (x_min_L2 + x_max_L2) / 2
        
        # Convert center to L0 coordinates (multiply by 4)
        center_y_L0 = center_y_L2 * 4
        center_x_L0 = center_x_L2 * 4
        
        # Calculate size at L2
        height_L2 = y_max_L2 - y_min_L2 + 1
        width_L2 = x_max_L2 - x_min_L2 + 1
        
        # Convert size to L0 (multiply by 4)
        height_L0 = height_L2 * 4
        width_L0 = width_L2 * 4
        
        print(f"  CC at L2: center=({center_y_L2:.1f}, {center_x_L2:.1f}), size={width_L2}×{height_L2}")
        print(f"  CC at L0: center=({center_y_L0:.1f}, {center_x_L0:.1f}), size={width_L0:.0f}×{height_L0:.0f}")
        
        # Set camera center (L0 coordinates)
        app.napari_view.viewer.camera.center = (
            current_slice * app.data.voxel_size_L0[0],
            center_y_L0 * app.data.voxel_size_L0[1],
            center_x_L0 * app.data.voxel_size_L0[2]
        )
        
        # Calculate zoom based on CC size at L0
        try:
            canvas_height, canvas_width = app.napari_view.viewer.window.qt_viewer.canvas.size
        except:
            canvas_height, canvas_width = 800, 800
        
        padding = 100  # pixels

        height_world = height_L0 * app.data.voxel_size_L0[1]
        width_world = width_L0 * app.data.voxel_size_L0[2]
        
        zoom_y = (canvas_height - 2*padding) / height_world if height_world > 0 else 1
        zoom_x = (canvas_width - 2*padding) / width_world if width_world > 0 else 1
        target_zoom = min(zoom_y, zoom_x)
        
        zoom_boost = 1.2
        final_zoom = max(0.2, min(target_zoom * zoom_boost, 50.0))
        
        app.napari_view.viewer.camera.zoom = final_zoom
        
        print(f"  ✓ Centered! Zoom: {final_zoom:.2f}")
    
    except Exception as e:
        print(f"[ERROR] Center failed: {e}")
        import traceback
        traceback.print_exc()


def _sync_cameras(app):
    """
    Bidirectional camera sync between skeleton and morphology views
    
    Detects which camera was rotated and syncs the other one to match
    """
    if not app.state.camera_sync_enabled:
        return
    
    try:
        skel_cam = app.skeleton_view.plotter.camera
        morph_cam = app.morphology_view.plotter.camera
        
        # Get current view directions and up vectors
        skel_pos = np.array(skel_cam.position)
        skel_focal = np.array(skel_cam.focal_point)
        skel_view = skel_pos - skel_focal
        skel_view_norm = skel_view / (np.linalg.norm(skel_view) + 1e-10)
        skel_up = np.array(skel_cam.up)
        
        morph_pos = np.array(morph_cam.position)
        morph_focal = np.array(morph_cam.focal_point)
        morph_view = morph_pos - morph_focal
        morph_view_norm = morph_view / (np.linalg.norm(morph_view) + 1e-10)
        morph_up = np.array(morph_cam.up)
        
        # Store current states if first run
        if not hasattr(app, '_last_skel_view'):
            app._last_skel_view = skel_view_norm.copy()
            app._last_skel_up = skel_up.copy()
            app._last_morph_view = morph_view_norm.copy()
            app._last_morph_up = morph_up.copy()
            return
        
        # Calculate how much each camera changed
        skel_view_change = np.linalg.norm(skel_view_norm - app._last_skel_view)
        skel_up_change = np.linalg.norm(skel_up - app._last_skel_up)
        skel_change = skel_view_change + skel_up_change
        
        morph_view_change = np.linalg.norm(morph_view_norm - app._last_morph_view)
        morph_up_change = np.linalg.norm(morph_up - app._last_morph_up)
        morph_change = morph_view_change + morph_up_change
        
        # Threshold for detecting change (avoid noise)
        threshold = 0.001
        
        if skel_change > threshold and skel_change > morph_change:
            # Skeleton camera was moved -> sync morphology to skeleton
            morph_distance = np.linalg.norm(morph_view)
            new_morph_pos = morph_focal + skel_view_norm * morph_distance
            
            morph_cam.position = tuple(new_morph_pos)
            morph_cam.up = tuple(skel_up)
            
            app.morphology_view.plotter.render()
            
            # Update stored state
            app._last_morph_view = skel_view_norm.copy()
            app._last_morph_up = skel_up.copy()
            
        elif morph_change > threshold and morph_change > skel_change:
            # Morphology camera was moved -> sync skeleton to morphology
            skel_distance = np.linalg.norm(skel_view)
            new_skel_pos = skel_focal + morph_view_norm * skel_distance
            
            skel_cam.position = tuple(new_skel_pos)
            skel_cam.up = tuple(morph_up)
            
            app.skeleton_view.plotter.render()
            
            # Update stored state
            app._last_skel_view = morph_view_norm.copy()
            app._last_skel_up = morph_up.copy()
        
        # Always update stored states
        app._last_skel_view = np.array(skel_cam.position) - np.array(skel_cam.focal_point)
        app._last_skel_view = app._last_skel_view / (np.linalg.norm(app._last_skel_view) + 1e-10)
        app._last_skel_up = np.array(skel_cam.up)
        
        app._last_morph_view = np.array(morph_cam.position) - np.array(morph_cam.focal_point)
        app._last_morph_view = app._last_morph_view / (np.linalg.norm(app._last_morph_view) + 1e-10)
        app._last_morph_up = np.array(morph_cam.up)
        
    except Exception as e:
        print(f"[SYNC] Camera sync error: {e}")


def _set_active_view(app, view_name, event):
    """Set which 3D view is active for keyboard control"""
    app.state.active_3d_view = view_name
    
    # Force focus back to main widget
    QtCore.QTimer.singleShot(10, app.setFocus)
    
    print(f"[FOCUS] Active 3D view: {view_name}")
    
    # Call original mouse press handler
    if view_name == 'skeleton':
        QtInteractor.mousePressEvent(app.skeleton_view.plotter.interactor, event)
    else:
        QtInteractor.mousePressEvent(app.morphology_view.plotter.interactor, event)

def _clear_morphology_actors(app):
    """Clear ALL morphology view actors - OPTIMIZED batch removal"""
    
    plotter = app.morphology_view.plotter
    renderer = plotter.renderer  # Direct VTK access for speed
    
    # Collect all actors to remove
    actors_to_remove = []
    
    # === Image Host actors ===
    if hasattr(app.morphology_view, 'outer_volume_actors'):
        actors_to_remove.extend(app.morphology_view.outer_volume_actors.values())
    
    if hasattr(app.morphology_view, 'inner_mesh_actors'):
        actors_to_remove.extend(app.morphology_view.inner_mesh_actors.values())
    
    # === Label mode actors (usually the most) ===
    if hasattr(app.morphology_view, 'label_mode_actors'):
        actors_to_remove.extend(app.morphology_view.label_mode_actors.values())
    
    # Image Host plane/contour/scale
    for attr in ['plane_actor', 'contour_actor', 'scale_line_actor', 'scale_text_actor']:
        actor = getattr(app.morphology_view, attr, None)
        if actor:
            actors_to_remove.append(actor)
            setattr(app.morphology_view, attr, None)
    
    # === Skeleton Host actors ===
    if app.morphology_view.skeleton_host_contour_actor:
        actors_to_remove.append(app.morphology_view.skeleton_host_contour_actor)
        app.morphology_view.skeleton_host_contour_actor = None
    
    for attr in ['skeleton_host_plane_actor', 'skeleton_host_scale_actor', 'skeleton_host_scale_text_actor']:
        actor = getattr(app.morphology_view, attr, None)
        if actor:
            actors_to_remove.append(actor)
            setattr(app.morphology_view, attr, None)
    
    # === Skeleton Host global morph actors ===
    if hasattr(app.morphology_view, 'global_morph_actors'):
        for mesh_name, data in app.morphology_view.global_morph_actors.items():
            if 'actor' in data:
                actors_to_remove.append(data['actor'])
    
    # === Batch remove using VTK directly (faster than PyVista wrapper) ===
    for actor in actors_to_remove:
        if actor is not None:
            try:
                renderer.RemoveActor(actor)
            except:
                pass
    
    # Reset dicts (after removal)
    app.morphology_view.outer_volume_actors = {}
    app.morphology_view.inner_mesh_actors = {}
    app.morphology_view.label_mode_actors = {}
    app.morphology_view.global_morph_actors = {}
    
def _zoom_to_path_bbox(app, xy_bbox, z_current):
    """Zoom to path bbox - preserve camera orientation"""
    center_x = (xy_bbox['x_min'] + xy_bbox['x_max']) / 2 * app.data.voxel_size_L2[2]
    center_y = (xy_bbox['y_min'] + xy_bbox['y_max']) / 2 * app.data.voxel_size_L2[1]
    center_z = z_current * app.data.voxel_size_L2[0]
    
    size_x = (xy_bbox['x_max'] - xy_bbox['x_min']) * app.data.voxel_size_L2[2]
    size_y = (xy_bbox['y_max'] - xy_bbox['y_min']) * app.data.voxel_size_L2[1]
    
    # CRITICAL FIX: Read BEFORE setting focal point
    current_pos = np.array(app.morphology_view.plotter.camera.position)
    current_focal = np.array(app.morphology_view.plotter.camera.focal_point)
    
    # Calculate view direction from ORIGINAL positions
    view_vector = current_pos - current_focal
    view_direction = view_vector / (np.linalg.norm(view_vector) + 1e-10)
    
    # Calculate new distance
    max_size = max(size_x, size_y)
    camera_distance = max_size * 2.0
    
    # Calculate new positions
    new_focal = np.array([center_x, center_y, center_z])
    new_position = new_focal + view_direction * camera_distance
    
    # Now set both together
    app.morphology_view.plotter.camera.focal_point = tuple(new_focal)
    app.morphology_view.plotter.camera.position = tuple(new_position)
    
    app.morphology_view.plotter.reset_camera_clipping_range()
    
    print(f"  ✓ Zoomed to bbox: {size_x:.1f} × {size_y:.1f} μm")
    print(f"    View direction preserved: {view_direction}")

# =========================================================================
# VIEW MANAGEMENT
# =========================================================================
    
def _reset_views(app):
    """Reset cameras to initial positions"""
    try:
        # Reset skeleton view camera
        app.skeleton_view.plotter.camera.position = (0, 0, -1000)
        app.skeleton_view.plotter.camera.focal_point = (0, 0, 0)
        app.skeleton_view.plotter.camera.up = (0, -1, 0)
        app.skeleton_view.plotter.reset_camera()
        
        # Reset morphology view camera
        app.morphology_view.plotter.camera.position = (0, 0, -1000)
        app.morphology_view.plotter.camera.focal_point = (0, 0, 0)
        app.morphology_view.plotter.camera.up = (0, -1, 0)
        
        if app.state.host_mode == 'skeleton' and app.state.sphere_position is not None:
            range_um = app.skeleton_host._get_dynamic_sphere_radius(app.state.sphere_position)
            app.skeleton_host._update_morphology_camera_to_sphere(app.state.sphere_position, range_um)
        else:
            app.morphology_view.plotter.reset_camera()
        
        print("[RESET] Camera orientations reset to initial view")
        
    except Exception as e:
        print(f"[ERROR] Reset views failed: {e}")

def _clear_tracking(app):
    """Clear all tracking state and reset views"""
    print("\n[CLEAR TRACKING] Clearing all...")
    
    # Disable auto-center
    app.state.auto_center_enabled = False
    app.center_btn.setChecked(False)
    app.center_btn.setText('Center on 2D CC')
    
    # Reset tracked state
    app.state.tracked_state = {
        'label_id': None,
        'label_name': None,
        'z_L0': None,
        'z_L2': None,
        'cc_id': None
    }
    
    # Reset instance tracking
    app.skeleton_host.current_skeleton_instance_id = -1
    
    # === Clear Napari boundary ===
    app.napari_view.boundary_layer.data = []
    
    # === Clear Skeleton View ===
    if app.skeleton_view.black_marker_actor:
        try:
            app.skeleton_view.plotter.remove_actor(app.skeleton_view.black_marker_actor)
        except:
            pass
        app.skeleton_view.black_marker_actor = None
    
    # Hide sphere widget
    if app.skeleton_view.sphere_widget:
        try:
            app.skeleton_view.sphere_widget.GetSphereProperty().SetOpacity(0)
            app.skeleton_view.sphere_widget.Off()
        except:
            pass
    app.state.sphere_position = None
    
    # Clear selection state
    app.skeleton_host._clear_selection_highlight()
    app.skeleton_host.selected_skeleton_indices = set()
    app.skeleton_host.selection_drawing = False
    app.draw_selection_btn.setChecked(False)
    app.skeleton_host._selected_kdtree = None
    app.skeleton_host._selected_list = None
    
    app.skeleton_view.plotter.render()
    
    # === Clear Morphology View ===
    _clear_morphology_actors(app)
    
    # Clear global morph actors
    for mesh_name, data in list(app.morphology_view.global_morph_actors.items()):
        try:
            app.morphology_view.plotter.remove_actor(data['actor'])
        except:
            pass
    app.morphology_view.global_morph_actors = {}
    
    # Clear plane actors
    for attr in ['plane_actor', 'scale_line_actor', 'scale_text_actor',
                'skeleton_host_plane_actor', 'skeleton_host_scale_actor', 'skeleton_host_scale_text_actor']:
        actor = getattr(app.morphology_view, attr, None)
        if actor:
            try:
                app.morphology_view.plotter.remove_actor(actor)
            except:
                pass
            setattr(app.morphology_view, attr, None)
    
    # Clear contour actors
    if app.morphology_view.contour_actor:
        try:
            app.morphology_view.plotter.remove_actor(app.morphology_view.contour_actor)
        except:
            pass
        app.morphology_view.contour_actor = None
    
    if app.morphology_view.skeleton_host_contour_actor:
        try:
            app.morphology_view.plotter.remove_actor(app.morphology_view.skeleton_host_contour_actor)
        except:
            pass
        app.morphology_view.skeleton_host_contour_actor = None
    
    app.morphology_view.plotter.render()
    
    # === Update UI ===
    app.show_plane_chk.setChecked(False)
    app.show_plane_contour_chk.setChecked(False)
    
    app.skeleton_view.skeleton_info.setText("Color-coded skeleton")
    app.morphology_view.morphology_info.setText("Outer/Inner volumes")
    app.slice_info.setText("Slice: 0 / 0  |  Z = 0.0 μm  |  No CC tracked")
    
    print("[CLEAR TRACKING] Done - all views cleared")