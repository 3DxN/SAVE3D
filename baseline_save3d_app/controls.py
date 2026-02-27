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

def _on_opacity_changed(app, value):
    """Update labels opacity"""
    opacity = value / 100.0
    app.napari_view.lab_layer.opacity = opacity

def _on_inner_opacity_changed(app, value):
    """Update inner lumen opacity in morphology view"""
    opacity = value / 100.0
    app.inner_opacity_label.setText(f'Inner Lumen Opacity: {value}%')
    app.morphology_view.inner_opacity = opacity
    for actor in app.morphology_view.inner_actors.values():
        actor.GetProperty().SetOpacity(opacity)
    app.morphology_view.plotter.render()

def _on_morph_light_angle_changed(app, value):
    """Update morphology view light angle"""
    app.morph_light_label.setText(f'Light Angle: {value}°')
    app.morphology_view.set_light_angle(value)

def _set_active_view(app, view_name, event):
    """Set which 3D view is active for keyboard control"""
    app.state.active_3d_view = view_name
    
    # Force focus back to main widget
    QtCore.QTimer.singleShot(10, app.setFocus)
    
    print(f"[FOCUS] Active 3D view: {view_name}")
    
    QtInteractor.mousePressEvent(app.morphology_view.plotter.interactor, event)
def _reset_2d_view(app):
    """Reset Napari 2D view to fit all data"""
    app.napari_view.viewer.reset_view()

def _reset_3d_view(app):
    """Reset morphology 3D camera to initial position"""
    try:
        app.morphology_view.plotter.camera.position = (0, 0, -1000)
        app.morphology_view.plotter.camera.focal_point = (0, 0, 0)
        app.morphology_view.plotter.camera.up = (0, -1, 0)
        app.morphology_view.plotter.reset_camera()
        print("[RESET] 3D camera reset")
    except Exception as e:
        print(f"[RESET] Error: {e}")

def _on_lock_angle_toggle(app):
    """Handle lock viewing angle checkbox toggle"""
    locked = app.lock_angle_chk.isChecked()
    if locked:
        print("[LOCK] Viewing angle LOCKED")
        app.morphology_view.plotter.enable_image_style()
    else:
        print("[LOCK] Viewing angle UNLOCKED")
        app.morphology_view.plotter.enable_trackball_style()

def _on_outer_opacity_changed(app, value):
    opacity = value / 100.0
    app.outer_opacity_label.setText(f'Mesh Opacity: {value}%')
    app.morphology_view.outer_opacity = opacity
    for actor in app.morphology_view.outer_actors.values():
        actor.GetProperty().SetOpacity(opacity)
    app.morphology_view.plotter.render()

def _on_show_planes_toggle(app):
    app.show_planes = app.show_planes_chk.isChecked()
    if not app.show_planes:
        app.morphology_view.clear_planes()
    else:
        z, y, x = app.napari_view.viewer.dims.current_step
        app.morphology_view.update_crosshair(z, y, x)
    app.morphology_view.plotter.render()