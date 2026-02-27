"""
Napari 2D View
NOTE: Only setup function here. All operations are in modes/*.

Mouse interaction:
  XY view:    left drag = crosshair, right drag = pan
  Ortho views: left drag = crosshair, right drag = pan
"""

import os
import numpy as np
import napari
from qtpy import QtWidgets, QtCore

from ..utils import _hex_to_rgba
from .control_panel import _create_control_panel

#os.environ['NAPARI_ASYNC'] = '1'
try:
    from dask.cache import Cache
    _dask_cache = Cache(4e9)
    _dask_cache.register()
    print("[CACHE] Dask 4GB cache registered")
except Exception as e:
    print(f"[CACHE] Dask cache not available: {e}")


class NapariViewController:
    """2D Napari view"""

    def __init__(self, app):
        self.app = app
        self.viewer = None
        self.img_layer = None
        self.lab_layer = None
        self._dragging = False
        self._ortho_viewers = []
        self._ortho_filters = []
        self._ortho_manager = None
        self._syncing = False  # <-- add this

    def _setup_napari_panel(self):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self.viewer = napari.Viewer(show=False)
        self.viewer.theme = 'light'

        try:
            from napari.settings import get_settings
            get_settings().experimental.async_ = False
            print("[ASYNC] Napari async rendering enabled")
        except Exception as e:
            print(f"[ASYNC] Could not enable async: {e}")

        try:
            self.viewer.window._qt_viewer.canvas.bgcolor = (0.97, 0.97, 0.97, 1.0)
        except:
            pass

        napari_window = self.viewer.window._qt_window
        self.viewer.axes.visible = True
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'μm'

        data = self.app.data

        is_rgb = len(data.img_full.shape) == 4
        self.img_layer = self.viewer.add_image(
            data.img_levels[data.display_level], name='Histology', rgb=is_rgb,
            opacity=0.8, scale=data.voxel_size_display, cache=True
        )

        from napari.utils.colormaps import DirectLabelColormap
        color_dict = {None: np.array([0, 0, 0, 0])}
        print(f"\n[COLORS] Building colormap:")
        for label_id, label_name in enumerate(data.label_names, start=1):
            hex_color = data.label_colors[label_name]
            color_dict[label_id] = np.array(_hex_to_rgba(hex_color))
            print(f"  {label_id}: {label_name} = {hex_color}")

        self.lab_layer = self.viewer.add_labels(
            data.lab_levels[data.display_level], name='Labels', opacity=0.5,
            scale=data.voxel_size_display,
            colormap=DirectLabelColormap(color_dict=color_dict), cache=True
        )
        print(f"[COLORS] Added labels with custom colormap")
        self.viewer.layers.selection.active = self.lab_layer

        # XY: left drag = crosshair, right drag = pan
        self._register_drag_callback(self.viewer)
        self._register_pan_callback(self.viewer)

        layout.addWidget(napari_window, stretch=1)
        controls = _create_control_panel(self.app)
        layout.addWidget(controls, stretch=0)

        from napari_orthogonal_views.ortho_view_manager import show_orthogonal_views, _get_manager
        show_orthogonal_views(self.viewer)

        def _hook_ortho_callbacks():
            try:
                m = _get_manager(self.viewer)
                self._ortho_manager = m 
                self._ortho_viewers = []
                for w in [m.right_widget, m.bottom_widget]:
                    if hasattr(w, 'viewer'):
                        self._ortho_viewers.append(w.viewer)
                print(f"[ORTHO] Found {len(self._ortho_viewers)} ortho viewers")

                ortho_widgets = [m.right_widget, m.bottom_widget]

                for ow in ortho_widgets:
                    vm = ow.vm_container.viewer_model

                    try:
                        ow.qt_viewer.canvas.bgcolor = (0.97, 0.97, 0.97, 1.0)
                    except:
                        pass

                    try:
                        vm.scale_bar.visible = True
                        vm.scale_bar.unit = 'μm'
                    except:
                        pass

                    self._register_ortho_vispy(ow, vm)

                    if ow is m.right_widget:
                        try:
                            from napari._qt.widgets.qt_scrollbar import ModifiedScrollBar
                            for sb in ow.qt_viewer.findChildren(ModifiedScrollBar):
                                sb.setStyleSheet("""
                                    QScrollBar::handle:horizontal { min-width: 4px; }
                                    QScrollBar::handle:vertical { min-height: 4px; }
                                """)
                        except Exception as e:
                            print(f"[ORTHO] Slider fix error: {e}")

                try:
                    m.set_cross_hairs(True)
                    cw = m.main_controls_widget.controls_widget
                    cw.cross_widget.setStyleSheet("QCheckBox { font-size: 8pt; }")
                    cw.zoom_widget.setStyleSheet("QCheckBox { font-size: 8pt; }")
                    m.main_controls_widget.show_checkbox.setVisible(False)
                    cw.center_widget.setVisible(False)
                    extra_style = "font-size: 8pt;"

                    from qtpy.QtWidgets import QLabel, QPushButton, QHBoxLayout, QSlider
                    for lbl in cw.findChildren(QLabel):
                        if 'Press T' in lbl.text():
                            lbl.setVisible(False)

                    from ..controls import (_on_lock_angle_toggle,_reset_2d_view, _reset_3d_view,
                                            _on_opacity_changed, _on_inner_opacity_changed,
                                            _on_morph_light_angle_changed, _on_outer_opacity_changed, _on_show_planes_toggle)
                    lay = cw.layout()

                    def _add_slider_row(label_text, lo, hi, val, cb):
                        lbl = QLabel(label_text)
                        lbl.setStyleSheet(extra_style)
                        sl = QSlider(QtCore.Qt.Horizontal)
                        sl.setRange(lo, hi)
                        sl.setValue(val)
                        sl.valueChanged.connect(cb)
                        lay.addWidget(lbl)
                        lay.addWidget(sl)
                        return sl

                    self.app.opacity_slider = _add_slider_row(
                        'Labels Opacity', 0, 100, 50,
                        lambda v: _on_opacity_changed(self.app, v))

                    if self.app.data.has_inner_mask:
                        self.app.inner_opacity_label = QLabel('Inner Lumen Opacity: 100%')
                        self.app.inner_opacity_label.setStyleSheet(extra_style)
                        lay.addWidget(self.app.inner_opacity_label)
                        self.app.inner_opacity_slider = QSlider(QtCore.Qt.Horizontal)
                        self.app.inner_opacity_slider.setRange(0, 100)
                        self.app.inner_opacity_slider.setValue(100)
                        self.app.inner_opacity_slider.valueChanged.connect(
                            lambda v: _on_inner_opacity_changed(self.app, v))
                        lay.addWidget(self.app.inner_opacity_slider)
                    else:
                        self.app.outer_opacity_label = QLabel('Mesh Opacity: 80%')
                        self.app.outer_opacity_label.setStyleSheet(extra_style)
                        lay.addWidget(self.app.outer_opacity_label)
                        self.app.outer_opacity_slider = QSlider(QtCore.Qt.Horizontal)
                        self.app.outer_opacity_slider.setRange(0, 100)
                        self.app.outer_opacity_slider.setValue(80)
                        self.app.outer_opacity_slider.valueChanged.connect(
                            lambda v: _on_outer_opacity_changed(self.app, v))
                        lay.addWidget(self.app.outer_opacity_slider)

                    self.app.morph_light_label = QLabel('Light Angle: 45°')
                    self.app.morph_light_label.setStyleSheet(extra_style)
                    lay.addWidget(self.app.morph_light_label)
                    self.app.morph_light_slider = QSlider(QtCore.Qt.Horizontal)
                    self.app.morph_light_slider.setRange(0, 360)
                    self.app.morph_light_slider.setValue(45)
                    self.app.morph_light_slider.valueChanged.connect(
                        lambda v: _on_morph_light_angle_changed(self.app, v))
                    lay.addWidget(self.app.morph_light_slider)

                    btn_w = QtWidgets.QWidget()
                    btn_l = QHBoxLayout(btn_w)
                    btn_l.setContentsMargins(0, 2, 0, 0)
                    btn_l.setSpacing(4)
                    r2d = QPushButton('Reset 2D')
                    r3d = QPushButton('Reset 3D')
                    r2d.setStyleSheet(extra_style)
                    r3d.setStyleSheet(extra_style)
                    r2d.setFixedHeight(20)
                    r3d.setFixedHeight(20)
                    r2d.clicked.connect(lambda: _reset_2d_view(self.app))
                    r3d.clicked.connect(lambda: _reset_3d_view(self.app))
                    btn_l.addWidget(r2d)
                    btn_l.addWidget(r3d)
                    lay.addWidget(btn_w)

                    # Lock angle checkbox
                    self.app.lock_angle_chk = QtWidgets.QCheckBox('Lock Viewing Angle')
                    self.app.lock_angle_chk.setStyleSheet(extra_style)
                    self.app.lock_angle_chk.setChecked(False)
                    self.app.lock_angle_chk.stateChanged.connect(lambda: _on_lock_angle_toggle(self.app))
                    lay.addWidget(self.app.lock_angle_chk)

                    self.app.show_planes_chk = QtWidgets.QCheckBox('Show Cutting Planes')
                    self.app.show_planes_chk.setStyleSheet(extra_style)
                    self.app.show_planes_chk.setChecked(True)
                    self.app.show_planes_chk.stateChanged.connect(lambda: _on_show_planes_toggle(self.app))
                    lay.addWidget(self.app.show_planes_chk)

                    m.main_controls_widget.setMaximumHeight(300)
                    
                    print("[ORTHO] Controls configured")

                except Exception as e:
                    print(f"[ORTHO] Controls setup error: {e}")
                    import traceback; traceback.print_exc()

            except Exception as e:
                print(f"[ORTHO] Hook error: {e}")
                import traceback; traceback.print_exc()
                return

            def _log(tag, v):
                try:
                    print(f"[DIMS]{tag} step={v.dims.current_step}")
                except Exception as e:
                    print(f"[DIMS]{tag} error: {e}")

            self.viewer.dims.events.current_step.connect(lambda e: _log(" MAIN", self.viewer))
            for i, ov in enumerate(self._ortho_viewers):
                ov.dims.events.current_step.connect(lambda e, ii=i, vv=ov: _log(f" ORTHO[{ii}]", vv))
                

        QtCore.QTimer.singleShot(2000, _hook_ortho_callbacks)

        NAPARI_NATIVE_DOCKS = {'console', 'layer controls', 'layer list'}
        for dock in self.viewer.window._qt_window.findChildren(QtWidgets.QDockWidget):
            if dock.windowTitle().lower() in NAPARI_NATIVE_DOCKS:
                dock.setVisible(False)

        try:
            self.viewer.window._qt_window.menuBar().setVisible(False)
        except:
            pass
        try:
            self.viewer.window._qt_window.statusBar().setVisible(False)
        except:
            pass
            

        return container

    # =========================================================================
    # ORTHO: vispy events + ev.pos -> world coords
    # =========================================================================

    def _world_from_vispy_event(self, ow, ev):
        """Convert vispy event canvas position to world coords."""
        try:
            pos_px = ev.pos if hasattr(ev, 'pos') and len(np.array(ev.pos)) == 2 else np.array(ev.position)[:2]
            canvas = ow.qt_viewer.canvas
            vm = ow.vm_container.viewer_model
            vb = canvas.view
            transform = vb.transform * vb.scene.transform
            mapped = transform.imap(list(pos_px))
            mapped = mapped[:2]
            position_world_slice = np.array(mapped[::-1])
            world = list(self.viewer.dims.point)
            for i, d in enumerate(vm.dims.displayed):
                if i < len(position_world_slice):
                    world[d] = position_world_slice[i]
            return tuple(world)
        except Exception as e:
            print(f"[ORTHO] world_from_event error: {e}")
            return None
    
    def _register_ortho_vispy(self, ow, vm_ref):
        """Register ortho: left drag = crosshair, right drag = pan."""
        pan_start = [None]
        # 左鍵 crosshair — 用 napari 層級攔截
        @vm_ref.mouse_drag_callbacks.append
        def on_drag(vr, event):
            if event.type == 'mouse_press' and event.button == 1:
                event.handled = True
                world = self._world_from_vispy_event(ow, event)
                self._sync_world_to_main(world, vm_ref)
            yield
            while event.type == 'mouse_move':
                if event.button == 1:
                    event.handled = True
                    world = self._world_from_vispy_event(ow, event)
                    self._sync_world_to_main(world, vm_ref)
                yield
            if event.button == 1:
                event.handled = True
                world = self._world_from_vispy_event(ow, event)
                self._sync_world_to_main(world, vm_ref, update_3d=True)
        
        @vm_ref.mouse_drag_callbacks.append
        def on_pan(vr, event):
            if event.type == 'mouse_press' and event.button == 2:
                event.handled = True
                pan_start[0] = np.array([event.native.x(), event.native.y()])
            yield
            while event.type == 'mouse_move':
                if event.button == 2 and pan_start[0] is not None:
                    event.handled = True
                    cur = np.array([event.native.x(), event.native.y()])
                    delta = cur - pan_start[0]
                    pan_start[0] = cur
                    try:
                        sc = ow.qt_viewer.canvas._scene_canvas
                        cam = None
                        for child in sc.central_widget.children:
                            if hasattr(child, 'camera'):
                                cam = child.camera
                                break
                            for subchild in getattr(child, 'children', []):
                                if hasattr(subchild, 'camera'):
                                    cam = subchild.camera
                                    break
                            if cam:
                                break
                        if cam:
                            cam.pan(delta * [-1, -1])
                            ow.qt_viewer.canvas._scene_canvas.update()
                    except Exception as e:
                        print(f"[ORTHO] pan error: {e}")
                yield
            if event.button == 2:
                pan_start[0] = None
                event.handled = True

    def _sync_world_to_main(self, world, vm_ref, update_3d=False):
        if world is None:
            return
        if self._syncing:
            return

        self._syncing = True
        try:
            world_point = tuple(
                max(r.start, min(p, r.stop))
                for p, r in zip(world, self.viewer.dims.range)
            )

            m = getattr(self, '_ortho_manager', None)
            right_widget = m.right_widget if m else None
            bottom_widget = m.bottom_widget if m else None

            if right_widget and hasattr(right_widget, '_block_center'):
                right_widget._block_center = True
            if bottom_widget and hasattr(bottom_widget, '_block_center'):
                bottom_widget._block_center = True

            try:
                self.viewer.dims.point = world_point
                for ow in [right_widget, bottom_widget]:
                    if ow is None:
                        continue
                    try:
                        ow.vm_container.viewer_model.dims.point = world_point
                    except Exception as e:
                        print(f"[ORTHO] direct point set error: {e}")
            finally:
                if right_widget and hasattr(right_widget, '_block_center'):
                    right_widget._block_center = False
                if bottom_widget and hasattr(bottom_widget, '_block_center'):
                    bottom_widget._block_center = False

            # Force repaint on next event loop tick to avoid OpenGL framebuffer conflicts
            def _force_repaint():
                for ow in [right_widget, bottom_widget]:
                    if ow is None:
                        continue
                    try:
                        ow.qt_viewer.canvas._scene_canvas.update()
                        ow.qt_viewer.canvas.native.repaint()  # Qt-level force repaint
                    except Exception:
                        pass

            QtCore.QTimer.singleShot(0, _force_repaint)

            if update_3d:
                self.app._update_crosshair(*self.viewer.dims.current_step)

        except Exception as e:
            print(f"[ORTHO] sync error: {e}")
        finally:
            self._syncing = False
            
    # =========================================================================
    # XY view callbacks
    # =========================================================================

    def _register_pan_callback(self, viewer_ref):
        pan_start = [None]  # local variable

        @viewer_ref.mouse_drag_callbacks.append
        def on_pan(vr, event):
            if event.type == 'mouse_press' and event.button == 2:
                event.handled = True
                pan_start[0] = np.array([event.native.x(), event.native.y()])
            yield
            while event.type == 'mouse_move':
                if pan_start[0] is not None and event.button == 2:
                    event.handled = True
                    cur = np.array([event.native.x(), event.native.y()])
                    delta = cur - pan_start[0]
                    pan_start[0] = cur
                    try:
                        cam = None
                        sc = vr.window._qt_viewer.canvas._scene_canvas
                        for child in sc.central_widget.children:
                            if hasattr(child, 'camera'):
                                cam = child.camera
                                break
                            for subchild in getattr(child, 'children', []):
                                if hasattr(subchild, 'camera'):
                                    cam = subchild.camera
                                    break
                            if cam:
                                break
                        if cam:
                            cam.pan(delta * [-1, -1])
                            vr.window._qt_viewer.canvas.update()
                    except Exception:
                        pass
                yield
            if pan_start[0] is not None:
                pan_start[0] = None

    def _register_drag_callback(self, viewer_ref):
        """Left-click drag = crosshair, release updates 3D."""
        @viewer_ref.mouse_drag_callbacks.append
        def on_drag(vr, event):
            if event.type == 'mouse_press' and event.button == 1:
                event.handled = True
                self._dragging = True
                self._apply_crosshair(vr)
            yield
            while event.type == 'mouse_move':
                if self._dragging:
                    event.handled = True
                    self._apply_crosshair(vr)
                yield
            if self._dragging:
                self._dragging = False
                event.handled = True
                self._apply_crosshair(vr)
                self.app._update_crosshair(*self.viewer.dims.current_step)

    def _apply_crosshair(self, viewer_ref, pos=None):
        if pos is None:
            pos = viewer_ref.cursor.position
        self._sync_world_to_main(tuple(pos), None)
