"""
Morphology 3D View - Baseline
"""

import json
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
        self.app = app
        
        # PyVista components (set during setup)
        self.plotter = None
        self.main_light = None
        
        # Mesh actors
        self.outer_actors = {}
        self.inner_actors = {}
        self.crosshair_actors = []

        
    def set_light_angle(self, angle_deg):
        """Rotate main light around Y axis"""
        if self.main_light is None:
            return
        r = 10
        rad = np.radians(angle_deg)
        x = r * np.cos(rad)
        z = r * np.sin(rad)
        self.main_light.SetPosition(x, 10, z)
        self.plotter.render()

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
        self.plotter.set_background([0.97, 0.97, 0.97])
        
        try:
            self.plotter.enable_anti_aliasing()
            print("  ✓ Morphology: Anti-aliasing enabled")
        except Exception as e:
            print(f"  ⚠ Morphology: Anti-aliasing not available: {e}")
        
        try:
            self.plotter.enable_shadows = True
            print("  ✓ Morphology: Shadows enabled")
        except Exception as e:
            print(f"  ⚠ Morphology: Shadows not available: {e}")
        
        # Three-light setup
        try:
            z_flip = 1
            light1 = pv.Light(
                position=(10, 10, 10 * z_flip),
                focal_point=(0, 0, 0),
                color='white'
            )
            light2 = pv.Light(
                position=(-10, -10, 10 * z_flip),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.6
            )
            light3 = pv.Light(
                position=(0, 10, -10 * z_flip),
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.4
            )
            self.main_light = light1
            self.plotter.add_light(self.main_light)
            self.plotter.add_light(light2)
            self.plotter.add_light(light3)
            print("  ✓ Morphology: Three-light setup added")
        except Exception as e:
            self.main_light = None
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
        
        # Disable VTK default keyboard events
        try:
            self.plotter.iren.remove_observers("KeyPressEvent")
            self.plotter.iren.remove_observers("CharEvent")
        except:
            pass

        # Opacity defaults
        if self.app.data.has_inner_mask:
            self.outer_opacity = 0.2
            self.inner_opacity = 1.0
        else:
            self.outer_opacity = 0.8
            self.inner_opacity = 0.0

        # Load all meshes at startup
        self.app.data._load_global_morphology_meshes()
        mesh_info_path = self.app.data.zarr_path.parent / "morphology_meshes" / "mesh_info.json"
        with open(mesh_info_path) as f:
            mesh_info = json.load(f)
        label_colors = mesh_info.get('label_colors', {})

        self.outer_actors = {}
        self.inner_actors = {}
        for mesh_name, mesh in self.app.data.global_morph_meshes.items():
            if mesh_name.startswith('outer_'):
                label_name = mesh_name[6:]
                color = label_colors.get(label_name, '#808080')
                actor = self.plotter.add_mesh(mesh, color=color, opacity=self.outer_opacity, smooth_shading=True)
                self.outer_actors[label_name] = actor
            elif mesh_name.startswith('inner_'):
                label_name = mesh_name[6:]
                color = label_colors.get(label_name, '#808080')
                actor = self.plotter.add_mesh(mesh, color=color, opacity=self.inner_opacity, smooth_shading=True)
                self.inner_actors[label_name] = actor

        self.plotter.reset_camera()

        # Setup Shift+hover to update napari crosshair
        self._setup_hover_callback()

        return container

    def _setup_hover_callback(self):
        """Right-click on 3D mesh surface → update napari crosshair via ray casting.
        Uses same approach as skeleton_view: RightButtonPressEvent + SetDisplayPoint/DisplayToWorld.
        """
        # Pre-collect all outer mesh points for ray casting
        pts = []
        for mesh in self.app.data.global_morph_meshes.values():
            if hasattr(mesh, 'points'):
                pts.append(mesh.points)
        self._mesh_points = np.concatenate(pts, axis=0) if pts else None
        print(f"  [MORPH] Ray casting over {len(self._mesh_points)} mesh points")

        def _find_nearest_to_click(click_x, click_y):
            if self._mesh_points is None:
                return None
            renderer = self.plotter.renderer
            cam = renderer.GetActiveCamera()
            cam_pos = np.array(cam.GetPosition())

            renderer.SetDisplayPoint(click_x, click_y, 0.5)
            renderer.DisplayToWorld()
            world_pt = np.array(renderer.GetWorldPoint()[:3])

            ray_dir = world_pt - cam_pos
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            vecs = self._mesh_points - cam_pos
            dots = np.dot(vecs, ray_dir)
            proj = dots[:, None] * ray_dir
            perp_dists = np.linalg.norm(vecs - proj, axis=1)
            perp_dists[dots < 0] = np.inf  # behind camera

            idx = np.argmin(perp_dists)
            if perp_dists[idx] == np.inf:
                return None
            scene_size = min(
                np.ptp(self._mesh_points[:, 0]),
                np.ptp(self._mesh_points[:, 1]),
                np.ptp(self._mesh_points[:, 2])
            )
            if perp_dists[idx] > scene_size * 0.1:
                return None
            return self._mesh_points[idx]

        import threading
        self._right_lock = threading.Lock()

        def _on_right_press(obj, event):
            if not self._right_lock.acquire(blocking=False):
                return  # already executing, skip
            try:
                click_x, click_y = self.plotter.iren.interactor.GetEventPosition()
                pt = _find_nearest_to_click(click_x, click_y)
                if pt is not None:
                    vs = self.app.data.voxel_size_L0
                    shape = self.app.data.lab_full.shape
                    x_world = max(0, min(pt[0], (shape[2]-1)*vs[2]))
                    y_world = max(0, min(pt[1], (shape[1]-1)*vs[1]))
                    z_world = max(0, min(pt[2], (shape[0]-1)*vs[0]))
                    viewer = self.app.napari_view.viewer
                    current = list(viewer.dims.point)
                    current[0] = z_world
                    current[1] = y_world
                    current[2] = x_world
                    viewer.dims.point = tuple(current)
                    self.app._update_crosshair(*viewer.dims.current_step)
            finally:
                self._right_lock.release()

        self.plotter.iren.interactor.AddObserver('RightButtonPressEvent', _on_right_press)
        print("  ✓ Morphology: Right-click ray casting crosshair registered")

    def update_crosshair(self, z, y, x):
        """
        Update three orthogonal cutting planes in 3D view to match napari crosshair.
        z, y, x are in voxel coordinates (dims.current_step).
        """
        vs = self.app.data.voxel_size_L0
        z_um = z * vs[0]
        y_um = y * vs[1]
        x_um = x * vs[2]

        # Remove old crosshair actors
        for actor in self.crosshair_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.crosshair_actors = []

        # Determine bounds from loaded meshes (fallback to data shape)
        shape = self.app.data.lab_full.shape
        z_max = shape[0] * vs[0]
        y_max = shape[1] * vs[1]
        x_max = shape[2] * vs[2]

        plane_color = [0.0, 0.0, 0.0]  # black
        plane_opacity = 0.15
        border_width = 3
        border_opacity = 0.9

        def _add_border(center, i_size, j_size, i_dir, j_dir, color):
            cx, cy, cz = center
            ix, iy, iz = [v * i_size / 2 for v in i_dir]
            jx, jy, jz = [v * j_size / 2 for v in j_dir]
            corners = [
                (cx - ix - jx, cy - iy - jy, cz - iz - jz),
                (cx + ix - jx, cy + iy - jy, cz + iz - jz),
                (cx + ix + jx, cy + iy + jy, cz + iz + jz),
                (cx - ix + jx, cy - iy + jy, cz - iz + jz),
            ]
            for k in range(4):
                line = pv.Line(corners[k], corners[(k+1) % 4])
                a = self.plotter.add_mesh(line, color=color, line_width=border_width,
                                          opacity=border_opacity, lighting=False)
                self.crosshair_actors.append(a)

        # XY plane (axial) — constant Z, magenta border
        xy_plane = pv.Plane(
            center=(x_max / 2, y_max / 2, z_um),
            direction=(0, 0, 1),
            i_size=x_max, j_size=y_max,
            i_resolution=1, j_resolution=1
        )
        a1 = self.plotter.add_mesh(xy_plane, color=plane_color, opacity=plane_opacity,
                                    show_edges=False, lighting=False)
        self.crosshair_actors.append(a1)
        _add_border((x_max/2, y_max/2, z_um), x_max, y_max, (1,0,0), (0,1,0), 'magenta')

        # XZ plane (coronal) — constant Y, cyan border
        xz_plane = pv.Plane(
            center=(x_max / 2, y_um, z_max / 2),
            direction=(0, 1, 0),
            i_size=z_max, j_size=x_max,
            i_resolution=1, j_resolution=1
        )
        a2 = self.plotter.add_mesh(xz_plane, color=plane_color, opacity=plane_opacity,
                                    show_edges=False, lighting=False)
        self.crosshair_actors.append(a2)
        _add_border((x_max/2, y_um, z_max/2), z_max, x_max, (0,0,1), (1,0,0), 'cyan')

        # YZ plane (sagittal) — constant X, yellow border
        yz_plane = pv.Plane(
            center=(x_um, y_max / 2, z_max / 2),
            direction=(1, 0, 0),
            i_size=z_max, j_size=y_max,
            i_resolution=1, j_resolution=1
        )
        a3 = self.plotter.add_mesh(yz_plane, color=plane_color, opacity=plane_opacity,
                                    show_edges=False, lighting=False)
        self.crosshair_actors.append(a3)
        _add_border((x_um, y_max/2, z_max/2), z_max, y_max, (0,0,1), (0,1,0), 'yellow')

        # 3D crosshair indicator at current position
        arm = min(x_max, y_max, z_max) * 0.08
        for p0, p1, color in [
            ((x_um - arm, y_um, z_um), (x_um + arm, y_um, z_um), 'red'),
            ((x_um, y_um - arm, z_um), (x_um, y_um + arm, z_um), 'green'),
            ((x_um, y_um, z_um - arm), (x_um, y_um, z_um + arm), 'blue'),
        ]:
            line = pv.Line(p0, p1)
            a = self.plotter.add_mesh(
                line, color=color, line_width=8,
                lighting=False, render_lines_as_tubes=True,
                opacity=1.0
            )
            self.crosshair_actors.append(a)

        self.plotter.render()