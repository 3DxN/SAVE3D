"""
Skeleton 3D View 
NOTE: Only setup function here. All operations are in modes/*.
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtWidgets, QtCore
from scipy.spatial import KDTree

from ..utils import _hex_to_rgb
from ..controls import _set_active_view


class SkeletonViewController:
    """
    Skeleton 3D view 
    """
    
    def __init__(self, app):
        """
        Initialize with app reference
        """
        self.app = app
        
        # PyVista components (set during setup)
        self.plotter = None
        
        # Actors
        self.skeleton_actors = {}
        self.black_marker_actor = None
        self.bounding_box_actor = None
        
        # Scale bar actors
        self.skeleton_scale_bar_actor = None
        self.skeleton_scale_text_actor = None
        
        # Widgets
        self.sphere_widget = None
        self.selection_sphere_widget = None
        
        # Selection highlight
        self.selection_highlight_actor = None
        
        # Info label (set during setup)
        self.skeleton_info = None
    
    def _setup_skeleton_panel(self):
        """Setup Skeleton view"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        
        info = QtWidgets.QGroupBox("Skeleton View @ L2")
        info_layout = QtWidgets.QVBoxLayout(info)
        
        self.skeleton_info = QtWidgets.QLabel("Color-coded skeleton")
        self.skeleton_info.setWordWrap(True)
        info_layout.addWidget(self.skeleton_info)
        
        layout.addWidget(info)
        
        self.plotter = QtInteractor(self.app)
        self.plotter.enable_trackball_style()
        self.plotter.set_background([0.95, 0.95, 0.95])
        
        try:
            self.plotter.add_axes(color='black', line_width=2)
        except:
            pass
        
        self.plotter.add_text(
            'Skeleton View',
            position='upper_left',
            font_size=10,
            color='black'
        )
        
        # Camera above, looking down
        self.plotter.camera.position = (0, 0, -1000)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, -1, 0)  # Y axis points down

        layout.addWidget(self.plotter.interactor, stretch=1)
        
        # Set focus on click
        self.plotter.interactor.mousePressEvent = lambda e: _set_active_view(self.app, 'skeleton', e)
        
        # Disable VTK default keyboard events (avoid arrow keys being overridden by zoom/rotate)
        try:
            self.plotter.iren.remove_observers("KeyPressEvent")
            self.plotter.iren.remove_observers("CharEvent")
        except:
            pass

        # Reset actors
        self.skeleton_actors = {}
        self.black_marker_actor = None
        self.skeleton_scale_bar_actor = None
        self.skeleton_scale_text_actor = None
        
        return container
    
    def _initialize_skeleton_view(self):
        """Initialize skeleton view with ALL meshes"""
        print("\n=== Initializing Skeleton View ===")
        
        app = self.app
        data = app.data
        
        # Enable anti-aliasing
        try:
            self.plotter.enable_anti_aliasing()
            print("  ✓ Anti-aliasing enabled")
        except Exception as e:
            print(f"  ⚠ Anti-aliasing not available: {e}")
        
        # Add enhanced lighting
        try:
            z_flip = 1
            # Light 1: main light from upper right front
            light1 = pv.Light(
                position=(10, 10, 10 * z_flip), 
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.8
            )
            
            # Light 2: fill light from lower left front
            light2 = pv.Light(
                position=(-10, -10, 10 * z_flip), 
                focal_point=(0, 0, 0),
                color='white',
                intensity=0.4
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
            print("  ✓ Enhanced lighting added (3 lights)")
        except Exception as e:
            print(f"  ⚠ Custom lighting not available: {e}")

        for label_name in data.label_names:
            if label_name not in data.skeleton_meshes:
                continue
            
            mesh = data.skeleton_meshes[label_name]
            color = _hex_to_rgb(data.label_colors[label_name])
            
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=1.0,
                smooth_shading=True,
                show_edges=False,
                metallic=0.0,
                specular=0.1,
                name=f'skeleton_{label_name}'
            )
            
            self.skeleton_actors[label_name] = actor
            print(f"  ✓ {label_name}: {mesh.n_points:,} pts")
        
        # Add bounding box
        self._add_full_image_bounding_box()
        
        try:
            self.plotter.reset_camera()
        except:
            pass
        
        self.plotter.render()

        # === Sphere Widget (initially OFF) ===
        initial_center = data.skeleton_coords[0]
        
        # Create callback with closure over app
        def on_sphere_drag(center, widget):
            dist, idx = data.skeleton_kdtree.query(center)
            nearest_pos = data.skeleton_coords[idx]
            widget.SetCenter(*nearest_pos)
            app.state.sphere_position = nearest_pos.copy()
            
            if app.state.host_mode == 'skeleton':
                app.skeleton_host._on_sphere_drag_lightweight(nearest_pos)
        
        skeleton_radius_voxels = 3
        skeleton_radius_um = skeleton_radius_voxels * data.voxel_size_L2[0]
        sphere_radius = skeleton_radius_um * 3.0  # 3x skeleton thickness

        self.sphere_widget = self.plotter.add_sphere_widget(
            callback=on_sphere_drag,
            center=initial_center,
            radius=sphere_radius,
            color='black',
            interaction_event='always',
            pass_widget=True,
        )
        
        try:
            prop = self.sphere_widget.GetSphereProperty()
            if prop:
                prop.SetOpacity(0.5)
        except:
            pass
    
        # Initially off (Image Host mode uses marker_actor)
        self.sphere_widget.Off()
        
        # === Right-click Jump (only in Skeleton Host) ===
        def on_right_click_pick(point):
            """Handle right-click on skeleton to jump sphere"""
            if point is None or app.state.host_mode != 'skeleton':
                return
    
            skeleton_host = app.skeleton_host
            
            # If in selection drawing mode
            if skeleton_host.selection_drawing:
                # Only accumulate mode allows jumping
                if not getattr(skeleton_host, 'selection_accumulate_mode', False):
                    return  # Single segment mode, no jumping
                
                dist, idx = data.skeleton_kdtree.query(point)
                nearest_pos = data.skeleton_coords[idx]
                
                if skeleton_host.selection_sphere_widget:
                    skeleton_host.selection_sphere_widget.SetCenter(*nearest_pos)
                    skeleton_host.last_drawing_idx = None
                    skeleton_host._allow_jump = True  # ✅ RESET positioning phase
                    self.plotter.render()
                return
            
            # If there's a selection, restrict to yellow region only
            if skeleton_host.selected_skeleton_indices:
                # Only find nearest point within selected region
                selected_list = list(skeleton_host.selected_skeleton_indices)
                selected_coords = data.skeleton_coords[selected_list]
                selected_tree = KDTree(selected_coords)
                
                dist, local_idx = selected_tree.query(point)
                global_idx = selected_list[local_idx]
                nearest_pos = data.skeleton_coords[global_idx]
            else:
                # No selection restriction, entire skeleton
                dist, idx = data.skeleton_kdtree.query(point)
                nearest_pos = data.skeleton_coords[idx]
            
            if self.sphere_widget:
                self.sphere_widget.SetCenter(*nearest_pos)
                self.sphere_widget.On()
                try:
                    self.sphere_widget.GetSphereProperty().SetOpacity(0.5)
                except:
                    pass

            app.state.sphere_position = nearest_pos.copy()
            skeleton_host._on_sphere_position_changed(nearest_pos)
            self.plotter.render()
        
        self.plotter.enable_point_picking(
            callback=on_right_click_pick,
            show_message=False,
            show_point=False,
        )
        
        print(f"[OK] Skeleton view initialized with Host Mode awareness")

    def _add_skeleton_scale_bar(self, x_max, y_max, z_max):
        """
        Add scale bar to skeleton view OUTSIDE the bounding box
        
        Places a 1mm scale bar outside the lower edge, right-aligned
        """
        try:
            # Scale bar length: 1mm = 1000 μm
            scale_length_um = 1000.0
            
            # Position OUTSIDE box: below the box, right-aligned
            offset_from_box = 300  # μm outside the box (below)
            offset_from_right = 200  # μm from right edge
            z_position = z_max / 2  # Middle height of the box
            
            # Right-align the scale bar
            bar_end_x = x_max - offset_from_right  # Right end near box edge
            bar_start_x = bar_end_x - scale_length_um  # Left end
            bar_y = y_max + offset_from_box  # Below the box
            bar_z = z_position
            
            # Create scale bar line
            scale_bar_points = np.array([
                [bar_start_x, bar_y, bar_z],
                [bar_end_x, bar_y, bar_z]
            ])
            
            scale_bar = pv.Line(scale_bar_points[0], scale_bar_points[1])
            
            self.skeleton_scale_bar_actor = self.plotter.add_mesh(
                scale_bar,
                color='black',
                line_width=3,
                render_lines_as_tubes=True,
                name='skeleton_scale_bar'
            )
            
            # Add end caps (small spheres at both ends)
            cap_radius = 15.0  # μm
            
            start_cap = pv.Sphere(radius=cap_radius, center=scale_bar_points[0])
            end_cap = pv.Sphere(radius=cap_radius, center=scale_bar_points[1])
            
            self.plotter.add_mesh(start_cap, color='black', name='scale_bar_cap_start')
            self.plotter.add_mesh(end_cap, color='black', name='scale_bar_cap_end')
            
            # Add text label BELOW the line, centered on the bar
            label_x = (bar_start_x + bar_end_x) / 2
            label_y = bar_y + 400  # BELOW the bar
            label_z = bar_z
            
            self.skeleton_scale_text_actor = self.plotter.add_point_labels(
                points=[[label_x, label_y, label_z]],
                labels=['1 mm'],
                font_size=10,
                text_color='black',
                point_size=0.1,
                render_points_as_spheres=False,
                always_visible=True,
                name='skeleton_scale_text'
            )
            
            print(f"  ✓ Scale bar added: 1 mm @ ({bar_end_x:.0f}, {bar_y:.0f}, {bar_z:.0f}) [right-aligned below box]")
            
        except Exception as e:
            print(f"  ✗ Scale bar creation failed: {e}")
            import traceback
            traceback.print_exc()

     
    def _add_full_image_bounding_box(self):
        """Add bounding box @ L2 world space"""
        data = self.app.data
        
        if data.label_names[0] in data.outer_masks_L2:
            shape_L2 = data.outer_masks_L2[data.label_names[0]].shape
            
            x_max = (shape_L2[2] - 1) * data.voxel_size_L2[2]
            y_max = (shape_L2[1] - 1) * data.voxel_size_L2[1]
            z_max = (shape_L2[0] - 1) * data.voxel_size_L2[0]
            
            bounds = (0, x_max, 0, y_max, 0, z_max)
            
            try:
                bbox = pv.Box(bounds=bounds)
                self.plotter.add_mesh(
                    bbox,
                    style='wireframe',
                    color='gray',
                    opacity=0.3,
                    line_width=2
                )
                print(f"  ✓ Bounding box added")
            except:
                pass
            
            self._add_skeleton_scale_bar(x_max, y_max, z_max)
