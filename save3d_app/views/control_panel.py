"""
UI Control panel creation
"""

from qtpy import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor 
import numpy as np
from ..controls import _on_host_mode_changed, _on_plane_toggle, _on_contour_toggle, _on_opacity_changed, _on_camera_sync_toggle, _update_visibility, _reset_2d_view
from ..controls import _center_on_component, _center_on_component, _reset_views, _clear_tracking
# =============================================================================
# CONTROL PANEL CREATION
# =============================================================================

def _create_control_panel(app):
    """
    Create control panel with all widgets
    
    Returns:
        QWidget: control panel widget
    """
    controls = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(controls)
    main_layout.setSpacing(4)
    main_layout.setContentsMargins(4, 4, 4, 4)
    
    # === Row 1: 4 Column Headers (Toggle Buttons) ===
    headers_widget = QtWidgets.QWidget()
    headers_layout = QtWidgets.QHBoxLayout(headers_widget)
    headers_layout.setSpacing(4)
    headers_layout.setContentsMargins(0, 0, 0, 0)
    
    app.toggle_2d = QtWidgets.QPushButton("▼ 2D View")
    app.toggle_3d = QtWidgets.QPushButton("▼ 3D Views")
    app.toggle_skeleton = QtWidgets.QPushButton("▼ Skeleton View")
    app.toggle_morph = QtWidgets.QPushButton("▼ Morphology View")
    
    for btn in [app.toggle_2d, app.toggle_3d, app.toggle_skeleton, app.toggle_morph]:
        btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                padding: 4px;
                background-color: #d0d0d0;
                border: 1px solid #999;
            }
            QPushButton:hover {
                background-color: #c0c0c0;
            }
        """)
        headers_layout.addWidget(btn, stretch=1)
    
    main_layout.addWidget(headers_widget)
    
    # === Content Area: Grid Layout ===
    app.content_widget = QtWidgets.QWidget()
    app.content_grid = QtWidgets.QGridLayout(app.content_widget)
    app.content_grid.setSpacing(4)
    app.content_grid.setContentsMargins(4, 4, 4, 4)
    
    # Set equal column stretch
    for col in range(4):
        app.content_grid.setColumnStretch(col, 1)
    
    # --- Row 0: Checkboxes ---
    app.show_boundary_chk = QtWidgets.QCheckBox('2D CC Boundary')
    app.show_boundary_chk.setChecked(True)
    app.show_boundary_chk.stateChanged.connect(lambda: _update_visibility(app))
    app.content_grid.addWidget(app.show_boundary_chk, 0, 0)
    
    app.sync_camera_chk = QtWidgets.QCheckBox('Sync Camera Rotation')
    app.sync_camera_chk.setChecked(False)
    app.sync_camera_chk.stateChanged.connect(lambda: _on_camera_sync_toggle(app))
    app.content_grid.addWidget(app.sync_camera_chk, 0, 1)
    
    app.show_marker_chk = QtWidgets.QCheckBox('Skeleton Marker')
    app.show_marker_chk.setChecked(True)
    app.show_marker_chk.stateChanged.connect(lambda: _update_visibility(app))
    app.content_grid.addWidget(app.show_marker_chk, 0, 2)
    
    app.show_plane_chk = QtWidgets.QCheckBox('Guide Plane')
    app.show_plane_chk.setChecked(False)
    app.show_plane_chk.stateChanged.connect(lambda: _on_plane_toggle(app))
    app.content_grid.addWidget(app.show_plane_chk, 0, 3)
    
    # --- Row 1: Buttons / Checkboxes ---
    app.reset_2d_btn = QtWidgets.QPushButton('Reset 2D View')
    app.reset_2d_btn.clicked.connect(lambda: _reset_2d_view(app))
    app.content_grid.addWidget(app.reset_2d_btn, 1, 0)
    
    app.reset_btn = QtWidgets.QPushButton('Reset 3D Views')
    app.reset_btn.clicked.connect(lambda: _reset_views(app))
    app.content_grid.addWidget(app.reset_btn, 1, 1)
    
    # Empty cell for Skeleton View (col 2)
    
    app.show_plane_contour_chk = QtWidgets.QCheckBox('Guide Plane CC Contour')
    app.show_plane_contour_chk.setChecked(True)
    app.show_plane_contour_chk.stateChanged.connect(lambda: _on_contour_toggle(app))
    app.content_grid.addWidget(app.show_plane_contour_chk, 1, 3)
    
    # --- Row 2: Buttons / Checkboxes ---
    app.center_btn = QtWidgets.QPushButton('Center on 2D CC')
    app.center_btn.clicked.connect(lambda: _center_on_component(app, from_button=True))
    app.content_grid.addWidget(app.center_btn, 2, 0)
    
    app.clear_btn = QtWidgets.QPushButton('Clear Tracking')
    app.clear_btn.clicked.connect(lambda: _clear_tracking(app))
    app.content_grid.addWidget(app.clear_btn, 2, 1)
    
    # Empty cell for Skeleton View (col 2)
    
    # Image Host Morphology Mode (row 2, col 3)
    app.image_morph_mode_widget = QtWidgets.QWidget()
    image_morph_layout = QtWidgets.QHBoxLayout(app.image_morph_mode_widget)
    image_morph_layout.setSpacing(2)
    image_morph_layout.setContentsMargins(0, 0, 0, 0)

    app.image_morph_mode_group = QtWidgets.QButtonGroup(app)

    app.image_instance_radio = QtWidgets.QRadioButton("Instance")
    app.image_label_radio = QtWidgets.QRadioButton("Label")
    app.image_label_radio.setChecked(True)  # Label
    app.image_znavigation_radio = QtWidgets.QRadioButton("Z Navigation")

    app.image_morph_mode_group.addButton(app.image_instance_radio, 0)
    app.image_morph_mode_group.addButton(app.image_label_radio, 1)
    app.image_morph_mode_group.addButton(app.image_znavigation_radio, 2)

    app.image_morph_mode_group.buttonClicked.connect(
        lambda btn: app.image_host._on_morph_mode_changed_image(btn)
    )

    image_morph_layout.addWidget(app.image_instance_radio)
    image_morph_layout.addWidget(app.image_label_radio)
    image_morph_layout.addWidget(app.image_znavigation_radio)

    app.content_grid.addWidget(app.image_morph_mode_widget, 3, 3)
    
    # --- Row 3: Slider ---
    slider_widget = QtWidgets.QWidget()
    slider_layout = QtWidgets.QVBoxLayout(slider_widget)
    slider_layout.setSpacing(1)
    slider_layout.setContentsMargins(0, 0, 0, 0)
    
    opacity_label = QtWidgets.QLabel('Labels Opacity')
    app.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    app.opacity_slider.setRange(0, 100)
    app.opacity_slider.setValue(50)
    app.opacity_slider.valueChanged.connect(lambda v: _on_opacity_changed(app, v))
    
    slider_layout.addWidget(opacity_label)
    slider_layout.addWidget(app.opacity_slider)
    app.content_grid.addWidget(slider_widget, 3, 0)
    
    main_layout.addWidget(app.content_widget)
    
    # --- Row 3: Morphology Mode (Skeleton Host only) ---
    app.morph_mode_widget = QtWidgets.QWidget()
    morph_mode_layout = QtWidgets.QVBoxLayout(app.morph_mode_widget)
    morph_mode_layout.setSpacing(2)
    morph_mode_layout.setContentsMargins(0, 0, 0, 0)

    app.morph_mode_group = QtWidgets.QButtonGroup(app)

    app.morph_instance_radio = QtWidgets.QRadioButton("Instance")
    app.morph_navigation_radio = QtWidgets.QRadioButton("Navigation")
    app.morph_navigation_radio.setChecked(True)  
    app.morph_selection_radio = QtWidgets.QRadioButton("Selection")

    app.morph_mode_group.addButton(app.morph_instance_radio, 0)
    app.morph_mode_group.addButton(app.morph_navigation_radio, 1)
    app.morph_mode_group.addButton(app.morph_selection_radio, 2)

    app.morph_mode_group.buttonClicked.connect(
        lambda btn: app.skeleton_host._on_morph_mode_changed_skeleton(btn)
    )

    # Selection mode buttons (only visible when Selection is active)
    app.selection_buttons_widget = QtWidgets.QWidget()
    selection_btn_layout = QtWidgets.QHBoxLayout(app.selection_buttons_widget)
    selection_btn_layout.setSpacing(4)
    selection_btn_layout.setContentsMargins(0, 4, 0, 0)
    
    app.draw_selection_btn = QtWidgets.QPushButton("🖌️ Draw")
    app.draw_selection_btn.setCheckable(True)
    app.draw_selection_btn.clicked.connect(
        lambda: app.skeleton_host._on_draw_selection_toggle()
    )
    
    app.clear_selection_btn = QtWidgets.QPushButton("🗑️ Clear")
    app.clear_selection_btn.clicked.connect(
        lambda: app.skeleton_host._on_clear_selection()
    )
    
    selection_btn_layout.addWidget(app.draw_selection_btn)
    selection_btn_layout.addWidget(app.clear_selection_btn)
    
    app.selection_buttons_widget.setVisible(False)  
    morph_mode_layout.addWidget(app.selection_buttons_widget)

    morph_mode_layout.addWidget(app.morph_instance_radio)
    morph_mode_layout.addWidget(app.morph_navigation_radio)
    morph_mode_layout.addWidget(app.morph_selection_radio)

    app.morph_mode_widget.setVisible(False)

    app.content_grid.addWidget(app.morph_mode_widget, 3, 3)

    # === Toggle Logic ===
    app.col_visible = [True, True, True, True]  # Track visibility
    
    def make_toggle(col_idx, btn):
        def toggle():
            app.col_visible[col_idx] = not app.col_visible[col_idx]
            visible = app.col_visible[col_idx]
            
            # Update button text
            titles = ["2D View", "3D Views", "Skeleton View", "Morphology View"]
            btn.setText(f"{'▼' if visible else '▶'} {titles[col_idx]}")
            
            # Show/hide all widgets in this column
            for row in range(app.content_grid.rowCount()):
                item = app.content_grid.itemAtPosition(row, col_idx)
                if item and item.widget():
                    item.widget().setVisible(visible)
        
        return toggle
    
    app.toggle_2d.clicked.connect(make_toggle(0, app.toggle_2d))
    app.toggle_3d.clicked.connect(make_toggle(1, app.toggle_3d))
    app.toggle_skeleton.clicked.connect(make_toggle(2, app.toggle_skeleton))
    app.toggle_morph.clicked.connect(make_toggle(3, app.toggle_morph))
    
    # === Row: Host Mode ===
    host_widget = QtWidgets.QWidget()
    host_layout = QtWidgets.QHBoxLayout(host_widget)
    host_layout.setContentsMargins(0, 4, 0, 0)
    host_layout.setSpacing(8)
    
    host_label = QtWidgets.QLabel("Host Mode:")
    host_label.setStyleSheet("font-weight: bold;")
    
    app.host_mode_group = QtWidgets.QButtonGroup(app)
    app.image_host_radio = QtWidgets.QRadioButton("Image Host (Napari)")
    app.image_host_radio.setChecked(True)
    app.host_mode_group.addButton(app.image_host_radio, 0)
    
    app.skeleton_host_radio = QtWidgets.QRadioButton("Skeleton Host (3D)")
    app.host_mode_group.addButton(app.skeleton_host_radio, 1)
    
    app.host_mode_group.buttonClicked.connect(lambda btn: _on_host_mode_changed(app, btn))
    
    host_layout.addWidget(host_label)
    host_layout.addWidget(app.image_host_radio)
    host_layout.addWidget(app.skeleton_host_radio)
    host_layout.addStretch()
    
    main_layout.addWidget(host_widget)
    
    # === Row: Label Colors (only show legend_labels) ===
    if app.data.legend_labels:
        colors_widget = QtWidgets.QWidget()
        colors_layout = QtWidgets.QHBoxLayout(colors_widget)
        colors_layout.setContentsMargins(0, 0, 0, 0)
        colors_layout.setSpacing(12)
        
        for label_name in app.data.legend_labels:
            color_hex = app.data.label_colors[label_name]
            
            label_item = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(label_item)
            item_layout.setSpacing(4)
            item_layout.setContentsMargins(0, 0, 0, 0)
            
            color_circle = QtWidgets.QLabel()
            color_circle.setFixedSize(14, 14)
            color_circle.setStyleSheet(f"""
                QLabel {{
                    background-color: {color_hex};
                    border: 1px solid #333;
                    border-radius: 7px;
                }}
            """)
            
            label_text = QtWidgets.QLabel(label_name)
            
            item_layout.addWidget(color_circle)
            item_layout.addWidget(label_text)
            colors_layout.addWidget(label_item)
        
        colors_layout.addStretch()
        main_layout.addWidget(colors_widget)
    
    # === Row: Slice Info ===
    app.slice_info = QtWidgets.QLabel("Slice: 0 / 0  |  Z = 0.0 μm  |  No CC tracked")
    app.slice_info.setStyleSheet("""
        QLabel {
            font-weight: bold;
            padding: 4px;
            background-color: #e0e0e0;
        }
    """)
    main_layout.addWidget(app.slice_info)
    
    return controls
