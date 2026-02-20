"""
UI Control panel - Baseline (Point-based Navigation)
"""

from qtpy import QtWidgets, QtCore


def _create_control_panel(app):
    """
    Minimal control panel - just the label color legend.
    Sliders and buttons are added dynamically into the plugin controls box.
    """
    controls = QtWidgets.QWidget()
    controls.setMaximumHeight(30)
    main_layout = QtWidgets.QHBoxLayout(controls)
    main_layout.setSpacing(12)
    main_layout.setContentsMargins(4, 2, 4, 2)

    # --- Label Colors Legend ---
    if app.data.legend_labels:
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
            main_layout.addWidget(label_item)

    main_layout.addStretch()
    return controls