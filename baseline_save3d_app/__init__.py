"""
SAVE-3D Baseline: Point-based Navigation

Package structure:
    baseline_app.py
    baseline_save3d_app/
    ├── app.py              # BaselineViewer class
    ├── controls.py         # UI controls
    ├── state.py            # Centralized state
    ├── utils.py            # Utility functions
    │
    ├── data/               # Data loading
    │   └── loader.py
    │
    └── views/              # View setup
        ├── napari_view.py
        ├── morphology_view.py
        └── control_panel.py
"""

# Suppress VTK warning messages (non-fatal OpenGL context warnings on Windows)
# These "wglMakeCurrent failed" errors occur during window operations but don't affect functionality
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

from .app import BaselineViewer

__all__ = ['BaselineViewer']
