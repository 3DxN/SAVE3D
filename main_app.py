"""
SAVE-3D Application Entry Point

Structure:
    main_app.py              ← This file (entry point)
    save3d_app/
    ├── __init__.py
    ├── app.py               # SAVE3DViewer class
    ├── controls.py          # UI callbacks
    ├── state.py             # Centralized state
    ├── utils.py             # Utility functions
    ├── data/                # Data loading
    ├── core/                # Core processing
    ├── views/               # View setup
    └── modes/               # Host modes

Usage:
    python main_app.py path/to/data.zarr
    
Or in Python:
    from main_app import main
    main('path/to/data.zarr')
"""

import sys
import os
from pathlib import Path


def main(zarr_path=None):
    """
    Main entry point for SAVE-3D application
    
    Args:
        zarr_path: Path to zarr data directory. If None, uses command line arg or default.
    
    Returns:
        int: Exit code (0 for success)
    """
    # === High DPI settings (must be set before QApplication) ===
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
    
    # === Import Qt (after setting env vars) ===
    from qtpy import QtWidgets, QtCore
    
    # === Get or create QApplication ===
    existing_app = QtWidgets.QApplication.instance()
    
    if not existing_app:
        # Set high DPI attributes before creating app
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = existing_app
    
    app.setApplicationName("SAVE-3D: 3-Panel 3D Viewer")
    
    # === Determine zarr path ===
    if zarr_path is None:
        if len(sys.argv) > 1:
            zarr_path = sys.argv[1]
        else:
            # Default path - modify as needed
            zarr_path = 'prostate_pathology.zarr'
    
    zarr_path = Path(zarr_path)
    
    # === Validate path ===
    if not zarr_path.exists():
        QtWidgets.QMessageBox.critical(
            None, 
            "SAVE-3D Error", 
            f"Zarr data not found:\n{zarr_path}\n\n"
            f"Usage: python main_app.py <path_to_zarr>"
        )
        return 1
    
    # === Launch application ===
    try:
        print(f"\n{'='*60}")
        print(f"  SAVE-3D: Structure-Aware Visualization & Exploration")
        print(f"  for 3D Densely Labeled Tissue Images")
        print(f"{'='*60}")
        print(f"  Data: {zarr_path}")
        print(f"{'='*60}\n")
        
        # Import from package
        from save3d_app import SAVE3DViewer
        
        # Create and show viewer
        viewer = SAVE3DViewer(zarr_path)
        viewer.show()
        
        # Run event loop
        return app.exec_()
        
    except ImportError as e:
        QtWidgets.QMessageBox.critical(
            None, 
            "SAVE-3D Import Error", 
            f"Failed to import SAVE-3D modules:\n{str(e)}\n\n"
            f"Make sure save3d_app/ package is in the same directory."
        )
        print(f"Import Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            None, 
            "SAVE-3D Error", 
            f"Application failed to start:\n{str(e)}"
        )
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
