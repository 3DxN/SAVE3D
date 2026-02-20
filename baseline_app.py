"""
SAVE-3D Baseline Application Entry Point (Point-based Navigation)

Usage:
    python baseline_app.py path/to/data.zarr

Or in Python:
    from baseline_app import main
    main('path/to/data.zarr')
"""

import sys
import os
from pathlib import Path

import vtk
vtk.vtkObject.GlobalWarningDisplayOff()


def main(zarr_path=None):
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'

    from qtpy import QtWidgets, QtCore

    existing_app = QtWidgets.QApplication.instance()
    if not existing_app:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = existing_app

    app.setApplicationName("SAVE-3D Baseline: Point-based Navigation")

    if zarr_path is None:
        if len(sys.argv) > 1:
            zarr_path = sys.argv[1]
        else:
            zarr_path = 'prostate_pathology.zarr'

    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        QtWidgets.QMessageBox.critical(
            None,
            "Baseline Error",
            f"Zarr data not found:\n{zarr_path}\n\n"
            f"Usage: python baseline_app.py <path_to_zarr>"
        )
        return 1

    try:
        print(f"\n{'='*60}")
        print(f"  SAVE-3D Baseline: Point-based Navigation")
        print(f"{'='*60}")
        print(f"  Data: {zarr_path}")
        print(f"{'='*60}\n")

        from baseline_save3d_app import BaselineViewer

        viewer = BaselineViewer(zarr_path)
        viewer.show()

        return app.exec_()

    except ImportError as e:
        QtWidgets.QMessageBox.critical(
            None,
            "Baseline Import Error",
            f"Failed to import baseline modules:\n{str(e)}\n\n"
            f"Make sure baseline_save3d_app/ package is in the same directory."
        )
        print(f"Import Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        QtWidgets.QMessageBox.critical(
            None,
            "Baseline Error",
            f"Application failed to start:\n{str(e)}"
        )
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())