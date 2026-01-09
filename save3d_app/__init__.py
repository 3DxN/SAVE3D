"""
Structure-Aware Visualization & Exploration for 3D densely labeled  tissue image (SAVE-3D)

Package structure:
    main_app.py
    save3d_app/
    ├── app.py              # Main app class
    ├── controls.py         # All UI controls
    ├── state.py             # Centralized state (Level 0)
    ├── utils.py             # Utility functions (Level 0)
    │
    ├── data/                # Data loading (Level 1)
    │   └── loader.py
    │
    ├── core/                # Core processing (Level 2)
    │   ├── mesh_builder.py
    │   └── cc_propagation.py
    │
    ├── views/               # View setup 
    │   ├── napari_view.py      
    │   ├── skeleton_view.py    
    │   ├── morphology_view.py  
    │   └── control_panel.py         
    │
    └── modes/               # All host modes logic 
        ├── image_host.py       
        └── skeleton_host.py    

"""

from .app import SAVE3DViewer

__all__ = ['SAVE3DViewer']
