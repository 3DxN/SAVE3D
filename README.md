# SAVE-3D: Structure-Aware Visualization & Exploration for 3D Densely Labeled Tissue Images

A tri-view interactive visualization system for exploring dense 3D tissue images. SAVE-3D uses **2D contours/shapes** (instead of points) as the linking unit between 2D and 3D views, enabling structure-based navigation and bidirectional exploration.

> Unlike point-based 2D-3D linking, SAVE-3D enables **structure–structure correspondence** by using 2D connected components as the navigation unit. The 3D view is separated into **topology (skeleton)** and **morphology (geometry)** to reduce occlusion and improve understanding of connectivity.

## Project Structure

```
SAVE3D/
├── main_app.py              # Visualization entry point
├── main_preprocessing.py    # Preprocessing entry point
├── doc/                     # Documentation and diagrams
│   ├── save3d_preprocessing_pipeline.drawio
│   ├── save3d_preprocessing_pipeline.drawio.png
│   ├── save3d_ui_architecture.drawio
│   └── save3d_ui_architecture.drawio.png
├── preprocessing/           # Preprocessing modules
│   ├── config.py            # GPU detection, constants
│   ├── colors.py            # Color assignment (Golden Ratio method)
│   ├── loaders.py           # Volume loading (.tif, .npy)
│   ├── pyramid.py           # Multi-resolution pyramid (GPU-accelerated)
│   ├── transforms.py        # Mask upsampling
│   ├── skeleton.py          # Skeleton processing & graph building
│   ├── cc_analysis.py       # 2D CC precomputation & adjacency
│   ├── mesh_builders.py     # Prebuild VTK meshes
│   ├── zarr_writer.py       # OME-NGFF Zarr output
│   └── utils.py             # Utilities
└── save3d_app/              # Visualization application
    ├── app.py               # Main application class
    ├── controls.py          # UI callbacks
    ├── state.py             # Application state
    ├── data/                # Data loading from Zarr
    ├── core/                # Core algorithms (CC propagation, mesh building)
    ├── views/               # View controllers (Napari, Skeleton, Morphology)
    └── modes/               # Host modes (ImageHost, SkeletonHost)
```

## Workflow Overview

### Preprocessing Pipeline

> See the flowchart in **`doc/save3d_preprocessing_pipeline.drawio`** for a visual overview.

1. **Load Image @ L0** — Load original resolution 3D image volume  

2. **Build Image Pyramid** — GPU-accelerated multi-resolution pyramid (L0 → L1 → L2 → L3)  

3. **Load Masks @ L2** — Load outer/inner segmentation masks with auto color assignment  

4. **Skeleton Intersection Analysis** — Compute skeleton overlap ratios with each label  

5. **Create Combined Labels** — Merge masks and upsample L2 → L0  

6. **Extract Skeleton Points** — Build KD-Tree data with 26-connectivity neighbors  

7. **Build Skeleton Graph** — Compute endpoints, bifurcations, and total length  

8. **Prebuild Meshes** — Generate VTK meshes for all visualization modes  
   - Morphology meshes (Navigation mode)  
   - Instance meshes (Instance mode)  
   - Image host meshes (per 3D-CC)  

9. **Precompute 2D CC & Adjacency** — Per z-slice CC properties and overlap relationships  

10. **Write Zarr** — Output OME-NGFF format with image/label pyramids  

**Output files:**
- `output.zarr/` — Main data (image + labels + masks)
- `metadata.json` — Resolution, colors, label names
- `cc_metadata.json` — 2D CC precomputed data (critical for runtime)
- `skeleton_points.json` — Skeleton points for KD-Tree
- `skeleton_graph.json` — Skeleton graph structure
- `morphology_meshes/` — Navigation mode meshes
- `skeleton_instance_meshes/` — Instance mode meshes
- `image_host_meshes/` — Image host mode meshes

### Visualization Application

> See the diagram in **`doc/save3d_ui_architecture.drawio`** for the UI architecture.

**Tri-View Design:**
| View | Purpose |
|------|---------|
| 2D Image View (Napari) | Raw image texture, label overlay, slice navigation |
| 3D Topology View | Skeleton backbone, color-coded by label, sphere widget |
| 3D Morphology View | Surface mesh rendering, outer/inner structures |

**Host Modes (Navigation Driver):**

| Mode | Direction | Description |
|------|-----------|-------------|
| Image Host | 2D → 3D | Double-click 2D shape to navigate; Napari drives |
| Skeleton Host | 3D → 2D | Drag sphere widget on skeleton; 3D drives |

**Morphology View Modes:**

| Image Host Modes | Skeleton Host Modes |
|------------------|---------------------|
| 0: Instance (single 3D CC) | 0: Instance (skeleton-connected) |
| 1: Label (all objects of label) | 1: Navigation (sphere + local morph) |
| 2: Z Navigation (follow overlap) | 2: Selection (draw to select region) |

## Usage

```bash
# Step 1: Preprocessing (configure paths in main_preprocessing.py)
python main_preprocessing.py

# Step 2: Launch visualization
python main_app.py output.zarr
```

## Input Data Requirements

| Data | Resolution | Format | Description |
|------|------------|--------|-------------|
| 3D Image | L0 (original) | .tif / .npy | Raw tissue image |
| Outer Masks | L2 (4x downsampled) | .tif | Segmentation masks per label |
| Inner Masks | L2 (4x downsampled) | .tif | Optional inner structure masks |
| Skeleton | L2 (4x downsampled) | .tif | 3D skeleton/centerline |

## Environment Setup

*(To be added)*

## Dependencies

- Python 3.10+
- napari
- pyvista / pyvistaqt
- zarr / dask
- scikit-image
- scipy
- CuPy (optional, for GPU acceleration)

## Author

Developed by **Huai-Ching Hsieh** and **Yang-Hsien Lin**, 2025–2026.  
