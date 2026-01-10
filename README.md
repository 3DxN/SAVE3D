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

### Python Version Requirement

> ⚠️ **Important: Use Python 3.10 or 3.11**
>
> This ensures compatibility with GPU acceleration packages (RAPIDS: cucim, cuml, cugraph).
> - Python 3.9 and below: Newer versions of napari are not supported
> - Python 3.12+: RAPIDS is not fully supported yet

### One-Click Setup (Recommended)

```bash
# Windows (PowerShell)
.\setup_env.ps1          # CPU version
.\setup_env.ps1 -GPU     # GPU version

# macOS/Linux
chmod +x setup_env.sh
./setup_env.sh           # CPU version
./setup_env.sh --gpu     # GPU version
```

### Manual Setup with uv

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package manager, recommended for use.

```bash
# 1. Install uv (if not already installed)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Python 3.10 (uv will download automatically)
uv python install 3.10

# 3. Create virtual environment and install dependencies (CPU version)
uv venv .venv --python 3.10
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

uv pip install -r requirements.txt

# 4. (Optional) GPU accelerated version - requires NVIDIA GPU + CUDA
uv pip install -r requirements-gpu.txt
```

### Alternative: pip + venv

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) GPU acceleration
pip install -r requirements-gpu.txt
```

### Alternative: conda (for RAPIDS GPU acceleration)

If you need full RAPIDS GPU acceleration (including cuCIM, cuML, cuGraph), use conda:

```bash
# Create conda environment (must use Python 3.10 or 3.11)
conda create -n save3d python=3.10
conda activate save3d

# Install base dependencies
pip install -r requirements.txt

# Install RAPIDS (requires NVIDIA GPU + CUDA 12.x)
conda install -c rapidsai -c conda-forge -c nvidia \
    cupy cucim cuml cugraph cuda-version=12.0
```

### Python Version Compatibility Matrix

| Python | napari | pyvista | CuPy | RAPIDS | Recommendation |
|--------|--------|---------|------|--------|----------------|
| 3.9 | ✅ | ✅ | ✅ | ⚠️ Limited | ❌ Not recommended |
| **3.10** | ✅ | ✅ | ✅ | ✅ | ✅ **Recommended** |
| **3.11** | ✅ | ✅ | ✅ | ✅ | ✅ Supported |
| 3.12 | ✅ | ✅ | ✅ | ❌ Not supported | ❌ Not recommended |

### Verify Installation

```bash
# Check if installation was successful
python -c "import napari; import pyvista; import zarr; print('✓ All packages installed')"

# Check GPU support (optional)
python -c "import cupy; print(f'✓ CuPy GPU: {cupy.cuda.runtime.getDeviceCount()} devices')"
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Scientific computing |
| scikit-image | ≥0.21.0 | Image processing |
| tifffile | ≥2023.7.0 | TIFF I/O |
| napari | ≥0.4.18 | 2D visualization |
| pyvista | ≥0.42.0 | 3D visualization |
| pyvistaqt | ≥0.11.0 | PyVista Qt integration |
| zarr | ≥2.16.0 | Chunked array storage |
| dask | ≥2023.9.0 | Parallel computing |
| qtpy | ≥2.4.0 | Qt abstraction |
| pyqt5 | ≥5.15.0 | Qt backend |
| tqdm | ≥4.66.0 | Progress bars |

### Optional GPU Dependencies

| Package | Purpose |
|---------|---------|
| cupy-cuda12x | GPU-accelerated NumPy (CUDA 12.x) |
| cucim | GPU image processing (via conda, Linux only) |
| cuml | GPU machine learning (via conda, Linux only) |
| cugraph | GPU graph processing (via conda, Linux only) |

## GPU Acceleration Features

SAVE-3D preprocessing supports GPU acceleration via CuPy for significant speedups:

| Feature | GPU Acceleration | Speedup |
|---------|------------------|---------|
| Image pyramid building | ✅ Pipelined CUDA streams | 5-10x |
| Mask upsampling | ✅ Chunked GPU processing | 3-5x |
| Connected component labeling | ✅ cupyx.scipy.ndimage.label | 2-5x |
| Skeleton point mapping | ✅ Batch L0 CC pre-computation | 10-20x |
| np.unique on large volumes | ✅ CuPy unique | 2-5x |

> **Note:** GPU acceleration works on both Windows and Linux. CuPy provides cross-platform support, while RAPIDS (cucim, cuml, cugraph) is Linux-only.

## Author

Developed by **Huai-Ching Hsieh** and **Yang-Hsien Lin**, 2025–2026.
