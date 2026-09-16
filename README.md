# SAVE-3D: Structure-Aware Visualization & Exploration for 3D Densely Labeled Tissue Images

A tri-view interactive visualization system for exploring 3D densely labeled tissue images. SAVE-3D employs **2D contours** as the linking unit between 2D and 3D views, enabling **structure-aware navigation** with reduced degrees of freedom. The 3D representation is decomposed into a **skeleton view (topological overview)** and a **morphology view (local structural detail)** to avoid occlusion while retaining access to detailed geometry. **Two complementary host modes** support bidirectional exploration between 2D cross-sectional images and 3D renderings.

## Project Structure

```
SAVE3D/
├── main_preprocessing.py    # Preprocessing entry point
├── main_app.py              # Visualization entry point
├── baseline_app.py          # Baseline viewer entry point (point-based navigation)
├── skeletonization.py       # Optional: generate a skeleton volume with kimimaro
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
│   ├── cc_analysis.py       # 2D contour precomputation & adjacency
│   ├── mesh_builders.py     # Prebuild VTK meshes
│   ├── zarr_writer.py       # OME-NGFF Zarr output
│   └── utils.py             # Utilities
├── save3d_app/              # Visualization application
│   ├── app.py               # Main application class
│   ├── controls.py          # UI callbacks
│   ├── state.py             # Application state
│   ├── data/                # Data loading from Zarr
│   ├── core/                # Core algorithms (contour propagation, mesh building)
│   ├── views/               # View controllers (Napari, Skeleton, Morphology)
│   └── modes/               # Host modes (ImageHost, SkeletonHost)
└── baseline_save3d_app/     # Baseline viewer (no skeleton view, no host modes)
```

## Workflow Overview

### Preprocessing Pipeline

> See the flowchart in **`doc/save3d_preprocessing_pipeline.drawio`** for a visual overview.

1. **Load Image @ L0** — Load original resolution 3D image volume  

2. **Build Image Pyramid** — GPU-accelerated multi-resolution pyramid (L0 → L1 → L2 → L3)  

3. **Load Masks @ L2** — Load outer/inner segmentation masks with auto color (if not pre-assigned)  

4. **Skeleton Intersection Analysis** — Assign skeleton labels by computing the intersection with segmentation masks

5. **Create Combined Labels** — Merge masks and upsample L2 → L0  

6. **Extract Skeleton Points** — Build KD-Tree data

7. **Build Skeleton Graph** — Compute endpoints, bifurcations, and total length (not used in current version)  

8. **Prebuild Meshes** — Generate VTK meshes for most visualization modes  

9. **Precompute 2D contours & Adjacency** — Per z-slice contours properties and overlap relationships  

10. **Write Zarr** — Output OME-NGFF format with image/label pyramids  

**Output files:**
- `output.zarr/` — Main data (image + labels + masks)
- `metadata.json` — Resolution, colors, label names
- `cc_metadata.json` — 2D contours precomputed data (critical for runtime)
- `skeleton_points.json` — Skeleton points for KD-Tree
- `skeleton_graph.json` — Skeleton graph structure
- `skeleton_meshes/` — Per-spatial-component skeleton meshes for the 3D topology view
- `morphology_meshes/` — Per-label meshes for image-host label mode and skeleton-host navigation mode
- `skeleton_instance_meshes/` — Meshes for skeleton-host instance mode
- `image_host_meshes/` — Meshes for image-host instance mode

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
| 0: Instance (3D object of the same label) | 0: Instance (same label, skeleton-connected) |
| 1: Label (all objects sharing the same label) | 1: Navigation (dynamic radius around widget) |
| 2: Z Navigation (overlap-based propagation path) | 2: Selection (brushing along skeleton) |

## Input Data Requirements

You supply four volumes, all shaped **(Z, Y, X)**. This repository ships **no sample data and no download script**.

| Data | Resolution | Format | Description |
|------|------------|--------|-------------|
| 3D Image | L0 (original) | `.tif` / `.tiff` / `.npy` | Raw tissue image. Grayscale or RGB |
| Outer Masks | L2 (4x downsampled) | `.tif` | **Required.** Segmentation masks (see the two formats below) |
| Inner Masks | L2 (4x downsampled) | `.tif` | Optional inner/lumen structure masks |
| Skeleton | L2 (4x downsampled) | `.tif` | 3D skeleton/centerline; any nonzero voxel is skeleton |

**Shape contract.** The masks and the skeleton must be *exactly* `L0_shape // DOWNSAMPLE_FACTOR` (4 by default). A mismatch only prints a warning while loading, but the upsampled result is checked strictly and the run aborts with `ValueError: Combined labels size mismatch!`.

**4D TIFFs** are treated as RGB: `(Z, Y, X, C)` is kept as-is and `(Z, C, Y, X)` is transposed, in both cases keeping the first three channels (`preprocessing/loaders.py`).

### Two mask formats

Choose one with `USE_SEPARATE_MASKS` in `main_preprocessing.py`:

**Format A — one binary TIFF per label** (`USE_SEPARATE_MASKS = True`)

```python
OUTER_MASKS_L2 = {'Benign': 'outer_benign.tif', 'HGPIN': 'outer_hgpin.tif'}
INNER_MASKS_L2 = {'Benign': 'inner_benign.tif', 'HGPIN': 'inner_hgpin.tif'}  # {} for none
```

The dict keys are the label names, and all of them appear in the control-panel legend.

**Format B — one integer label volume** (`USE_SEPARATE_MASKS = False`)

```python
LABEL_OUTER_PATH = 'labels.tif'   # voxel values: 0 = background, 1, 2, 3, ...
LABEL_INNER_PATH = None           # optional
LABEL_DICT = {1: 'Benign', 2: 'HGPIN'}   # {} to auto-name everything
```

Every nonzero value in the volume becomes a label. Values listed in `LABEL_DICT` take that name and appear in the legend; the rest are auto-named `Label_{value}` and are omitted from the legend.

**Colors.** `USER_LABEL_COLORS` may be partial or empty — any label without an explicit hex color is assigned one automatically using a golden-ratio hue sequence.

## Usage

### Step 0 (optional) — Generate a skeleton

Skip this if you already have a skeleton volume. `skeletonization.py` runs kimimaro TEASAR over a mask TIFF and writes a `uint8` volume with skeleton voxels set to 255. It has no command-line interface — edit `tiff_path`, `output_tiff_path`, `voxel_spacing`, and the `teasar_params` block at the top of the file. The input mask must already be at L2.

```bash
python skeletonization.py
```

### Step 1 — Preprocessing

```bash
python main_preprocessing.py
```

There is **no command-line interface and no config file**: every setting is a Python constant inside the `CONFIGURATION` block of `main()` in `main_preprocessing.py`. Edit that block before running. All input paths are **relative to the current working directory**, so run the script from the directory holding your TIFFs.

| Constant | Default | Meaning |
|----------|---------|---------|
| `IMAGE_PATH` | `IDC-P_8x_111325.tif` | L0 image |
| `VOXEL_SIZE_L0` | `(0.9667*2,) * 3` μm | Physical voxel size; drives every world coordinate |
| `PYRAMID_LEVELS` | `4` | Depth of the image pyramid |
| `PROCESSING_LEVEL` | `2` | Sets `DOWNSAMPLE_FACTOR = 2 ** N` (= 4) |
| `SKELETON_L2` | `output_skeleton_32x_092625.tiff` | Skeleton volume @ L2 |
| `OUT_ZARR` | `prostate_pathology10_gpu.zarr` | Output store |
| `USE_SEPARATE_MASKS` | `True` | Format A (`True`) vs Format B (`False`) |
| `OUTER_MASKS_L2`, `INNER_MASKS_L2` | 4 prostate labels | Format A mask paths |
| `LABEL_OUTER_PATH`, `LABEL_INNER_PATH`, `LABEL_DICT` | crypt mask | Format B mask paths |
| `USER_LABEL_COLORS` | 4 hex colors | Partial or empty is fine |

GPU acceleration is detected automatically through CuPy; set `FORCE_CPU_MODE = True` in `preprocessing/config.py` to disable it.

### Output layout — keep everything together

All artifacts are written next to the store, into `Path(OUT_ZARR).parent`. If `OUT_ZARR` is a bare name, they land in the working directory as siblings of the `.zarr`. The application resolves every sidecar as `zarr_path.parent / <name>`, so **the `.zarr` directory and all of its sidecars must stay in the same folder**.

```
<data-dir>/
├── out.zarr/                    # image pyramid "0".."3", segmentation/,
│                                #   outer_masks_L2/, inner_masks_L2/
├── metadata.json                # [required] resolution levels, label names/colors,
│                                #   legend labels, has_inner_mask, skeleton intersections
├── cc_metadata.json             # [required] per-label per-z contours, skeleton
│                                #   markers, overlap relationships
├── skeleton_points.json         # [required] KD-Tree points, labels, cc/instance ids
├── skeleton_graph.json          # written, but not read by the current application
├── image_host_meshes.json       # per-label 3D-component mesh index
├── skeleton_meshes/             # skeleton_cc{N}_{label}.vtk + skeleton_mesh_info.json
├── morphology_meshes/           # outer_{label}.vtk, inner_{label}.vtk + mesh_info.json
├── skeleton_instance_meshes/    # inst_{id:04d}_{outer,inner}.vtk
└── image_host_meshes/<label>/   # cc_{id:04d}_{outer,inner}.vtk
```

The three `[required]` files cause startup to fail if missing, as do the `segmentation` and `outer_masks_L2` groups inside the store. A missing mesh directory only logs a warning and degrades the corresponding view.

### Step 2 — Launch the viewer

```bash
python main_app.py path/to/out.zarr
```

The zarr path is the only argument; there are no flags. Omitting it falls back to `prostate_pathology.zarr`, and a path that does not exist produces an error dialog and exit code 1.

A second viewer is included for comparison against point-based navigation:

```bash
python baseline_app.py path/to/out.zarr
```

It reads the same store and the same sidecars (minus the skeleton-specific ones), has no skeleton view and no host modes, and renders pyramid level L1 rather than L0.

If the project is installed as a package, `save3d-preprocess` and `save3d-app` are equivalent console entry points.

## Environment Setup

### Python Version Requirement

> **Important: Use Python 3.10 or 3.11**

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

## Author

Developed by **Huai-Ching Hsieh** and **Yang-Hsien Lin**, 2025–2026.  
