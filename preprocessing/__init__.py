"""
SAVE-3D Preprocessing Pipeline

Package structure:
    preprocessing/
    ├── config.py          # GPU detection, constants
    ├── colors.py          # Color utilities
    ├── loaders.py         # Data loading (volume, labels)
    ├── pyramid.py         # Image pyramid building (GPU/CPU)
    ├── transforms.py      # Mask upsampling, transforms
    ├── skeleton.py        # Skeleton processing & graph
    ├── cc_analysis.py     # 2D CC precomputation
    ├── mesh_builders.py   # All mesh generation
    ├── zarr_writer.py     # Zarr output
    └── utils.py           # Progress tracker, helpers

Usage:
    python main_preprocessing.py
"""

from .config import GPU_AVAILABLE, FORCE_CPU_MODE
from .colors import hex_to_rgb, rgb_to_hex, assign_colors
from .loaders import load_volume, load_labels_from_volume
from .pyramid import build_pyramid_with_gpu_local_mean
from .transforms import upsample_mask_nearest
from .skeleton import (
    preprocess_skeleton_intersections,
    build_skeleton_graph,
    extract_skeleton_points_for_kdtree,
    compute_skeleton_labels
)
from .cc_analysis import precompute_2d_cc_and_adjacency
from .mesh_builders import (
    prebuild_morphology_meshes,
    prebuild_instance_meshes,
    prebuild_image_host_meshes,
    prebuild_skeleton_meshes
)
from .zarr_writer import write_zarr_pathology
from .utils import SimpleProgress

__all__ = [
    # Config
    'GPU_AVAILABLE',
    'FORCE_CPU_MODE',
    # Colors
    'hex_to_rgb',
    'rgb_to_hex', 
    'assign_colors',
    # Loaders
    'load_volume',
    'load_labels_from_volume',
    # Pyramid
    'build_pyramid_with_gpu_local_mean',
    # Transforms
    'upsample_mask_nearest',
    # Skeleton
    'preprocess_skeleton_intersections',
    'build_skeleton_graph',
    'extract_skeleton_points_for_kdtree',
    'compute_skeleton_labels',
    # CC Analysis
    'precompute_2d_cc_and_adjacency',
    # Mesh Builders
    'prebuild_morphology_meshes',
    'prebuild_instance_meshes',
    'prebuild_image_host_meshes',
    'prebuild_skeleton_meshes',
    # Zarr
    'write_zarr_pathology',
    # Utils
    'SimpleProgress',
]
