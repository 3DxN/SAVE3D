"""
Data Loading Module - Baseline
"""

from pathlib import Path
import json
import numpy as np
import dask.array as da
import zarr
import pyvista as pv


class DataLoader:
    """
    Data loader - loads all data and stores as attributes
    """

    def __init__(self):
        self.zarr_path = None

        # === Metadata ===
        self.metadata = None
        self.voxel_size_L0 = None
        self.voxel_size_L2 = None
        self.processing_level = None
        self.downsample_factor = None
        self.label_names = None
        self.label_colors = None
        self.legend_labels = None
        self.has_inner_mask = False

        # === Zarr Data ===
        self.img_full = None
        self.lab_full = None
        self.img_levels = []
        self.lab_levels = []
        self.outer_masks_L2 = {}
        self.inner_masks_L2 = {}

        # === Global Morphology Meshes ===
        self.global_morph_meshes = {}
        self.morph_meshes_loaded = False

    def load_all(self, zarr_path):
        """Load all data. Returns self for chaining."""
        self.zarr_path = Path(zarr_path)
        self._load_metadata()
        self._load_zarr_data()
        return self

    def _load_metadata(self):
        """Load metadata.json"""
        metadata_path = self.zarr_path.parent / "metadata.json"

        if not metadata_path.exists():
            raise ValueError(f"metadata.json not found")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        if 'resolution_levels' not in self.metadata:
            raise ValueError("metadata.json must have 'resolution_levels' key. Run new preprocessing.")

        self.voxel_size_L0 = tuple(self.metadata['resolution_levels']['L0']['voxel_size_um'])
        self.voxel_size_L2 = tuple(self.metadata['resolution_levels']['L2']['voxel_size_um'])
        self.processing_level = self.metadata['processing_level']
        self.downsample_factor = self.metadata['downsample_factor']
        self.label_colors = self.metadata['label_colors']
        self.label_names = self.metadata['label_names']
        self.legend_labels = self.metadata.get('legend_labels', self.label_names)
        self.has_inner_mask = self.metadata.get('has_inner_mask', True)

        print(f"[OK] Metadata: {len(self.label_names)} labels")
        print(f"  Voxel size @ L0: {self.voxel_size_L0} μm")
        print(f"  Voxel size @ L2: {self.voxel_size_L2} μm")
        print(f"  Has inner mask: {self.has_inner_mask}")

    def _load_zarr_data(self):
        """Load zarr arrays (image, labels, masks)"""
        print(f"Loading Zarr: {self.zarr_path}")

        store = zarr.open(str(self.zarr_path), mode='r')

        # Histology @ L0
        level_keys = sorted([int(k) for k in store.keys() if k.isdigit()])
        self.img_levels = []
        for level in level_keys:
            self.img_levels.append(da.from_zarr(store[str(level)]))
        self.img_full = self.img_levels[0]
        print(f"Histology @ L0: {self.img_full.shape}")

        # Combined labels @ L0
        if 'segmentation' not in store:
            raise ValueError("Zarr must have 'segmentation' group")
        seg_group = store['segmentation']
        self.lab_levels = []
        for level in level_keys:
            if str(level) in seg_group:
                self.lab_levels.append(da.from_zarr(seg_group[str(level)]))
        self.lab_full = self.lab_levels[0]
        print(f"Combined labels @ L0: {self.lab_full.shape}")

        # Outer masks @ L2
        if 'outer_masks_L2' not in store:
            raise ValueError("Zarr must have 'outer_masks_L2' group. Run new preprocessing.")
        self.outer_masks_L2 = {}
        outer_group = store['outer_masks_L2']
        for label_name in self.label_names:
            if label_name in outer_group:
                self.outer_masks_L2[label_name] = da.from_zarr(outer_group[label_name])
        print(f"Outer @ L2: {list(self.outer_masks_L2.keys())}")

        # Inner masks @ L2 (optional)
        self.inner_masks_L2 = {}
        if 'inner_masks_L2' in store and self.has_inner_mask:
            inner_group = store['inner_masks_L2']
            for label_name in self.label_names:
                if label_name in inner_group:
                    self.inner_masks_L2[label_name] = da.from_zarr(inner_group[label_name])
            print(f"Inner @ L2: {list(self.inner_masks_L2.keys())}")
        else:
            print(f"Inner @ L2: None (outer-only mode)")

    def _load_global_morphology_meshes(self):
        """Load prebuilt morphology meshes"""
        if self.morph_meshes_loaded:
            return

        mesh_dir = self.zarr_path.parent / "morphology_meshes"
        info_path = mesh_dir / "mesh_info.json"

        if not info_path.exists():
            print("[WARN] morphology_meshes/mesh_info.json not found")
            print("       Run preprocessing with prebuild_morphology_meshes()")
            return

        print("\n=== Loading Global Morphology Meshes ===")

        with open(info_path, 'r') as f:
            mesh_info = json.load(f)

        for mesh_name, info in mesh_info['meshes'].items():
            mesh_path = mesh_dir / info['path']

            if mesh_path.exists():
                mesh = pv.read(str(mesh_path))
                self.global_morph_meshes[mesh_name] = mesh
                print(f"  ✓ {mesh_name}: {mesh.n_points:,} vertices")
            else:
                print(f"  ✗ {mesh_name}: file not found")

        self.morph_meshes_loaded = True
        print(f"✓ Loaded {len(self.global_morph_meshes)} meshes")

