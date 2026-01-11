"""
Data Loading Module
"""

from pathlib import Path
import json
import numpy as np
import dask.array as da
import zarr
import pyvista as pv
from scipy.spatial import KDTree
import tifffile


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
        
        # === Skeleton ===
        self.skeleton_meshes = {}
        self.skeleton_coords = None
        self.skeleton_coords_voxel = None
        self.skeleton_labels = None
        self.skeleton_cc_ids = None
        self.skeleton_z_L0 = None
        self.skeleton_instance_ids = None
        self.skeleton_kdtree = None
        self.skeleton_instances = []
        self.skeleton_neighbors = []      # ← 新增
        
        # === CC Metadata ===
        self.cc_metadata = None
        
        # === Image Host Meshes ===
        self.image_host_mesh_info = None
        
        # === Global Morphology Meshes ===
        self.global_morph_meshes = {}
        self.morph_meshes_loaded = False
    
    def load_all(self, zarr_path):
        """Load all data. Returns self for chaining."""
        self.zarr_path = Path(zarr_path)
        self._load_metadata()
        self._load_zarr_data()
        self._load_skeleton_meshes()
        self._load_cc_metadata()
        return self
    
    def _load_metadata(self):
        """
        Load metadata.json 
        """
        metadata_path = self.zarr_path.parent / "metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"metadata.json not found")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # New format with resolution_levels (required)
        if 'resolution_levels' not in self.metadata:
            raise ValueError("metadata.json must have 'resolution_levels' key. Run new preprocessing.")
        
        self.voxel_size_L0 = tuple(self.metadata['resolution_levels']['L0']['voxel_size_um'])
        self.voxel_size_L2 = tuple(self.metadata['resolution_levels']['L2']['voxel_size_um'])
        self.processing_level = self.metadata['processing_level']
        self.downsample_factor = self.metadata['downsample_factor']
        self.label_colors = self.metadata['label_colors']
        self.label_names = self.metadata['label_names']
        self.legend_labels = self.metadata.get('legend_labels', self.label_names)  # Backward compat: show all
        self.has_inner_mask = self.metadata.get('has_inner_mask', True)  # Default True for backward compat
        
        print(f"[OK] Metadata: {len(self.label_names)} labels")
        print(f"  Voxel size @ L0: {self.voxel_size_L0} μm")
        print(f"  Voxel size @ L2: {self.voxel_size_L2} μm")
        print(f"  Processing level: L{self.processing_level} ({self.downsample_factor}x)")
        print(f"  Has inner mask: {self.has_inner_mask}")
        print(f"  Legend labels: {len(self.legend_labels)} (for control panel)")
        if self.legend_labels:
            print(f"\n[METADATA] Legend colors:")
            for label_name in self.legend_labels:
                color = self.label_colors.get(label_name, 'NOT FOUND')
                print(f"  {label_name}: {color}")
        else:
            print(f"\n[METADATA] No legend (auto-named labels)")
        print()
        
        # Load skeleton points
        skeleton_points_path = self.zarr_path.parent / "skeleton_points.json"
        with open(skeleton_points_path, 'r') as f:
            skeleton_points = json.load(f)

        self.skeleton_coords = np.array(skeleton_points['coords_world'])
        self.skeleton_labels = np.array(skeleton_points['labels'])
        self.skeleton_kdtree = KDTree(self.skeleton_coords)
        self.skeleton_cc_ids = np.array(skeleton_points['cc_ids'], dtype=np.int32)
        self.skeleton_z_L0 = np.array(skeleton_points['z_L0'], dtype=np.int32)
        self.skeleton_coords_voxel = np.array(skeleton_points['coords_voxel'], dtype=np.int32)
        self.skeleton_neighbors = skeleton_points.get('neighbors', [])
        self.skeleton_instance_ids = np.array(skeleton_points.get('instance_ids', []), dtype=np.int32)
        self.skeleton_instances = skeleton_points.get('instances', [])
        print(f"[OK] Skeleton KD-Tree: {len(self.skeleton_coords)} points")

        self.global_morph_meshes = {}  # {mesh_name: pv.PolyData}
        self.morph_meshes_loaded = False

        # Load image host mesh info
        image_host_mesh_path = self.zarr_path.parent / "image_host_meshes.json"
        if image_host_mesh_path.exists():
            with open(image_host_mesh_path, 'r') as f:
                self.image_host_mesh_info = json.load(f)
            print(f"[OK] Image host mesh info loaded")
        else:
            self.image_host_mesh_info = None
            print("[WARN] image_host_meshes.json not found")

    def _load_zarr_data(self):
        """
        Load zarr arrays (image, labels, masks)
        """
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
        
        # Outer masks @ L2 (required)
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

    def _load_skeleton_meshes(self):
        """
        Load skeleton meshes for all labels
        
        First tries to load prebuilt meshes from skeleton_meshes/
        Falls back to generating meshes on-the-fly if prebuilt not available
        """
        print("\n=== Loading Skeleton Meshes @ L2 ===")
        
        self.skeleton_meshes = {}
        
        # === Try loading prebuilt meshes first (FAST PATH) ===
        skeleton_mesh_dir = self.zarr_path.parent / "skeleton_meshes"
        skeleton_mesh_info_path = skeleton_mesh_dir / "skeleton_mesh_info.json"
        
        if skeleton_mesh_info_path.exists():
            print("  Found prebuilt skeleton meshes, loading...")
            try:
                with open(skeleton_mesh_info_path, 'r') as f:
                    mesh_info = json.load(f)
                
                for label_name, info in mesh_info.get('meshes', {}).items():
                    mesh_path = self.zarr_path.parent / info['path']
                    if mesh_path.exists():
                        mesh = pv.read(str(mesh_path))
                        self.skeleton_meshes[label_name] = mesh
                        print(f"    ✓ {label_name}: {mesh.n_points:,} pts, {mesh.n_cells:,} faces")
                    else:
                        print(f"    ✗ {label_name}: mesh file not found")
                
                if self.skeleton_meshes:
                    print(f"[OK] Loaded {len(self.skeleton_meshes)} prebuilt skeleton meshes (FAST)")
                    return
            except Exception as e:
                print(f"  ⚠ Failed to load prebuilt meshes: {e}")
                print("  Falling back to on-the-fly generation...")
        
        # === Fallback: Generate meshes on-the-fly (SLOW PATH) ===
        print("  No prebuilt meshes found, generating on-the-fly...")
        print("  (Run preprocessing with prebuild_skeleton_meshes() for faster startup)")
        
        if 'skeleton_intersections' not in self.metadata:
            print("[WARN] No skeleton intersection data")
            return
        
        # Load skeleton
        skeleton_L2 = None
        skeleton_path_candidates = [
            self.zarr_path.parent / "output_skeleton_32x_092625.tiff",
            self.zarr_path.parent / "output_skeleton_32x_092625.tif",
        ]
        
        for skeleton_path in skeleton_path_candidates:
            if skeleton_path.exists():
                skeleton_L2 = tifffile.imread(str(skeleton_path))
                print(f"  Loaded skeleton: {skeleton_path}")
                break
        
        if skeleton_L2 is None:
            print("[WARN] Skeleton file not found")
            return
        
        print(f"  Skeleton shape @ L2: {skeleton_L2.shape}")
        
        skeleton_binary = skeleton_L2 > 0
        total_voxels = np.sum(skeleton_binary)
        print(f"  Total skeleton voxels: {total_voxels:,}")
        
        if total_voxels == 0:
            return
        
        # Recreate label assignments
        skeleton_labels = np.zeros_like(skeleton_binary, dtype=np.uint8)
        
        print("  Recreating label assignments...")
        for label_idx, label_name in enumerate(self.label_names, start=1):
            if label_name not in self.outer_masks_L2:
                continue
            
            available = (skeleton_binary > 0) & (skeleton_labels == 0)
            outer_mask = self.outer_masks_L2[label_name][:].compute()
            intersection = available & (outer_mask > 0)
            skeleton_labels[intersection] = label_idx
            
            voxel_count = np.sum(intersection)
            print(f"    {label_name}: {voxel_count:,} voxels")
        
        # Generate meshes
        print("  Generating meshes...")
        for label_idx, label_name in enumerate(self.label_names, start=1):
            label_mask = (skeleton_labels == label_idx)
            
            if np.sum(label_mask) == 0:
                print(f"    {label_name}: No voxels, skipping")
                continue
            
            try:
                from skimage.morphology import binary_dilation, ball
                label_mask_dilated = binary_dilation(label_mask, ball(3))
            except:
                label_mask_dilated = label_mask
            
            data_transposed = np.transpose(label_mask_dilated, (2, 1, 0))
            
            try:
                grid = pv.ImageData(
                    dimensions=data_transposed.shape,
                    spacing=self.voxel_size_L2,
                    origin=(0, 0, 0)
                )
                
                grid.point_data['values'] = data_transposed.ravel(order='F')
                mesh = grid.contour(isosurfaces=[0.5])
                
                if mesh.n_cells > 0:
                    self.skeleton_meshes[label_name] = mesh
                    n_faces = mesh.n_faces_strict if hasattr(mesh, 'n_faces_strict') else mesh.n_cells
                    print(f"    ✓ {label_name}: {mesh.n_points:,} pts, {n_faces:,} faces")
                
            except Exception as e:
                print(f"    ✗ {label_name}: mesh creation failed - {e}")
        
        print(f"[OK] Generated {len(self.skeleton_meshes)} skeleton meshes (on-the-fly)")

    def _load_cc_metadata(self):
        """
        Load CC metadata
        
        """
        cc_metadata_path = self.zarr_path.parent / "cc_metadata.json"
        
        if not cc_metadata_path.exists():
            print("[ERROR] cc_metadata.json not found!")
            raise ValueError("CC metadata required")
        
        with open(cc_metadata_path, 'r') as f:
            self.cc_metadata = json.load(f)
        
        print(f"[OK] CC metadata loaded")
        
        # Verify structure
        for label_id in self.cc_metadata['labels']:
            label_data = self.cc_metadata['labels'][label_id]
            for z_str in label_data['layers']:
                layer = label_data['layers'][z_str]
                for cc in layer['outer']:
                    assert 'skeleton_marker' in cc
                    assert 'position' in cc['skeleton_marker']
                    assert 'has_skeleton' in cc['skeleton_marker']
        
        print("[OK] CC metadata validation passed")
    
    def _load_global_morphology_meshes(self):
        """
        Lazy load prebuilt morphology meshes for Skeleton Host Mode 2
        Called once when first switching to Skeleton Host

        """
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

