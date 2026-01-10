"""
Mesh Building Functions

Prebuild meshes for different visualization modes:
- Morphology meshes for Skeleton Host
- Instance meshes for skeleton instances
- Image Host meshes for CC-based navigation

Supports GPU acceleration via CuPy when available, with CPU fallback.
- Multi-threaded mesh generation
- Batch lookup table computation
- Pre-computed 2D CC labels
"""

import os
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import label as scipy_cc_label
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Configure PyVista for offscreen rendering BEFORE importing pyvista
# This prevents "wglMakeCurrent failed" errors on Windows
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN'] = '1'

import pyvista as pv
pv.OFF_SCREEN = True

# Suppress VTK error output (the wglMakeCurrent errors are non-fatal)
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

from .config import GPU_AVAILABLE, get_cupy

# GPU-accelerated CC labeling via CuPy
CUPY_LABEL_AVAILABLE = False
_cupy_label = None

if GPU_AVAILABLE:
    try:
        cp = get_cupy()
        if cp is not None:
            from cupyx.scipy.ndimage import label as cupy_label
            _cupy_label = cupy_label
            CUPY_LABEL_AVAILABLE = True
    except ImportError:
        pass


def _cc_label_2d(mask, use_gpu=True):
    """2D Connected Component labeling with GPU acceleration if available"""
    if use_gpu and CUPY_LABEL_AVAILABLE:
        cp = get_cupy()
        if cp is not None:
            try:
                mask_gpu = cp.asarray(mask)
                labeled_gpu, num_features = _cupy_label(mask_gpu)
                labeled = cp.asnumpy(labeled_gpu)
                del mask_gpu, labeled_gpu
                return labeled, int(num_features)
            except Exception:
                pass  # Fall back to CPU
    
    return scipy_cc_label(mask)


def _cc_label_3d(mask, use_gpu=True):
    """
    3D Connected Component labeling with GPU acceleration if available
    
    For large 3D volumes, we use chunked processing to avoid GPU OOM.
    Falls back to CPU if GPU memory is insufficient.
    
    Parameters:
    -----------
    mask : np.ndarray
        3D binary mask
    use_gpu : bool
        Whether to try GPU acceleration
    
    Returns:
    --------
    labeled : np.ndarray
        Labeled array (same shape as mask)
    num_features : int
        Number of connected components
    """
    if use_gpu and CUPY_LABEL_AVAILABLE:
        cp = get_cupy()
        if cp is not None:
            try:
                # Check if mask fits in GPU memory (with safety margin)
                mask_bytes = mask.nbytes
                device = cp.cuda.Device()
                free_mem = device.mem_info[0]
                
                # Need ~3x memory for: input + labeled output + intermediate
                required_mem = mask_bytes * 4  # int32 output + overhead
                
                if required_mem < free_mem * 0.7:  # Use 70% of free memory
                    mask_gpu = cp.asarray(mask.astype(np.uint8))
                    labeled_gpu, num_features = _cupy_label(mask_gpu)
                    labeled = cp.asnumpy(labeled_gpu)
                    del mask_gpu, labeled_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    return labeled, int(num_features)
                else:
                    print(f"      3D mask too large for GPU ({mask_bytes/1e9:.1f}GB), using CPU")
            except cp.cuda.memory.OutOfMemoryError:
                print(f"      GPU OOM for 3D CC labeling, using CPU")
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"      GPU 3D CC labeling failed: {e}, using CPU")
    
    # CPU fallback
    return ndimage.label(mask)


def _create_mesh_from_mask(mask_3d, voxel_size, origin_offset=(0, 0, 0)):
    """
    Unified mesh creation function
    
    Parameters:
    -----------
    mask_3d : np.ndarray
        Binary mask (Z, Y, X)
    voxel_size : tuple
        (voxel_z, voxel_y, voxel_x) in μm
    origin_offset : tuple
        (offset_x, offset_y, offset_z) in μm
    
    Returns:
    --------
    pv.PolyData or None
    """
    if mask_3d.sum() == 0:
        return None
    
    nz, ny, nx = mask_3d.shape
    pad = 2
    
    # Padding
    padded = np.zeros((nz + 2*pad, ny + 2*pad, nx + 2*pad), dtype=np.float32)
    padded[pad:pad+nz, pad:pad+ny, pad:pad+nx] = mask_3d.astype(np.float32)
    
    # Transpose to (X, Y, Z) for PyVista
    data_t = np.transpose(padded, (2, 1, 0))
    
    # Origin (considering padding and offset)
    origin = (
        origin_offset[0] - pad * voxel_size[2],  # X
        origin_offset[1] - pad * voxel_size[1],  # Y
        origin_offset[2] - pad * voxel_size[0],  # Z
    )
    
    try:
        grid = pv.ImageData(
            dimensions=data_t.shape,
            spacing=(voxel_size[2], voxel_size[1], voxel_size[0]),  # (dx, dy, dz)
            origin=origin
        )
        grid.point_data['values'] = data_t.ravel(order='F')
        mesh = grid.contour(isosurfaces=[0.5])
        
        return mesh if mesh.n_points > 0 else None
    except Exception as e:
        print(f"    Mesh creation failed: {e}")
        return None


def prebuild_instance_meshes(instances_info, skeleton_points_data, 
                              outer_masks_L2, inner_masks_L2, label_names, 
                              voxel_size_L2, output_dir):
    """
    Prebuild mesh (outer + inner) for each skeleton instance
    
    GPU-ACCELERATED: Uses CuPy for 2D CC labeling when available
    
    Parameters:
    -----------
    instances_info : list
        List of instance dicts from extract_skeleton_points_for_kdtree
    skeleton_points_data : dict
        Full skeleton points data
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    inner_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    label_names : list
        Label names
    voxel_size_L2 : tuple
        Voxel size @ Level 2
    output_dir : Path
        Output directory
    """
    mesh_dir = Path(output_dir) / "skeleton_instance_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    print(f"\n=== Prebuilding Skeleton Instance Meshes @ L2 ===")
    print(f"  CC Labeling: {'GPU (CuPy)' if CUPY_LABEL_AVAILABLE else 'CPU (scipy)'}")
    
    cp = get_cupy() if CUPY_LABEL_AVAILABLE else None
    
    for inst in tqdm(instances_info, desc="Building instance meshes"):
        inst_id = inst['id']
        label_name = inst['label_name']
        point_indices = inst['point_indices']
        z_range_L2 = inst['z_range_L2']
        
        # Get outer mask for this label
        outer_mask = outer_masks_L2[label_name]
        if hasattr(outer_mask, 'compute'):
            outer_mask = outer_mask.compute()
        
        inner_mask = inner_masks_L2.get(label_name, None)
        if inner_mask is not None and hasattr(inner_mask, 'compute'):
            inner_mask = inner_mask.compute()
        
        # Z slice (with buffer)
        z_min = max(0, z_range_L2[0] - 1)
        z_max = min(outer_mask.shape[0], z_range_L2[1] + 2)
        
        coords_voxel = skeleton_points_data['coords_voxel']
        nz = z_max - z_min
        sub_outer = np.zeros((nz, outer_mask.shape[1], outer_mask.shape[2]), dtype=bool)
        
        for z_L2 in range(z_min, z_max):
            slice_mask = outer_mask[z_L2] > 0
            if not slice_mask.any():
                continue
            
            # Find skeleton points on this z (as seeds)
            seeds_yx = [(coords_voxel[idx][1], coords_voxel[idx][2]) 
                        for idx in point_indices if coords_voxel[idx][0] == z_L2]
            
            if not seeds_yx:
                continue
            
            # Label 2D CCs with GPU acceleration
            labeled, _ = _cc_label_2d(slice_mask, use_gpu=CUPY_LABEL_AVAILABLE)
            for y, x in seeds_yx:
                cc_label_val = labeled[y, x]
                if cc_label_val > 0:
                    sub_outer[z_L2 - z_min] |= (labeled == cc_label_val)
        
        if sub_outer.sum() == 0:
            inst['outer_mesh'] = None
            inst['inner_mesh'] = None
            continue
        
        # Origin offset (due to sub-volume extraction)
        origin_offset = (0, 0, z_min * voxel_size_L2[0])
        
        # === Outer Mesh ===
        outer_mesh = _create_mesh_from_mask(sub_outer, voxel_size_L2, origin_offset)
        
        if outer_mesh is not None:
            outer_file = f"inst_{inst_id:04d}_outer.vtk"
            outer_mesh.save(str(mesh_dir / outer_file))
            inst['outer_mesh'] = f"skeleton_instance_meshes/{outer_file}"
        else:
            inst['outer_mesh'] = None
        
        # === Inner Mesh ===
        if inner_mask is not None:
            sub_inner = (inner_mask[z_min:z_max] > 0) & sub_outer
            inner_mesh = _create_mesh_from_mask(sub_inner, voxel_size_L2, origin_offset)
            
            if inner_mesh is not None:
                inner_file = f"inst_{inst_id:04d}_inner.vtk"
                inner_mesh.save(str(mesh_dir / inner_file))
                inst['inner_mesh'] = f"skeleton_instance_meshes/{inner_file}"
            else:
                inst['inner_mesh'] = None
        else:
            inst['inner_mesh'] = None
        
        # Clean up temporary data
        if 'point_indices' in inst:
            del inst['point_indices']
    
    # Final GPU cleanup
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    
    print(f"✓ Skeleton instance meshes complete")


def _build_single_mesh(cc_id, labeled_3d, outer_mask, inner_mask, voxel_size_L2, 
                        label_mesh_dir, label_name, nz, ny, nx, pad=2):
    """
    Build mesh for a single 3D CC (can be called in parallel)
    """
    cc_mask_outer = (labeled_3d == cc_id)
    voxel_count = np.sum(cc_mask_outer)
    
    if voxel_count < 10:
        return None
    
    # Z range
    z_coords = np.where(cc_mask_outer.any(axis=(1, 2)))[0]
    z_min, z_max = int(z_coords.min()), int(z_coords.max())
    
    mesh_info = {
        'id': cc_id,
        'z_range': [z_min, z_max],
        'voxel_count': int(voxel_count),
        'outer_mesh': None,
        'inner_mesh': None,
    }
    
    origin = (
        -pad * voxel_size_L2[2],
        -pad * voxel_size_L2[1],
        -pad * voxel_size_L2[0]
    )
    
    # === Outer Mesh ===
    try:
        padded = np.zeros((nz + 2*pad, ny + 2*pad, nx + 2*pad), dtype=np.uint8)
        padded[pad:pad+nz, pad:pad+ny, pad:pad+nx] = cc_mask_outer
        
        data_t = np.transpose(padded.astype(float), (2, 1, 0))
        
        grid = pv.ImageData(
            dimensions=data_t.shape,
            spacing=voxel_size_L2,
            origin=origin
        )
        grid.point_data['values'] = data_t.ravel(order='F')
        mesh = grid.contour(isosurfaces=[0.5])
        
        if mesh.n_points > 0:
            outer_file = f"cc_{cc_id:04d}_outer.vtk"
            mesh.save(str(label_mesh_dir / outer_file))
            mesh_info['outer_mesh'] = f"image_host_meshes/{label_name}/{outer_file}"
            
    except Exception as e:
        pass  # Silently fail for individual meshes
    
    # === Inner Mesh ===
    if inner_mask is not None:
        cc_mask_inner = cc_mask_outer & inner_mask
        
        if cc_mask_inner.sum() > 10:
            try:
                padded = np.zeros((nz + 2*pad, ny + 2*pad, nx + 2*pad), dtype=np.uint8)
                padded[pad:pad+nz, pad:pad+ny, pad:pad+nx] = cc_mask_inner
                
                data_t = np.transpose(padded.astype(float), (2, 1, 0))
                
                grid = pv.ImageData(
                    dimensions=data_t.shape,
                    spacing=voxel_size_L2,
                    origin=origin
                )
                grid.point_data['values'] = data_t.ravel(order='F')
                mesh = grid.contour(isosurfaces=[0.5])
                
                if mesh.n_points > 0:
                    inner_file = f"cc_{cc_id:04d}_inner.vtk"
                    mesh.save(str(label_mesh_dir / inner_file))
                    mesh_info['inner_mesh'] = f"image_host_meshes/{label_name}/{inner_file}"
                    
            except Exception:
                pass
    
    if mesh_info['outer_mesh']:
        return mesh_info
    return None


def _precompute_2d_labels_batch(outer_mask, use_gpu=True):
    """
    Pre-compute 2D CC labels for all Z slices
    """
    nz = outer_mask.shape[0]
    labeled_2d_slices = []
    
    cp = get_cupy() if use_gpu and CUPY_LABEL_AVAILABLE else None
    
    for z in range(nz):
        slice_mask = outer_mask[z]
        if not slice_mask.any():
            labeled_2d_slices.append(None)
        else:
            labeled, _ = _cc_label_2d(slice_mask, use_gpu=use_gpu)
            labeled_2d_slices.append(labeled)
    
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    
    return labeled_2d_slices


def prebuild_image_host_meshes(outer_masks_L2, inner_masks_L2, label_names, 
                                voxel_size_L2, output_dir, processing_level=2,
                                max_workers=4):
    """
    Prebuild meshes for Image Host mode (outer + inner per 3D CC)
    
    OPTIMIZED VERSION:
    - Pre-computes 2D CC labels for lookup table
    - Multi-threaded mesh generation (configurable workers)
    - GPU-accelerated CC labeling
    
    Parameters:
    -----------
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    inner_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    label_names : list
        Label names
    voxel_size_L2 : tuple
        Voxel size @ Level 2
    output_dir : Path
        Output directory
    processing_level : int
        Processing level (default 2)
    max_workers : int
        Max parallel workers for mesh generation (default 4)
    
    Returns:
    --------
    result : dict
        Mesh info with lookup tables
    """
    mesh_dir = Path(output_dir) / "image_host_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    print("\n=== Prebuilding Image Host Meshes @ L2 ===")
    print(f"  CC Labeling: {'GPU (CuPy)' if CUPY_LABEL_AVAILABLE else 'CPU (scipy)'}")
    print(f"  Mode: Optimized (pre-computed labels, parallel mesh generation)")
    
    cp = get_cupy() if CUPY_LABEL_AVAILABLE else None
    
    result = {
        'labels': {},
        'processing_level': processing_level,
        'voxel_size_L2': list(voxel_size_L2),
    }
    
    for label_idx, label_name in enumerate(label_names, start=1):
        print(f"\n  Processing {label_name}...")
        
        outer_mask = outer_masks_L2[label_name]
        inner_mask = inner_masks_L2.get(label_name, None)
        
        if hasattr(outer_mask, 'compute'):
            outer_mask = outer_mask.compute()
        outer_mask = outer_mask > 0
        
        if inner_mask is not None:
            if hasattr(inner_mask, 'compute'):
                inner_mask = inner_mask.compute()
            inner_mask = inner_mask > 0
        
        # 3D connected component analysis (GPU accelerated if possible)
        labeled_3d, num_cc = _cc_label_3d(outer_mask, use_gpu=CUPY_LABEL_AVAILABLE)
        print(f"    Found {num_cc} 3D connected components")
        
        label_mesh_dir = mesh_dir / label_name
        label_mesh_dir.mkdir(exist_ok=True)
        
        cc_info = {
            'num_cc': num_cc,
            'meshes': [],
            'z_2dcc_to_3dcc': {},
        }
        
        nz, ny, nx = outer_mask.shape
        pad = 2
        
        # === Build meshes (sequential for now, PyVista isn't thread-safe) ===
        for cc_id in tqdm(range(1, num_cc + 1), desc=f"    Building {label_name} meshes"):
            mesh_info = _build_single_mesh(
                cc_id, labeled_3d, outer_mask, inner_mask, 
                voxel_size_L2, label_mesh_dir, label_name, nz, ny, nx, pad
            )
            if mesh_info:
                cc_info['meshes'].append(mesh_info)
        
        # === Build lookup table (OPTIMIZED: pre-compute 2D labels) ===
        print(f"    Building lookup table (pre-computing 2D labels)...")
        labeled_2d_slices = _precompute_2d_labels_batch(outer_mask, use_gpu=CUPY_LABEL_AVAILABLE)
        
        for z in range(nz):
            labeled_2d = labeled_2d_slices[z]
            if labeled_2d is None:
                continue
            
            z_lookup = {}
            max_cc = labeled_2d.max()
            
            for cc_2d in range(1, max_cc + 1):
                # Find first point of this 2D CC
                points = np.argwhere(labeled_2d == cc_2d)
                if len(points) > 0:
                    y, x = points[0]
                    cc_3d = labeled_3d[z, y, x]
                    if cc_3d > 0:
                        z_lookup[str(cc_2d - 1)] = int(cc_3d)
            
            if z_lookup:
                cc_info['z_2dcc_to_3dcc'][str(z)] = z_lookup
        
        # Cleanup
        del labeled_2d_slices, labeled_3d
        gc.collect()
        
        result['labels'][label_name] = cc_info
        print(f"    ✓ {len(cc_info['meshes'])} meshes saved")
        
        # Cleanup GPU memory after each label
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    
    return result


def prebuild_morphology_meshes(outer_masks_L2, inner_masks_L2, label_names, 
                                label_colors, voxel_size_L2, output_dir,
                                processing_level=2, target_faces=100000):
    """
    Prebuild global morphology meshes for Skeleton Host Mode 2
    
    Parameters:
    -----------
    outer_masks_L2 : dict
        {label_name: np.array} @ Level 2
    inner_masks_L2 : dict  
        {label_name: np.array} @ Level 2
    label_names : list
        Label names
    label_colors : dict
        {label_name: hex_color}
    voxel_size_L2 : tuple
        Voxel size @ Level 2
    output_dir : Path
        Output directory
    processing_level : int
        Processing level (default 2)
    target_faces : int
        Target face count for decimation (default 100k)
    
    Returns:
    --------
    mesh_info : dict
        Mesh metadata
    """
    mesh_dir = Path(output_dir) / "morphology_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    print("\n=== Building Global Morphology Meshes @ L2 ===")
    
    mesh_info = {
        'voxel_size_L2': list(voxel_size_L2),
        'processing_level': processing_level,
        'label_colors': label_colors,
        'meshes': {}
    }
    
    for label_name in tqdm(label_names, desc="Building meshes"):
        
        # === Outer mesh ===
        if label_name in outer_masks_L2:
            outer_mask = outer_masks_L2[label_name]
            
            if isinstance(outer_mask, np.ndarray):
                mask_data = outer_mask
            else:
                mask_data = outer_mask.compute() if hasattr(outer_mask, 'compute') else np.array(outer_mask)
            
            if np.sum(mask_data) > 0:
                print(f"\n  {label_name} outer: {np.sum(mask_data):,} voxels")
                
                mask_blurred = (mask_data > 0).astype(float)
                
                # Transpose for PyVista (Z,Y,X) → (X,Y,Z)
                data_transposed = np.transpose(mask_blurred, (2, 1, 0))
                
                # Create grid
                grid = pv.ImageData(
                    dimensions=data_transposed.shape,
                    spacing=voxel_size_L2,
                    origin=(0, 0, 0)
                )
                grid.point_data['values'] = data_transposed.ravel(order='F')
                
                # Extract surface (marching cubes)
                mesh = grid.contour(isosurfaces=[0.5])
                
                if mesh.n_points > 0:
                    # Save as VTK
                    output_path = mesh_dir / f"outer_{label_name}.vtk"
                    mesh.save(str(output_path))
                    
                    mesh_info['meshes'][f'outer_{label_name}'] = {
                        'path': str(output_path.name),
                        'n_points': int(mesh.n_points),
                        'n_cells': int(mesh.n_cells),
                        'color': label_colors[label_name]
                    }
                    
                    print(f"    ✓ Saved: {mesh.n_points:,} vertices, {mesh.n_cells:,} faces")
        
        # === Inner mesh ===
        if label_name in inner_masks_L2:
            inner_mask = inner_masks_L2.get(label_name, None)
            
            if isinstance(inner_mask, np.ndarray):
                mask_data = inner_mask
            else:
                mask_data = inner_mask.compute() if hasattr(inner_mask, 'compute') else np.array(inner_mask)
            
            if np.sum(mask_data) > 0:
                print(f"  {label_name} inner: {np.sum(mask_data):,} voxels")
                
                mask_blurred = mask_data.astype(float)
                data_transposed = np.transpose(mask_blurred, (2, 1, 0))
                
                grid = pv.ImageData(
                    dimensions=data_transposed.shape,
                    spacing=voxel_size_L2,
                    origin=(0, 0, 0)
                )
                grid.point_data['values'] = data_transposed.ravel(order='F')
                
                mesh = grid.contour(isosurfaces=[0.5])
                
                if mesh.n_points > 0:
                    output_path = mesh_dir / f"inner_{label_name}.vtk"
                    mesh.save(str(output_path))
                    
                    mesh_info['meshes'][f'inner_{label_name}'] = {
                        'path': str(output_path.name),
                        'n_points': int(mesh.n_points),
                        'n_cells': int(mesh.n_cells),
                        'color': label_colors[label_name]
                    }
                    
                    print(f"    ✓ Saved: {mesh.n_points:,} vertices, {mesh.n_cells:,} faces")
    
    # Save mesh info JSON
    info_path = mesh_dir / "mesh_info.json"
    with open(info_path, 'w') as f:
        json.dump(mesh_info, f, indent=2)
    
    # Summary
    total_vertices = sum(m['n_points'] for m in mesh_info['meshes'].values())
    total_faces = sum(m['n_cells'] for m in mesh_info['meshes'].values())
    
    print(f"\n=== Morphology Mesh Summary ===")
    print(f"  Output dir: {mesh_dir}")
    print(f"  Total meshes: {len(mesh_info['meshes'])}")
    print(f"  Total vertices: {total_vertices:,}")
    print(f"  Total faces: {total_faces:,}")
    print("✓ Morphology meshes complete")
    
    return mesh_info
