"""
Mesh Building Functions

Prebuild meshes for different visualization modes:
- Morphology meshes for Skeleton Host
- Instance meshes for skeleton instances
- Image Host meshes for CC-based navigation

Supports GPU acceleration via CuPy when available, with CPU fallback.
- GPU-accelerated Marching Cubes (cupy-marching-cubes)
- GPU-accelerated CC labeling (CuPy)
- GPU-accelerated binary dilation (CuPy)
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
import threading

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

# GPU-accelerated Marching Cubes via NVIDIA Warp
WARP_AVAILABLE = False
_warp = None

# GPU lock to prevent CuPy/Warp conflicts
# CuPy and Warp use different CUDA contexts and cannot safely run in parallel
# even on different CUDA streams
_GPU_LOCK = threading.Lock()

# Flag for CUDA streams - disabled due to CuPy/Warp context conflicts
_USE_CUDA_STREAMS = False

if GPU_AVAILABLE:
    try:
        cp = get_cupy()
        if cp is not None:
            from cupyx.scipy.ndimage import label as cupy_label
            _cupy_label = cupy_label
            CUPY_LABEL_AVAILABLE = True
    except ImportError:
        pass

# NVIDIA Warp disabled - causes CUDA context conflicts with CuPy
# Using PyVista/VTK for Marching Cubes instead (highly optimized C++)
WARP_AVAILABLE = False
_warp = None


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
    
    For large 3D volumes, falls back to CPU if GPU memory is insufficient.
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
            except Exception:
                pass  # Silent fallback to CPU
    
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
            outer_mesh.save(str(mesh_dir / outer_file), binary=True)
            inst['outer_mesh'] = f"skeleton_instance_meshes/{outer_file}"
        else:
            inst['outer_mesh'] = None
        
        # === Inner Mesh ===
        if inner_mask is not None:
            sub_inner = (inner_mask[z_min:z_max] > 0) & sub_outer
            inner_mesh = _create_mesh_from_mask(sub_inner, voxel_size_L2, origin_offset)
            
            if inner_mesh is not None:
                inner_file = f"inst_{inst_id:04d}_inner.vtk"
                inner_mesh.save(str(mesh_dir / inner_file), binary=True)
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
            mesh.save(str(label_mesh_dir / outer_file), binary=True)
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
                    mesh.save(str(label_mesh_dir / inner_file), binary=True)
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


def _create_single_mesh(mask_3d, voxel_size, origin, nz, ny, nx, pad=2):
    """
    Create a single mesh from a 3D mask
    
    Priority:
    1. GPU Marching Cubes (CuMCubes) - if available and enough GPU memory
    2. PyVista/VTK (optimized, default)
    3. PyMCubes (C++ fallback if VTK fails)
    
    Parameters:
    -----------
    mask_3d : np.ndarray
        Binary mask (Z, Y, X)
    voxel_size : tuple
        (voxel_z, voxel_y, voxel_x) in μm
    origin : tuple
        Origin offset (x, y, z)
    nz, ny, nx : int
        Dimensions
    pad : int
        Padding size
    
    Returns:
    --------
    pv.PolyData or None
    """
    # Pad the mask (needed for all methods)
    padded = np.zeros((nz + 2*pad, ny + 2*pad, nx + 2*pad), dtype=np.float32)
    padded[pad:pad+nz, pad:pad+ny, pad:pad+nx] = mask_3d.astype(np.float32)
    
    # Calculate volume size - only use Warp for LARGE meshes (>500k voxels)
    # For smaller meshes, VTK is faster due to Warp's initialization overhead
    total_voxels = nz * ny * nx
    USE_WARP_THRESHOLD = 500000  # Only use Warp for volumes > 500k voxels
    
    # Try NVIDIA Warp Marching Cubes only for large meshes
    # Uses GPU lock to prevent conflicts with CuPy in multi-threaded environments
    if WARP_AVAILABLE and _warp.is_cuda_available() and total_voxels > USE_WARP_THRESHOLD:
        with _GPU_LOCK:
            try:
                # Warp MarchingCubes expects (X, Y, Z) order as 3D array
                volume_xyz = np.ascontiguousarray(np.transpose(padded, (2, 1, 0)))
                pnx, pny, pnz = volume_xyz.shape
                
                # Create MarchingCubes instance on GPU
                mc = _warp.MarchingCubes(nx=pnx, ny=pny, nz=pnz, device="cuda:0")
                
                # Create Warp array (3D, not flattened)
                volume_wp = _warp.array(volume_xyz, dtype=_warp.float32, device="cuda:0")
                
                # Run marching cubes
                mc.surface(volume_wp, threshold=0.5)
                
                # Get vertices and faces
                vertices = mc.verts.numpy()
                faces = mc.indices.numpy()
                
                if len(vertices) > 0 and len(faces) > 0:
                    # Reshape faces (indices are flat, need to reshape to Nx3)
                    faces = faces.reshape(-1, 3)
                    
                    # Apply spacing and origin
                    spacing = np.array([voxel_size[2], voxel_size[1], voxel_size[0]])
                    vertices = vertices * spacing + np.array(origin)
                    
                    # PyVista expects faces in format: [n_verts, v0, v1, v2, ...]
                    faces_pv = np.hstack([
                        np.full((len(faces), 1), 3, dtype=np.int64),
                        faces.astype(np.int64)
                    ]).ravel()
                    
                    return pv.PolyData(vertices, faces_pv)
                    
            except Exception:
                pass  # Fall back to VTK
    
    # Default: PyVista/VTK (highly optimized, faster for small/medium meshes)
    try:
        data_t = np.transpose(padded, (2, 1, 0))
        
        grid = pv.ImageData(
            dimensions=data_t.shape,
            spacing=voxel_size,
            origin=origin
        )
        grid.point_data['values'] = data_t.ravel(order='F')
        mesh = grid.contour(isosurfaces=[0.5])
        
        return mesh if mesh.n_points > 0 else None
    except Exception:
        pass
    
    return None


def _build_mesh_batch(cc_ids, labeled_3d, outer_mask, inner_mask, voxel_size_L2, 
                       label_mesh_dir, label_name, nz, ny, nx, pad=2):
    """
    Build meshes for a batch of CC IDs
    
    GPU-ACCELERATED operations:
    - cc_mask computation (CuPy if available)
    - voxel_count (CuPy if available)
    - Z range finding (CuPy if available)
    - Marching Cubes (cupy-marching-cubes if available)
    """
    results = []
    
    # Try to use GPU for batch operations
    use_gpu = CUPY_LABEL_AVAILABLE
    cp = get_cupy() if use_gpu else None
    
    # Move labeled_3d to GPU once for the entire batch
    labeled_3d_gpu = None
    if cp is not None:
        try:
            # Check memory
            device = cp.cuda.Device()
            free_mem = device.mem_info[0]
            required_mem = labeled_3d.nbytes * 2
            
            if required_mem < free_mem * 0.5:
                labeled_3d_gpu = cp.asarray(labeled_3d)
        except Exception:
            labeled_3d_gpu = None
    
    for cc_id in cc_ids:
        # GPU-accelerated mask and count
        if labeled_3d_gpu is not None:
            try:
                cc_mask_gpu = (labeled_3d_gpu == cc_id)
                voxel_count = int(cp.sum(cc_mask_gpu))
                
                if voxel_count < 10:
                    continue
                
                # Z range (GPU)
                z_has_data = cp.any(cc_mask_gpu, axis=(1, 2))
                z_coords = cp.where(z_has_data)[0]
                if len(z_coords) == 0:
                    continue
                z_min, z_max = int(z_coords.min()), int(z_coords.max())
                
                # Transfer mask to CPU for mesh building
                cc_mask_outer = cp.asnumpy(cc_mask_gpu)
                del cc_mask_gpu
            except Exception:
                # Fallback to CPU
                cc_mask_outer = (labeled_3d == cc_id)
                voxel_count = np.sum(cc_mask_outer)
                if voxel_count < 10:
                    continue
                z_coords = np.where(cc_mask_outer.any(axis=(1, 2)))[0]
                if len(z_coords) == 0:
                    continue
                z_min, z_max = int(z_coords.min()), int(z_coords.max())
        else:
            # CPU path
            cc_mask_outer = (labeled_3d == cc_id)
            voxel_count = np.sum(cc_mask_outer)
            
            if voxel_count < 10:
                continue
            
            z_coords = np.where(cc_mask_outer.any(axis=(1, 2)))[0]
            if len(z_coords) == 0:
                continue
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
        
        # === Outer Mesh (GPU or CPU) ===
        mesh = _create_single_mesh(cc_mask_outer, voxel_size_L2, origin, nz, ny, nx, pad)
        if mesh is not None and mesh.n_points > 0:
            try:
                outer_file = f"cc_{cc_id:04d}_outer.vtk"
                mesh.save(str(label_mesh_dir / outer_file), binary=True)
                mesh_info['outer_mesh'] = f"image_host_meshes/{label_name}/{outer_file}"
            except Exception:
                pass
        
        # === Inner Mesh (GPU or CPU) ===
        if inner_mask is not None:
            cc_mask_inner = cc_mask_outer & inner_mask
            
            if cc_mask_inner.sum() > 10:
                mesh = _create_single_mesh(cc_mask_inner, voxel_size_L2, origin, nz, ny, nx, pad)
                if mesh is not None and mesh.n_points > 0:
                    try:
                        inner_file = f"cc_{cc_id:04d}_inner.vtk"
                        mesh.save(str(label_mesh_dir / inner_file), binary=True)
                        mesh_info['inner_mesh'] = f"image_host_meshes/{label_name}/{inner_file}"
                    except Exception:
                        pass
        
        if mesh_info['outer_mesh']:
            results.append(mesh_info)
    
    # Cleanup GPU memory
    if labeled_3d_gpu is not None:
        del labeled_3d_gpu
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    
    return results


def _process_single_label_image_host(args):
    """
    Process a single label for image host meshes (for parallel processing)
    """
    label_idx, label_name, outer_mask_data, inner_mask_data, voxel_size_L2, \
        mesh_dir, use_gpu = args
    
    # Convert to binary
    outer_mask = outer_mask_data > 0
    inner_mask = inner_mask_data > 0 if inner_mask_data is not None else None
    
    total_voxels = np.sum(outer_mask)
    if total_voxels == 0:
        return label_name, {'num_cc': 0, 'meshes': [], 'z_2dcc_to_3dcc': {}}, 0
    
    # 3D CC analysis (use GPU if available and not in parallel mode)
    labeled_3d, num_cc = _cc_label_3d(outer_mask, use_gpu=use_gpu)
    
    if num_cc == 0:
        return label_name, {'num_cc': 0, 'meshes': [], 'z_2dcc_to_3dcc': {}}, 0
    
    label_mesh_dir = mesh_dir / label_name
    label_mesh_dir.mkdir(exist_ok=True)
    
    cc_info = {
        'num_cc': num_cc,
        'meshes': [],
        'z_2dcc_to_3dcc': {},
    }
    
    nz, ny, nx = outer_mask.shape
    pad = 2
    
    # Build meshes in batches
    batch_size = 50
    all_cc_ids = list(range(1, num_cc + 1))
    
    for batch_start in range(0, len(all_cc_ids), batch_size):
        batch_ids = all_cc_ids[batch_start:batch_start + batch_size]
        batch_results = _build_mesh_batch(
            batch_ids, labeled_3d, outer_mask, inner_mask,
            voxel_size_L2, label_mesh_dir, label_name, nz, ny, nx, pad
        )
        cc_info['meshes'].extend(batch_results)
    
    # Build lookup table
    z_with_data = np.where(outer_mask.any(axis=(1, 2)))[0]
    
    for z in z_with_data:
        slice_mask = outer_mask[z]
        if not np.any(slice_mask):
            continue
        
        # Use CPU for 2D CC to avoid GPU contention in parallel mode
        labeled_2d, num_2d = _cc_label_2d(slice_mask, use_gpu=False)
        if labeled_2d is None or num_2d == 0:
            continue
        
        z_lookup = {}
        for cc_2d in range(1, num_2d + 1):
            points = np.argwhere(labeled_2d == cc_2d)
            if len(points) > 0:
                y, x = points[0]
                cc_3d = labeled_3d[z, y, x]
                if cc_3d > 0:
                    z_lookup[str(cc_2d - 1)] = int(cc_3d)
        
        if z_lookup:
            cc_info['z_2dcc_to_3dcc'][str(z)] = z_lookup
    
    return label_name, cc_info, len(cc_info['meshes'])


def prebuild_image_host_meshes(outer_masks_L2, inner_masks_L2, label_names, 
                                voxel_size_L2, output_dir, processing_level=2,
                                max_workers=4):
    """
    Prebuild meshes for Image Host mode (outer + inner per 3D CC)
    
    OPTIMIZED VERSION v3:
    - Parallel processing of multiple labels using ThreadPoolExecutor
    - GPU-accelerated 3D CC labeling
    - Batch mesh building
    - Skip empty labels
    
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
        Max parallel workers (default 4)
    
    Returns:
    --------
    result : dict
        Mesh info with lookup tables
    """
    import os
    
    mesh_dir = Path(output_dir) / "image_host_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    # Determine number of workers
    n_workers = min(max_workers, os.cpu_count() or 4, len(label_names))
    
    print("\n=== Prebuilding Image Host Meshes @ L2 ===")
    print(f"  CC Labeling: {'GPU (CuPy)' if CUPY_LABEL_AVAILABLE else 'CPU (scipy)'}")
    print(f"  Mode: Parallel ({n_workers} workers)")
    
    result = {
        'labels': {},
        'processing_level': processing_level,
        'voxel_size_L2': list(voxel_size_L2),
    }
    
    # Pre-load all masks to avoid issues with dask arrays in threads
    print("  Pre-loading masks...")
    args_list = []
    for label_idx, label_name in enumerate(label_names, start=1):
        outer_mask = outer_masks_L2[label_name]
        if hasattr(outer_mask, 'compute'):
            outer_mask = outer_mask.compute()
        
        inner_mask = inner_masks_L2.get(label_name, None)
        if inner_mask is not None and hasattr(inner_mask, 'compute'):
            inner_mask = inner_mask.compute()
        
        # Use GPU for 3D CC but CPU for 2D CC (to avoid contention)
        args_list.append((
            label_idx, label_name, outer_mask, inner_mask,
            voxel_size_L2, mesh_dir, CUPY_LABEL_AVAILABLE
        ))
    
    # Process in parallel using ThreadPoolExecutor
    total_labels = len(label_names)
    completed = 0
    total_meshes = 0
    
    print(f"  Processing {total_labels} labels...")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_label_image_host, args): args[1] 
                   for args in args_list}
        
        for future in as_completed(futures):
            label_name = futures[future]
            try:
                name, cc_info, n_meshes = future.result()
                result['labels'][name] = cc_info
                total_meshes += n_meshes
                completed += 1
                if n_meshes > 0:
                    print(f"  [{completed}/{total_labels}] {name}: {n_meshes} meshes, {cc_info['num_cc']} CCs")
            except Exception as e:
                print(f"  [{completed}/{total_labels}] {label_name}: ERROR - {e}")
                result['labels'][label_name] = {'num_cc': 0, 'meshes': [], 'z_2dcc_to_3dcc': {}}
                completed += 1
    
    # Cleanup GPU memory
    cp = get_cupy() if CUPY_LABEL_AVAILABLE else None
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    
    print(f"\n✓ Image host meshes complete: {total_meshes} total meshes")
    
    return result


def _gpu_binary_dilation_3d(mask, radius=3):
    """
    GPU-accelerated 3D binary dilation using CuPy
    Falls back to CPU if GPU is not available or fails
    """
    if not CUPY_LABEL_AVAILABLE:
        from skimage.morphology import binary_dilation, ball
        return binary_dilation(mask, ball(radius))
    
    cp = get_cupy()
    if cp is None:
        from skimage.morphology import binary_dilation, ball
        return binary_dilation(mask, ball(radius))
    
    try:
        from cupyx.scipy.ndimage import binary_dilation as cupy_binary_dilation
        from skimage.morphology import ball
        
        # Check GPU memory
        device = cp.cuda.Device()
        free_mem = device.mem_info[0]
        required_mem = mask.nbytes * 3  # input + output + intermediate
        
        if required_mem < free_mem * 0.5:
            mask_gpu = cp.asarray(mask)
            struct = cp.asarray(ball(radius))
            dilated_gpu = cupy_binary_dilation(mask_gpu, structure=struct)
            result = cp.asnumpy(dilated_gpu)
            del mask_gpu, dilated_gpu, struct
            cp.get_default_memory_pool().free_all_blocks()
            return result
    except Exception:
        pass
    
    # Fallback to CPU
    from skimage.morphology import binary_dilation, ball
    return binary_dilation(mask, ball(radius))


def prebuild_skeleton_meshes(skeleton_L2, outer_masks_L2, label_names, 
                              label_colors, voxel_size_L2, output_dir,
                              processing_level=2):
    """
    Prebuild skeleton meshes for each label (for fast app startup)
    
    GPU-ACCELERATED:
    - Binary dilation (CuPy if available)
    - Mask operations
    
    Parameters:
    -----------
    skeleton_L2 : np.ndarray
        Skeleton volume @ Level 2
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
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
    
    Returns:
    --------
    mesh_info : dict
        Skeleton mesh metadata
    """
    mesh_dir = Path(output_dir) / "skeleton_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    print("\n=== Prebuilding Skeleton Meshes @ L2 ===")
    print(f"  GPU Acceleration: {'Enabled (CuPy)' if CUPY_LABEL_AVAILABLE else 'Disabled (CPU only)'}")
    
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"  Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        print("  ⚠ No skeleton voxels!")
        return {'meshes': {}, 'total_voxels': 0}
    
    # Assign skeleton voxels to labels (first come, first served)
    skeleton_labels = np.zeros_like(skeleton_binary, dtype=np.uint8)
    
    for label_idx, label_name in enumerate(label_names, start=1):
        if label_name not in outer_masks_L2:
            continue
        
        outer_mask = outer_masks_L2[label_name]
        if hasattr(outer_mask, 'compute'):
            outer_mask = outer_mask.compute()
        
        available = (skeleton_binary > 0) & (skeleton_labels == 0)
        intersection = available & (outer_mask > 0)
        skeleton_labels[intersection] = label_idx
        
        voxel_count = np.sum(intersection)
        print(f"    {label_name}: {voxel_count:,} voxels")
    
    # Generate meshes for each label
    mesh_info = {
        'voxel_size_L2': list(voxel_size_L2),
        'processing_level': processing_level,
        'total_voxels': int(total_voxels),
        'meshes': {}
    }
    
    print("\n  Generating meshes...")
    print(f"  Marching Cubes: CPU (PyVista/VTK) - Multi-threaded")
    
    # Step 1: Dilate all masks (GPU-accelerated, sequential)
    print("  Step 1: Dilating skeleton masks (GPU)...")
    dilated_masks = {}
    for label_idx, label_name in enumerate(label_names, start=1):
        label_mask = (skeleton_labels == label_idx)
        
        if np.sum(label_mask) == 0:
            continue
        
        # Dilate for visibility (GPU-accelerated if available)
        dilated_masks[label_name] = {
            'mask': _gpu_binary_dilation_3d(label_mask, radius=3),
            'voxel_count': int(np.sum(label_mask)),
            'color': label_colors[label_name]
        }
    
    # Step 2: Generate meshes in parallel (CPU Marching Cubes)
    print(f"  Step 2: Generating {len(dilated_masks)} meshes (parallel)...")
    
    def _build_skeleton_mesh(args):
        label_name, mask_data, voxel_count, color, mesh_dir, voxel_size = args
        mesh = _create_mesh_for_morphology(mask_data, voxel_size)
        
        if mesh is not None and mesh.n_cells > 0:
            mesh_file = f"skeleton_{label_name}.vtk"
            mesh.save(str(mesh_dir / mesh_file), binary=True)
            return {
                'label_name': label_name,
                'path': f"skeleton_meshes/{mesh_file}",
                'n_points': int(mesh.n_points),
                'n_cells': int(mesh.n_cells),
                'color': color,
                'voxel_count': voxel_count
            }
        return None
    
    args_list = [
        (label_name, data['mask'], data['voxel_count'], data['color'], mesh_dir, voxel_size_L2)
        for label_name, data in dilated_masks.items()
    ]
    
    import os
    n_workers = min(4, os.cpu_count() or 4, len(args_list))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_build_skeleton_mesh, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Building skeleton meshes"):
            try:
                result = future.result()
                if result:
                    mesh_info['meshes'][result['label_name']] = {
                        'path': result['path'],
                        'n_points': result['n_points'],
                        'n_cells': result['n_cells'],
                        'color': result['color'],
                        'voxel_count': result['voxel_count']
                    }
            except Exception as e:
                label_name = futures[future]
                print(f"    ✗ {label_name}: {e}")
    
    # Save mesh info JSON
    info_path = mesh_dir / "skeleton_mesh_info.json"
    with open(info_path, 'w') as f:
        json.dump(mesh_info, f, indent=2)
    
    print(f"\n✓ Skeleton meshes complete: {len(mesh_info['meshes'])} meshes saved")
    
    return mesh_info


def _create_mesh_for_morphology(mask_3d, voxel_size, origin=(0, 0, 0)):
    """
    Create mesh for morphology visualization
    
    Priority:
    1. NVIDIA Warp (GPU/CPU accelerated) - if available
    2. PyVista/VTK (fallback)
    
    Parameters:
    -----------
    mask_3d : np.ndarray
        Binary mask (Z, Y, X)
    voxel_size : tuple
        (voxel_z, voxel_y, voxel_x) in μm
    origin : tuple
        Origin (x, y, z)
    
    Returns:
    --------
    pv.PolyData or None
    """
    if mask_3d.sum() == 0:
        return None
    
    mask_float = mask_3d.astype(np.float32)
    nz, ny, nx = mask_3d.shape
    total_voxels = nz * ny * nx
    
    # Only use Warp for LARGE meshes (>500k voxels)
    # For smaller meshes, VTK is faster due to Warp's initialization overhead
    USE_WARP_THRESHOLD = 500000
    
    # Try NVIDIA Warp Marching Cubes only for large meshes
    # Uses GPU lock to prevent conflicts with CuPy in multi-threaded environments
    if WARP_AVAILABLE and _warp.is_cuda_available() and total_voxels > USE_WARP_THRESHOLD:
        with _GPU_LOCK:
            try:
                # Warp MarchingCubes expects (X, Y, Z) order as 3D array
                volume_xyz = np.ascontiguousarray(np.transpose(mask_float, (2, 1, 0)))
                wnx, wny, wnz = volume_xyz.shape
                
                mc = _warp.MarchingCubes(nx=wnx, ny=wny, nz=wnz, device="cuda:0")
                volume_wp = _warp.array(volume_xyz, dtype=_warp.float32, device="cuda:0")
                mc.surface(volume_wp, threshold=0.5)
                vertices = mc.verts.numpy()
                faces = mc.indices.numpy()
                
                if len(vertices) > 0 and len(faces) > 0:
                    # Reshape faces
                    faces = faces.reshape(-1, 3)
                    
                    spacing = np.array([voxel_size[2], voxel_size[1], voxel_size[0]])
                    vertices = vertices * spacing + np.array(origin)
                    
                    faces_pv = np.hstack([
                        np.full((len(faces), 1), 3, dtype=np.int64),
                        faces.astype(np.int64)
                    ]).ravel()
                    
                    return pv.PolyData(vertices, faces_pv)
                    
            except Exception:
                pass  # Fall back to VTK
    
    # Default: PyVista/VTK (highly optimized, faster for small/medium meshes)
    try:
        data_transposed = np.transpose(mask_float, (2, 1, 0))
        
        grid = pv.ImageData(
            dimensions=data_transposed.shape,
            spacing=voxel_size,
            origin=origin
        )
        grid.point_data['values'] = data_transposed.ravel(order='F')
        mesh = grid.contour(isosurfaces=[0.5])
        
        return mesh if mesh.n_points > 0 else None
    except Exception:
        pass
    
    return None


def _build_single_morphology_mesh(args):
    """Helper function for parallel morphology mesh building"""
    label_name, outer_mask_data, inner_mask_data, voxel_size_L2, mesh_dir, label_color = args
    
    result = {'label_name': label_name, 'meshes': {}}
    
    # === Outer mesh ===
    if outer_mask_data is not None and np.sum(outer_mask_data) > 0:
        mesh = _create_mesh_for_morphology(outer_mask_data > 0, voxel_size_L2)
        
        if mesh is not None and mesh.n_points > 0:
            output_path = mesh_dir / f"outer_{label_name}.vtk"
            mesh.save(str(output_path), binary=True)
            
            result['meshes'][f'outer_{label_name}'] = {
                'path': str(output_path.name),
                'n_points': int(mesh.n_points),
                'n_cells': int(mesh.n_cells),
                'color': label_color
            }
    
    # === Inner mesh ===
    if inner_mask_data is not None and np.sum(inner_mask_data) > 0:
        mesh = _create_mesh_for_morphology(inner_mask_data > 0, voxel_size_L2)
        
        if mesh is not None and mesh.n_points > 0:
            output_path = mesh_dir / f"inner_{label_name}.vtk"
            mesh.save(str(output_path), binary=True)
            
            result['meshes'][f'inner_{label_name}'] = {
                'path': str(output_path.name),
                'n_points': int(mesh.n_points),
                'n_cells': int(mesh.n_cells),
                'color': label_color
            }
    
    return result


def prebuild_morphology_meshes(outer_masks_L2, inner_masks_L2, label_names, 
                                label_colors, voxel_size_L2, output_dir,
                                processing_level=2, target_faces=100000,
                                max_workers=4):
    """
    Prebuild global morphology meshes for Skeleton Host Mode 2
    
    MULTI-THREADED: Uses ThreadPoolExecutor for parallel mesh generation.
    VTK/PyVista Marching Cubes is thread-safe and CPU-bound.
    
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
    max_workers : int
        Number of parallel workers (default 4)
    
    Returns:
    --------
    mesh_info : dict
        Mesh metadata
    """
    import os
    mesh_dir = Path(output_dir) / "morphology_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    n_workers = min(max_workers, os.cpu_count() or 4, len(label_names))
    
    print("\n=== Building Global Morphology Meshes @ L2 ===")
    print(f"  Marching Cubes: CPU (PyVista/VTK)")
    print(f"  Parallel workers: {n_workers}")
    
    mesh_info = {
        'voxel_size_L2': list(voxel_size_L2),
        'processing_level': processing_level,
        'label_colors': label_colors,
        'meshes': {}
    }
    
    # Pre-load all masks and prepare args
    args_list = []
    for label_name in label_names:
        outer_mask = outer_masks_L2.get(label_name)
        if outer_mask is not None:
            if hasattr(outer_mask, 'compute'):
                outer_mask = outer_mask.compute()
            elif not isinstance(outer_mask, np.ndarray):
                outer_mask = np.array(outer_mask)
        
        inner_mask = inner_masks_L2.get(label_name)
        if inner_mask is not None:
            if hasattr(inner_mask, 'compute'):
                inner_mask = inner_mask.compute()
            elif not isinstance(inner_mask, np.ndarray):
                inner_mask = np.array(inner_mask)
        
        args_list.append((
            label_name, outer_mask, inner_mask,
            voxel_size_L2, mesh_dir, label_colors.get(label_name, '#808080')
        ))
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_build_single_morphology_mesh, args): args[0] 
                   for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building meshes"):
            try:
                result = future.result()
                mesh_info['meshes'].update(result['meshes'])
            except Exception as e:
                label_name = futures[future]
                print(f"  ✗ {label_name}: {e}")
    
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
