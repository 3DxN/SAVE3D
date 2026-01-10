"""
2D Connected Component Analysis

Precomputes CC relationships and adjacency for fast runtime propagation.
Supports GPU acceleration via CuPy when available, with CPU fallback.
- Pre-computes all CC labels for all slices upfront
- Uses CuPy for GPU-accelerated numpy operations
- Batch processing for better GPU utilization
"""

import numpy as np
from scipy.ndimage import label as scipy_cc_label
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc

from .config import GPU_AVAILABLE, get_cupy

# GPU-accelerated CC labeling via CuPy
CUPY_LABEL_AVAILABLE = False
_cupy_label = None
_cp = None

if GPU_AVAILABLE:
    try:
        _cp = get_cupy()
        if _cp is not None:
            from cupyx.scipy.ndimage import label as cupy_label
            _cupy_label = cupy_label
            CUPY_LABEL_AVAILABLE = True
    except ImportError:
        pass


def _cc_label_2d(mask, use_gpu=True):
    """
    2D Connected Component labeling with GPU acceleration if available
    """
    if use_gpu and CUPY_LABEL_AVAILABLE and _cp is not None:
        try:
            mask_gpu = _cp.asarray(mask)
            labeled_gpu, num_features = _cupy_label(mask_gpu)
            labeled = _cp.asnumpy(labeled_gpu)
            del mask_gpu, labeled_gpu
            return labeled, int(num_features)
        except Exception:
            pass  # Fall back to CPU
    
    # CPU fallback
    return scipy_cc_label(mask)


def _precompute_all_cc_labels(mask_vol, use_gpu=True):
    """
    Pre-compute CC labels for ALL Z slices at once
    
    This avoids redundant computation when checking overlaps
    
    Parameters:
    -----------
    mask_vol : np.ndarray
        3D mask volume (Z, Y, X)
    use_gpu : bool
        Whether to use GPU acceleration
    
    Returns:
    --------
    labeled_slices : list
        List of (labeled_array, num_features) for each Z slice
    """
    nz = mask_vol.shape[0]
    labeled_slices = []
    
    if use_gpu and CUPY_LABEL_AVAILABLE and _cp is not None:
        # Batch process on GPU for better efficiency
        batch_size = 32  # Process 32 slices at a time
        
        for batch_start in range(0, nz, batch_size):
            batch_end = min(batch_start + batch_size, nz)
            
            for z in range(batch_start, batch_end):
                slice_data = mask_vol[z]
                if not np.any(slice_data):
                    labeled_slices.append((None, 0))
                else:
                    try:
                        mask_gpu = _cp.asarray(slice_data > 0)
                        labeled_gpu, num_features = _cupy_label(mask_gpu)
                        labeled = _cp.asnumpy(labeled_gpu)
                        labeled_slices.append((labeled, int(num_features)))
                        del mask_gpu, labeled_gpu
                    except Exception:
                        # Fallback to CPU
                        labeled, num_features = scipy_cc_label(slice_data > 0)
                        labeled_slices.append((labeled, num_features))
            
            # Free GPU memory after each batch
            _cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU processing
        for z in range(nz):
            slice_data = mask_vol[z]
            if not np.any(slice_data):
                labeled_slices.append((None, 0))
            else:
                labeled, num_features = scipy_cc_label(slice_data > 0)
                labeled_slices.append((labeled, num_features))
    
    return labeled_slices


def _compute_cc_properties_gpu(labeled, cc_mask, skeleton_slice, voxel_size_L2, z):
    """
    Compute CC properties using GPU when available
    """
    if CUPY_LABEL_AVAILABLE and _cp is not None:
        try:
            cc_mask_gpu = _cp.asarray(cc_mask)
            coords_gpu = _cp.argwhere(cc_mask_gpu)
            
            if len(coords_gpu) == 0:
                del cc_mask_gpu, coords_gpu
                return None
            
            y_min = int(_cp.min(coords_gpu[:, 0]))
            y_max = int(_cp.max(coords_gpu[:, 0]))
            x_min = int(_cp.min(coords_gpu[:, 1]))
            x_max = int(_cp.max(coords_gpu[:, 1]))
            area = len(coords_gpu)
            centroid_y = float(_cp.mean(coords_gpu[:, 0]))
            centroid_x = float(_cp.mean(coords_gpu[:, 1]))
            
            # Skeleton marker
            skeleton_gpu = _cp.asarray(skeleton_slice)
            skeleton_intersection = cc_mask_gpu & skeleton_gpu
            
            if _cp.any(skeleton_intersection):
                skel_coords = _cp.argwhere(skeleton_intersection)
                marker_y = float(_cp.mean(skel_coords[:, 0]))
                marker_x = float(_cp.mean(skel_coords[:, 1]))
                has_skeleton = True
            else:
                marker_y = centroid_y
                marker_x = centroid_x
                has_skeleton = False
            
            del cc_mask_gpu, coords_gpu, skeleton_gpu
            
            return {
                'bbox': [y_min, y_max, x_min, x_max],
                'area': area,
                'centroid': [centroid_y, centroid_x],
                'marker_y': marker_y,
                'marker_x': marker_x,
                'has_skeleton': has_skeleton
            }
        except Exception:
            pass  # Fall back to CPU
    
    # CPU fallback
    coords = np.argwhere(cc_mask)
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    area = len(coords)
    centroid_y = coords[:, 0].mean()
    centroid_x = coords[:, 1].mean()
    
    skeleton_intersection = cc_mask & skeleton_slice
    if np.any(skeleton_intersection):
        skel_coords = np.argwhere(skeleton_intersection)
        marker_y = skel_coords[:, 0].mean()
        marker_x = skel_coords[:, 1].mean()
        has_skeleton = True
    else:
        marker_y = centroid_y
        marker_x = centroid_x
        has_skeleton = False
    
    return {
        'bbox': [int(y_min), int(y_max), int(x_min), int(x_max)],
        'area': int(area),
        'centroid': [float(centroid_y), float(centroid_x)],
        'marker_y': float(marker_y),
        'marker_x': float(marker_x),
        'has_skeleton': has_skeleton
    }


def _compute_overlaps_fast(cc_mask, prev_labeled, next_labeled, prev_mask, next_mask):
    """
    Compute overlaps with previous and next slices efficiently
    """
    overlaps_prev = []
    overlaps_next = []
    
    # Previous slice overlaps
    if prev_labeled is not None and prev_mask is not None:
        overlap_mask = cc_mask & (prev_mask > 0)
        if np.any(overlap_mask):
            overlapped_ids = np.unique(prev_labeled[overlap_mask])
            overlapped_ids = overlapped_ids[overlapped_ids > 0]
            
            for prev_id in overlapped_ids:
                overlap_area = np.sum(overlap_mask & (prev_labeled == prev_id))
                overlaps_prev.append({'cc_id': int(prev_id), 'area': int(overlap_area)})
    
    # Next slice overlaps
    if next_labeled is not None and next_mask is not None:
        overlap_mask = cc_mask & (next_mask > 0)
        if np.any(overlap_mask):
            overlapped_ids = np.unique(next_labeled[overlap_mask])
            overlapped_ids = overlapped_ids[overlapped_ids > 0]
            
            for next_id in overlapped_ids:
                overlap_area = np.sum(overlap_mask & (next_labeled == next_id))
                overlaps_next.append({'cc_id': int(next_id), 'area': int(overlap_area)})
    
    return overlaps_prev, overlaps_next


def precompute_2d_cc_and_adjacency(outer_masks_L2, inner_masks_L2, skeleton_L2, 
                                    label_names, voxel_size_L2, label_colors,
                                    processing_level=2, downsample_factor=4):
    """
    Precompute 2D connected components and adjacency relationships @ Level 2
    
    - Pre-computes all CC labels upfront (avoids redundant computation)
    - Uses GPU for CC labeling and numpy operations
    - Simplified overlap computation (same label only, not cross-label)
    
    Parameters:
    -----------
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    inner_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    skeleton_L2 : np.array
        Skeleton volume @ Level 2
    label_names : list
        List of label names
    voxel_size_L2 : tuple
        (vz, vy, vx) in μm at Level 2
    label_colors : dict
        {label_name: hex_color}
    processing_level : int
        Processing level (default 2)
    downsample_factor : int
        Downsample factor (default 4)
    
    Returns:
    --------
    cc_metadata : dict
        Precomputed CC info with overlap relationships
    """
    print("\n=== Precomputing 2D CC and Adjacency @ L2 ===")
    print(f"  CC Labeling: {'GPU (CuPy)' if CUPY_LABEL_AVAILABLE else 'CPU (scipy)'}")
    print("  Mode: Optimized (pre-computed labels)")
    print("This enables fast runtime overlap propagation")
    
    nz, ny, nx = outer_masks_L2[label_names[0]].shape
    skeleton_binary = skeleton_L2 > 0
    
    cc_metadata = {
        'metadata': {
            'voxel_size_L2': list(voxel_size_L2),
            'processing_level': processing_level,
            'downsample_factor': downsample_factor,
            'shape_L2': [nz, ny, nx],
            'label_names': label_names
        },
        'labels': {}
    }
    
    # Process each label
    for label_idx, label_name in enumerate(label_names, start=1):
        print(f"\n--- Processing {label_name} (label {label_idx}) ---")
        
        outer_mask = outer_masks_L2[label_name]
        inner_mask = inner_masks_L2.get(label_name, None)
        
        cc_metadata['labels'][str(label_idx)] = {
            'name': label_name,
            'color': label_colors[label_name],
            'layers': {}
        }
        
        # ===        Pre-compute all CC labels for this mask       ===
        print(f"    Pre-computing CC labels for all {nz} slices...")
        labeled_slices = _precompute_all_cc_labels(outer_mask, use_gpu=CUPY_LABEL_AVAILABLE)
        
        # Process each Z layer
        layers_with_cc = 0
        for z in tqdm(range(nz), desc=f"  {label_name} layers", unit="slice"):
            labeled_outer, num_outer = labeled_slices[z]
            
            if labeled_outer is None or num_outer == 0:
                continue
            
            outer_slice = outer_mask[z]
            skeleton_slice = skeleton_binary[z]
            layer_data = {'outer': [], 'inner': []}
            
            # Get prev/next labeled slices (already computed!)
            prev_labeled, _ = labeled_slices[z-1] if z > 0 else (None, 0)
            next_labeled, _ = labeled_slices[z+1] if z < nz-1 else (None, 0)
            prev_mask = outer_mask[z-1] if z > 0 else None
            next_mask = outer_mask[z+1] if z < nz-1 else None
            
            # Process each CC
            for cc_id in range(1, num_outer + 1):
                cc_mask = (labeled_outer == cc_id)
                
                # Compute properties
                props = _compute_cc_properties_gpu(labeled_outer, cc_mask, skeleton_slice, voxel_size_L2, z)
                if props is None:
                    continue
                
                # Compute overlaps (same label only for speed)
                overlaps_prev, overlaps_next = _compute_overlaps_fast(
                    cc_mask, prev_labeled, next_labeled, prev_mask, next_mask
                )
                
                # Add label info to overlaps
                for ov in overlaps_prev:
                    ov['label'] = label_idx
                for ov in overlaps_next:
                    ov['label'] = label_idx
                
                # Convert to 3D world coordinates
                marker_pos_world = [
                    props['marker_x'] * voxel_size_L2[2],  # X
                    props['marker_y'] * voxel_size_L2[1],  # Y
                    z * voxel_size_L2[0]                   # Z
                ]
                
                cc_info = {
                    'cc_id': int(cc_id),
                    'bbox': props['bbox'],
                    'area': props['area'],
                    'centroid': props['centroid'],
                    'skeleton_marker': {
                        'position': marker_pos_world,
                        'has_skeleton': props['has_skeleton']
                    },
                    'overlaps_prev': overlaps_prev,
                    'overlaps_next': overlaps_next
                }
                
                layer_data['outer'].append(cc_info)
            
            # === Inner CCs (if present) ===
            inner_slice = inner_mask[z] if inner_mask is not None else None
            if inner_slice is not None and np.any(inner_slice):
                labeled_inner, num_inner = _cc_label_2d(inner_slice > 0, use_gpu=CUPY_LABEL_AVAILABLE)
                
                if labeled_inner is not None:
                    for cc_id in range(1, num_inner + 1):
                        cc_mask = (labeled_inner == cc_id)
                        coords = np.argwhere(cc_mask)
                        
                        if len(coords) == 0:
                            continue
                        
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        
                        cc_info = {
                            'cc_id': int(cc_id),
                            'bbox': [int(y_min), int(y_max), int(x_min), int(x_max)],
                            'area': int(len(coords)),
                            'centroid': [float(coords[:, 0].mean()), float(coords[:, 1].mean())],
                            'overlaps_prev': [],
                            'overlaps_next': []
                        }
                        
                        layer_data['inner'].append(cc_info)
            
            # Store layer data
            if layer_data['outer'] or layer_data['inner']:
                cc_metadata['labels'][str(label_idx)]['layers'][str(z)] = layer_data
                layers_with_cc += 1
        
        # Cleanup
        del labeled_slices
        if _cp is not None:
            _cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        print(f"  ✓ {label_name}: {layers_with_cc} layers with CCs")
    
    print(f"\n✓ CC metadata complete")
    return cc_metadata
