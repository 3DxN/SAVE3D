"""
2D Connected Component Analysis

Precomputes CC relationships and adjacency for fast runtime propagation.
"""

import numpy as np
from scipy.ndimage import label as cc_label
from tqdm import tqdm


def precompute_2d_cc_and_adjacency(outer_masks_L2, inner_masks_L2, skeleton_L2, 
                                    label_names, voxel_size_L2, label_colors,
                                    processing_level=2, downsample_factor=4):
    """
    Precompute 2D connected components and adjacency relationships @ Level 2
    This is the KEY optimization for runtime performance
    
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
        
        # Process each Z layer with tqdm
        for z in tqdm(range(nz), desc=f"  {label_name} layers", unit="slice"):
            outer_slice = outer_mask[z]
            inner_slice = inner_mask[z] if inner_mask is not None else np.zeros_like(outer_slice)
            skeleton_slice = skeleton_binary[z]
            
            layer_data = {'outer': [], 'inner': []}
            
            # === Outer CCs ===
            if np.any(outer_slice):
                labeled_outer, num_outer = cc_label(outer_slice)
                
                for cc_id in range(1, num_outer + 1):
                    cc_mask = (labeled_outer == cc_id)
                    coords = np.argwhere(cc_mask)
                    
                    if len(coords) == 0:
                        continue
                    
                    # Bbox and properties
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    area = len(coords)
                    centroid_y = coords[:, 0].mean()
                    centroid_x = coords[:, 1].mean()
                    
                    # Skeleton marker position
                    skeleton_intersection = cc_mask & skeleton_slice
                    if np.any(skeleton_intersection):
                        # Has skeleton: use intersection centroid
                        skel_coords = np.argwhere(skeleton_intersection)
                        marker_y = skel_coords[:, 0].mean()
                        marker_x = skel_coords[:, 1].mean()
                        has_skeleton = True
                    else:
                        # No skeleton: use CC centroid
                        marker_y = centroid_y
                        marker_x = centroid_x
                        has_skeleton = False
                    
                    # Convert to 3D world coordinates
                    marker_pos_world = [
                        marker_x * voxel_size_L2[2],  # X
                        marker_y * voxel_size_L2[1],  # Y
                        z * voxel_size_L2[0]          # Z
                    ]
                    
                    cc_info = {
                        'cc_id': int(cc_id),
                        'bbox': [int(y_min), int(y_max), int(x_min), int(x_max)],
                        'area': int(area),
                        'centroid': [float(centroid_y), float(centroid_x)],
                        'skeleton_marker': {
                            'position': marker_pos_world,
                            'has_skeleton': has_skeleton
                        },
                        'overlaps_prev': [],
                        'overlaps_next': []
                    }
                    
                    # Compute overlaps with z-1 (same label + other labels)
                    if z > 0:
                        # Same label
                        prev_outer = outer_mask[z-1]
                        prev_labeled, prev_num = cc_label(prev_outer)
                        
                        overlap_mask = cc_mask & (prev_outer > 0)
                        if np.any(overlap_mask):
                            overlapped_labels = np.unique(prev_labeled[overlap_mask])
                            overlapped_labels = overlapped_labels[overlapped_labels > 0]
                            
                            for prev_label in overlapped_labels:
                                overlap_area = np.sum(overlap_mask & (prev_labeled == prev_label))
                                cc_info['overlaps_prev'].append({
                                    'label': label_idx,
                                    'cc_id': int(prev_label),
                                    'area': int(overlap_area)
                                })
                        
                        # Other labels
                        for other_label_idx, other_label_name in enumerate(label_names, start=1):
                            if other_label_idx != label_idx:
                                other_prev = outer_masks_L2[other_label_name][z-1]
                                other_labeled, other_num = cc_label(other_prev)
                                
                                overlap_mask = cc_mask & (other_prev > 0)
                                if np.any(overlap_mask):
                                    overlapped_labels = np.unique(other_labeled[overlap_mask])
                                    overlapped_labels = overlapped_labels[overlapped_labels > 0]
                                    
                                    for other_cc_id in overlapped_labels:
                                        overlap_area = np.sum(overlap_mask & (other_labeled == other_cc_id))
                                        cc_info['overlaps_prev'].append({
                                            'label': other_label_idx,
                                            'cc_id': int(other_cc_id),
                                            'area': int(overlap_area)
                                        })
                    
                    # Compute overlaps with z+1 (same label + other labels)
                    if z < nz - 1:
                        # Same label
                        next_outer = outer_mask[z+1]
                        next_labeled, next_num = cc_label(next_outer)
                        
                        overlap_mask = cc_mask & (next_outer > 0)
                        if np.any(overlap_mask):
                            overlapped_labels = np.unique(next_labeled[overlap_mask])
                            overlapped_labels = overlapped_labels[overlapped_labels > 0]
                            
                            for next_label in overlapped_labels:
                                overlap_area = np.sum(overlap_mask & (next_labeled == next_label))
                                cc_info['overlaps_next'].append({
                                    'label': label_idx,
                                    'cc_id': int(next_label),
                                    'area': int(overlap_area)
                                })
                        
                        # Other labels
                        for other_label_idx, other_label_name in enumerate(label_names, start=1):
                            if other_label_idx != label_idx:
                                other_next = outer_masks_L2[other_label_name][z+1]
                                other_labeled, other_num = cc_label(other_next)
                                
                                overlap_mask = cc_mask & (other_next > 0)
                                if np.any(overlap_mask):
                                    overlapped_labels = np.unique(other_labeled[overlap_mask])
                                    overlapped_labels = overlapped_labels[overlapped_labels > 0]
                                    
                                    for other_cc_id in overlapped_labels:
                                        overlap_area = np.sum(overlap_mask & (other_labeled == other_cc_id))
                                        cc_info['overlaps_next'].append({
                                            'label': other_label_idx,
                                            'cc_id': int(other_cc_id),
                                            'area': int(overlap_area)
                                        })
                    
                    layer_data['outer'].append(cc_info)
            
            # === Inner CCs (same logic) ===
            if inner_slice is not None and np.any(inner_slice):
                labeled_inner, num_inner = cc_label(inner_slice)
                
                for cc_id in range(1, num_inner + 1):
                    cc_mask = (labeled_inner == cc_id)
                    coords = np.argwhere(cc_mask)
                    
                    if len(coords) == 0:
                        continue
                    
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    area = len(coords)
                    centroid_y = coords[:, 0].mean()
                    centroid_x = coords[:, 1].mean()
                    
                    cc_info = {
                        'cc_id': int(cc_id),
                        'bbox': [int(y_min), int(y_max), int(x_min), int(x_max)],
                        'area': int(area),
                        'centroid': [float(centroid_y), float(centroid_x)],
                        'overlaps_prev': [],
                        'overlaps_next': []
                    }
                    
                    # Compute overlaps with z-1
                    if z > 0:
                        prev_inner = inner_mask[z-1]
                        prev_labeled, prev_num = cc_label(prev_inner)
                        
                        overlap_mask = cc_mask & (prev_inner > 0)
                        if np.any(overlap_mask):
                            overlapped_labels = np.unique(prev_labeled[overlap_mask])
                            overlapped_labels = overlapped_labels[overlapped_labels > 0]
                            
                            for prev_label in overlapped_labels:
                                overlap_area = np.sum(overlap_mask & (prev_labeled == prev_label))
                                cc_info['overlaps_prev'].append({
                                    'label': label_idx,
                                    'cc_id': int(prev_label),
                                    'area': int(overlap_area)
                                })
                    
                    # Compute overlaps with z+1
                    if z < nz - 1:
                        next_inner = inner_mask[z+1]
                        next_labeled, next_num = cc_label(next_inner)
                        
                        overlap_mask = cc_mask & (next_inner > 0)
                        if np.any(overlap_mask):
                            overlapped_labels = np.unique(next_labeled[overlap_mask])
                            overlapped_labels = overlapped_labels[overlapped_labels > 0]
                            
                            for next_label in overlapped_labels:
                                overlap_area = np.sum(overlap_mask & (next_labeled == next_label))
                                cc_info['overlaps_next'].append({
                                    'label': label_idx,
                                    'cc_id': int(next_label),
                                    'area': int(overlap_area)
                                })
                    
                    layer_data['inner'].append(cc_info)
            
            # Store layer data
            if layer_data['outer'] or layer_data['inner']:
                cc_metadata['labels'][str(label_idx)]['layers'][str(z)] = layer_data
        
        print(f"  ✓ {label_name}: {len(cc_metadata['labels'][str(label_idx)]['layers'])} layers with CCs")
    
    print(f"\n✓ CC metadata complete")
    return cc_metadata
