"""
Skeleton Processing

Functions for skeleton intersection analysis, graph building,
and KD-Tree point extraction.
"""

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial import KDTree
from tqdm import tqdm
import time

from .config import GPU_AVAILABLE, get_cupy


def preprocess_skeleton_intersections(skeleton_L2, outer_masks_L2, label_names, label_colors):
    """
    Simple skeleton intersection @ Level 2 - pure NumPy
    
    Parameters:
    -----------
    skeleton_L2 : np.array
        Binary skeleton volume @ Level 2
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    label_names : list
        List of label names
    label_colors : dict
        {label_name: hex_color}
    
    Returns:
    --------
    results : dict
        {label_name: {voxels, proportion, color}}
    """
    print("\n=== Simple Skeleton Intersection @ L2 ===")
    
    # Step 1: Create binary skeleton
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        return {}
    
    # Step 2: Track which label owns each voxel
    skeleton_labels = np.zeros_like(skeleton_binary, dtype=np.uint8)
    results = {}
    
    # Step 3: Process each label (first come, first served)
    print("Processing labels...")
    for idx, name in enumerate(label_names, start=1):
        # Find skeleton voxels in this mask that aren't claimed yet
        available = (skeleton_binary > 0) & (skeleton_labels == 0)
        mask = outer_masks_L2[name] > 0
        intersection = available & mask
        
        # Count and claim these voxels
        count = np.sum(intersection)
        skeleton_labels[intersection] = idx
        
        proportion = 100.0 * count / total_voxels
        print(f"  {name}: {count:,} voxels ({proportion:.2f}%)")
        
        results[name] = {
            'voxels': int(count),
            'proportion': float(proportion),
            'color': label_colors[name]
        }
    
    # Step 4: Report unclassified
    unclassified = np.sum(skeleton_labels == 0)
    print(f"  Unclassified: {unclassified:,} voxels ({100.0*unclassified/total_voxels:.2f}%)")
    print("✓ Done")
    
    return results


def build_skeleton_graph(skeleton_L2, outer_masks_L2, label_names, voxel_size_L2,
                         processing_level=2, target_max_points=50000):
    """
    Build skeleton graph with connectivity, distances, and branch detection
    
    Parameters:
    -----------
    skeleton_L2 : np.array
        Binary skeleton volume @ Level 2
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    label_names : list
        List of label names
    voxel_size_L2 : tuple
        (vz, vy, vx) in μm at Level 2
    processing_level : int
        Processing level (default 2)
    target_max_points : int
        If total voxels > this, downsample skeleton first
    
    Returns:
    --------
    skeleton_graph : dict
        Graph structure with nodes, edges, and summary
    """
    print("\n=== Building Skeleton Graph @ L2 ===")
    t_start = time.time()
    
    # Step 1: Get skeleton voxels
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        print("⚠️ No skeleton voxels!")
        return None
    
    # Step 2: Adaptive sampling (if needed)
    if total_voxels > target_max_points:
        print(f"Skeleton too dense ({total_voxels:,} > {target_max_points:,}), thinning...")
        
        # Morphological thinning to reduce points while preserving topology
        from skimage.morphology import skeletonize_3d
        skeleton_binary = skeletonize_3d(skeleton_binary)
        total_voxels_after = np.sum(skeleton_binary)
        print(f"After thinning: {total_voxels_after:,} voxels")
        
        # If still too many, subsample
        if total_voxels_after > target_max_points:
            sampling_rate = int(np.ceil(total_voxels_after / target_max_points))
            print(f"Still too many, will sample graph nodes 1/{sampling_rate}")
        else:
            sampling_rate = 1
    else:
        sampling_rate = 1
        print(f"Skeleton size OK, keeping all points")
    
    # Step 3: Get voxel coordinates
    coords_zyx = np.argwhere(skeleton_binary)  # (N, 3) in [z, y, x]
    n_voxels = len(coords_zyx)
    print(f"Skeleton voxels: {n_voxels:,}")
    
    # Step 4: Build label volume (same as before)
    print("Assigning labels to skeleton voxels...")
    skeleton_labels = np.zeros_like(skeleton_binary, dtype=np.uint8)
    
    for label_idx, label_name in enumerate(label_names, start=1):
        available = (skeleton_binary > 0) & (skeleton_labels == 0)
        mask = outer_masks_L2[label_name] > 0
        intersection = available & mask
        skeleton_labels[intersection] = label_idx
        count = np.sum(intersection)
        print(f"  {label_name}: {count:,} voxels")
    
    # Step 5: Build voxel → node_id mapping
    print("Building coordinate index...")
    voxel_to_node = {}
    for node_id, (z, y, x) in enumerate(coords_zyx):
        voxel_to_node[(z, y, x)] = node_id
    
    # Step 6: Find connectivity using 26-neighborhood
    print("Finding 26-connectivity edges...")
    
    # 26-neighborhood offsets
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append((dz, dy, dx))
    
    edges = []
    edge_set = set()  # To avoid duplicates
    neighbor_lists = [[] for _ in range(n_voxels)]
    
    for node_id, (z, y, x) in enumerate(tqdm(coords_zyx, desc="Finding edges", unit="node")):
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            neighbor_key = (nz, ny, nx)
            
            if neighbor_key in voxel_to_node:
                neighbor_id = voxel_to_node[neighbor_key]
                
                # Add to neighbor list
                neighbor_lists[node_id].append(neighbor_id)
                
                # Add edge (avoid duplicates by ordering)
                edge_key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    
                    # Calculate distance in μm
                    coord_a = coords_zyx[node_id]
                    coord_b = coords_zyx[neighbor_id]
                    
                    dist_um = np.sqrt(
                        ((coord_a[0] - coord_b[0]) * voxel_size_L2[0]) ** 2 +
                        ((coord_a[1] - coord_b[1]) * voxel_size_L2[1]) ** 2 +
                        ((coord_a[2] - coord_b[2]) * voxel_size_L2[2]) ** 2
                    )
                    
                    edges.append({
                        'id': len(edges),
                        'node_a': int(edge_key[0]),
                        'node_b': int(edge_key[1]),
                        'distance_um': float(dist_um)
                    })
    
    print(f"Found {len(edges):,} edges")
    
    # Step 7: Build nodes with degree info
    print("Building node list with degree info...")
    nodes = []
    num_endpoints = 0
    num_bifurcations = 0
    
    for node_id, (z, y, x) in enumerate(coords_zyx):
        degree = len(neighbor_lists[node_id])
        is_endpoint = (degree == 1)
        is_bifurcation = (degree >= 3)
        
        if is_endpoint:
            num_endpoints += 1
        if is_bifurcation:
            num_bifurcations += 1
        
        # World coordinates (x, y, z) in μm
        coord_world = [
            float(x * voxel_size_L2[2]),  # X
            float(y * voxel_size_L2[1]),  # Y
            float(z * voxel_size_L2[0])   # Z
        ]
        
        # Label at this voxel
        label_id = int(skeleton_labels[z, y, x])
        
        nodes.append({
            'id': int(node_id),
            'coord_world': coord_world,
            'coord_voxel': [int(z), int(y), int(x)],
            'label_id': label_id,
            'degree': int(degree),
            'is_endpoint': is_endpoint,
            'is_bifurcation': is_bifurcation,
            'neighbors': [int(n) for n in neighbor_lists[node_id]]
        })
    
    # Step 8: Apply sampling if needed (keep keypoints!)
    if sampling_rate > 1:
        print(f"Applying sampling (1/{sampling_rate}), preserving keypoints...")
        
        # Always keep endpoints and bifurcations
        sampled_nodes = []
        old_to_new_id = {}
        
        for node in nodes:
            keep = (
                node['is_endpoint'] or 
                node['is_bifurcation'] or 
                node['id'] % sampling_rate == 0
            )
            
            if keep:
                new_id = len(sampled_nodes)
                old_to_new_id[node['id']] = new_id
                node_copy = node.copy()
                node_copy['id'] = new_id
                sampled_nodes.append(node_copy)
        
        # Rebuild edges for sampled nodes
        sampled_edges = []
        for edge in edges:
            if edge['node_a'] in old_to_new_id and edge['node_b'] in old_to_new_id:
                sampled_edges.append({
                    'id': len(sampled_edges),
                    'node_a': old_to_new_id[edge['node_a']],
                    'node_b': old_to_new_id[edge['node_b']],
                    'distance_um': edge['distance_um']
                })
        
        # Update neighbor lists
        for node in sampled_nodes:
            node['neighbors'] = [
                old_to_new_id[n] for n in node['neighbors'] 
                if n in old_to_new_id
            ]
            node['degree'] = len(node['neighbors'])
        
        print(f"After sampling: {len(sampled_nodes):,} nodes, {len(sampled_edges):,} edges")
        nodes = sampled_nodes
        edges = sampled_edges
        
        # Recount
        num_endpoints = sum(1 for n in nodes if n['is_endpoint'])
        num_bifurcations = sum(1 for n in nodes if n['is_bifurcation'])
    
    # Step 9: Calculate total skeleton length
    total_length_um = sum(e['distance_um'] for e in edges)
    
    # Step 10: Build summary
    t_elapsed = time.time() - t_start
    
    skeleton_graph = {
        'nodes': nodes,
        'edges': edges,
        'summary': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'num_endpoints': num_endpoints,
            'num_bifurcations': num_bifurcations,
            'total_length_um': float(total_length_um),
            'voxel_size_L2': list(voxel_size_L2),
            'processing_level': processing_level,
            'label_names': label_names,
            'build_time_sec': float(t_elapsed)
        }
    }
    
    print(f"\n=== Skeleton Graph Summary ===")
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Endpoints: {num_endpoints}")
    print(f"  Bifurcations: {num_bifurcations}")
    print(f"  Total length: {total_length_um/1000:.2f} mm")
    print(f"  Build time: {t_elapsed:.1f}s")
    print("✓ Skeleton graph complete")
    
    return skeleton_graph


def extract_skeleton_points_for_kdtree(skeleton_L2, outer_masks_L2, label_names, voxel_size_L2, 
                                        labels_L0, downsample_factor=4, processing_level=2,
                                        target_max_points=100000):
    """
    Extract skeleton points with mapping to Level 0 CC
    
    Each skeleton point @ Level 2 corresponds to downsample_factor^3 Level 0 voxels.
    Collects all intersecting (z, cc_id) pairs, sorts by z and takes middle.
    
    Also computes 26-connectivity neighbors and per-label instance segmentation.
    
    Parameters:
    -----------
    skeleton_L2 : np.array
        Skeleton @ Level 2
    outer_masks_L2 : dict
        {label_name: mask_array} @ Level 2
    label_names : list
        Label names
    voxel_size_L2 : tuple
        (vz, vy, vx) in μm @ Level 2
    labels_L0 : np.array
        Combined labels @ Level 0 (for CC mapping)
    downsample_factor : int
        Factor between L0 and L2 (default 4)
    processing_level : int
        Processing level (default 2)
    target_max_points : int
        Maximum points to extract (warning if exceeded)
    
    Returns:
    --------
    skeleton_points : dict
        Skeleton points data for KD-Tree with neighbor and instance info
    """
    print("\n=== Extracting Skeleton Points for KD-Tree @ L2 ===")
    
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        return None
    
    coords_zyx = np.argwhere(skeleton_binary)
    sampling_rate = 1
    
    if total_voxels <= target_max_points:
        pass
    else:
        print(f"  ⚠ Large skeleton ({total_voxels:,} voxels), may be slow")
    
    n_points = len(coords_zyx)
    print(f"Sampled points: {n_points:,}")
    
    # Assign labels from L2 masks
    print("Assigning labels...")
    labels = np.zeros(n_points, dtype=np.uint8)
    skeleton_labels_vol = np.zeros_like(skeleton_binary, dtype=np.uint8)
    
    for label_idx, label_name in enumerate(label_names, start=1):
        available = (skeleton_binary > 0) & (skeleton_labels_vol == 0)
        mask = outer_masks_L2[label_name] > 0
        intersection = available & mask
        skeleton_labels_vol[intersection] = label_idx
        print(f"  {label_name}: {np.sum(intersection):,} voxels")
    
    for i, (z, y, x) in enumerate(coords_zyx):
        labels[i] = skeleton_labels_vol[z, y, x]
    
    # === Map to Level 0 CC ===
    print("Mapping skeleton points to Level 0 CC...")
    
    cc_ids_L0 = np.zeros(n_points, dtype=np.int32)
    z_L0_arr = np.zeros(n_points, dtype=np.int32)
    
    factor = downsample_factor
    
    # Cache: (label_idx, z_L0) → labeled array
    labeled_cache = {}
    
    L0_shape = labels_L0.shape
    print(f"  Level 0 shape: {L0_shape}")
    
    for i, (z_L2, y_L2, x_L2) in enumerate(tqdm(coords_zyx, desc="Mapping to L0")):
        label_idx = labels[i]
        
        if label_idx == 0:
            cc_ids_L0[i] = -1
            z_L0_arr[i] = z_L2 * factor + factor // 2
            continue
        
        # L2 voxel → factor^3 L0 voxels
        z0_start = z_L2 * factor
        y0_start = y_L2 * factor
        x0_start = x_L2 * factor
        
        z0_end = min(z0_start + factor, L0_shape[0])
        y0_end = min(y0_start + factor, L0_shape[1])
        x0_end = min(x0_start + factor, L0_shape[2])
        
        # Collect all intersecting (z, cc_id) pairs
        intersections = []
        
        for z0 in range(z0_start, z0_end):
            cache_key = (label_idx, z0)
            if cache_key not in labeled_cache:
                slice_data = labels_L0[z0]
                if hasattr(slice_data, 'compute'):
                    slice_data = slice_data.compute()
                label_mask = (slice_data == label_idx)
                if label_mask.any():
                    labeled, _ = cc_label(label_mask)
                    labeled_cache[cache_key] = labeled
                else:
                    labeled_cache[cache_key] = None
            
            labeled = labeled_cache[cache_key]
            if labeled is None:
                continue
            
            # Check factor x factor region
            for y0 in range(y0_start, y0_end):
                for x0 in range(x0_start, x0_end):
                    if y0 < labeled.shape[0] and x0 < labeled.shape[1]:
                        cc_id_1indexed = labeled[y0, x0]
                        if cc_id_1indexed > 0:
                            intersections.append((z0, cc_id_1indexed - 1))
        
        if intersections:
            # Sort by z, take middle
            intersections.sort(key=lambda x: x[0])
            mid_idx = len(intersections) // 2
            z_L0_arr[i] = intersections[mid_idx][0]
            cc_ids_L0[i] = intersections[mid_idx][1]
        else:
            z_L0_arr[i] = z0_start + factor // 2
            cc_ids_L0[i] = -1
    
    # World coordinates
    coords_world = np.zeros((n_points, 3), dtype=np.float32)
    coords_world[:, 0] = coords_zyx[:, 2] * voxel_size_L2[2]
    coords_world[:, 1] = coords_zyx[:, 1] * voxel_size_L2[1]
    coords_world[:, 2] = coords_zyx[:, 0] * voxel_size_L2[0]
    
    # Summary
    valid_cc = np.sum(cc_ids_L0 >= 0)
    invalid_cc = np.sum(cc_ids_L0 < 0)
    print(f"\n=== Mapping Summary ===")
    print(f"  Valid: {valid_cc:,} ({100*valid_cc/n_points:.1f}%)")
    print(f"  Invalid: {invalid_cc:,} ({100*invalid_cc/n_points:.1f}%)")
    
    # =============================================================
    # Compute 26-connectivity neighbors
    # =============================================================
    print("\n=== Computing 26-connectivity Neighbors ===")
    
    # Build voxel coord → index lookup
    voxel_to_idx = {tuple(coord): i for i, coord in enumerate(coords_zyx)}
    
    # 26-connectivity offsets
    offsets_26 = [(dz, dy, dx) 
                  for dz in [-1, 0, 1] 
                  for dy in [-1, 0, 1] 
                  for dx in [-1, 0, 1]
                  if not (dz == 0 and dy == 0 and dx == 0)]
    
    neighbors_list = []
    for i, (z, y, x) in enumerate(tqdm(coords_zyx, desc="Finding neighbors")):
        point_neighbors = []
        for dz, dy, dx in offsets_26:
            neighbor_key = (z + dz, y + dy, x + dx)
            if neighbor_key in voxel_to_idx:
                point_neighbors.append(voxel_to_idx[neighbor_key])
        neighbors_list.append(point_neighbors)
    
    avg_neighbors = np.mean([len(n) for n in neighbors_list])
    isolated_points = sum(1 for n in neighbors_list if len(n) == 0)
    print(f"  Average neighbors per point: {avg_neighbors:.1f}")
    print(f"  Isolated points (no neighbors): {isolated_points}")
    
    # =============================================================
    # Per-label Instance segmentation (BFS on neighbor graph)
    # =============================================================
    print("\n=== Segmenting Skeleton into Instances ===")
    
    instance_ids = np.full(n_points, -1, dtype=np.int32)
    instances_info = []
    instance_counter = 0
    
    for label_idx, label_name in enumerate(label_names, start=1):
        # Find all points for this label
        label_point_indices = np.where(labels == label_idx)[0]
        
        if len(label_point_indices) == 0:
            print(f"  {label_name}: 0 points, skipping")
            continue
        
        print(f"  {label_name}: {len(label_point_indices)} points")
        
        # BFS to find connected components
        visited = set()
        label_instances = 0
        
        for start_idx in label_point_indices:
            if start_idx in visited:
                continue
            
            # BFS to find connected component
            component = []
            queue = [int(start_idx)]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if labels[current] != label_idx:
                    continue
                    
                visited.add(current)
                component.append(current)
                
                # Only traverse same-label neighbors
                for neighbor_idx in neighbors_list[current]:
                    if neighbor_idx not in visited and labels[neighbor_idx] == label_idx:
                        queue.append(neighbor_idx)
            
            if component:
                # Assign instance ID
                for idx in component:
                    instance_ids[idx] = instance_counter
                
                # Collect instance info
                component_coords = coords_zyx[component]
                z_min = int(component_coords[:, 0].min())
                z_max = int(component_coords[:, 0].max())
                
                # Collect (z_L0, cc_id) pairs for this instance
                z_cc_pairs = []
                for idx in component:
                    if cc_ids_L0[idx] >= 0:
                        z_cc_pairs.append((int(z_L0_arr[idx]), int(cc_ids_L0[idx])))
                
                instances_info.append({
                    'id': instance_counter,
                    'label_idx': int(label_idx),
                    'label_name': label_name,
                    'point_count': len(component),
                    'point_indices': component,  # Temporary, used for mesh building
                    'z_range_L2': [z_min, z_max],
                    'z_cc_pairs': list(set(z_cc_pairs)),
                })
                
                instance_counter += 1
                label_instances += 1
        
        print(f"    → {label_instances} instances")
    
    print(f"\n  Total instances: {instance_counter}")
    
    # =============================================================
    # Build result
    # =============================================================
    result = {
        'coords_world': coords_world.tolist(),
        'coords_voxel': coords_zyx.tolist(),
        'labels': labels.tolist(),
        'cc_ids': cc_ids_L0.tolist(),
        'z_L0': z_L0_arr.tolist(),
        'neighbors': neighbors_list,
        'instance_ids': instance_ids.tolist(),
        'instances': instances_info,
        'label_names': label_names,
        'voxel_size_L2': list(voxel_size_L2),
        'processing_level': processing_level,
        'downsample_factor': downsample_factor,
        'total_raw_voxels': int(total_voxels),
        'total_sampled_points': n_points,
        'sampling_rate': sampling_rate,
    }
    
    print("✓ Done")
    return result
