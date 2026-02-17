"""
Skeleton Processing

Functions for skeleton intersection analysis, graph building,
and KD-Tree point extraction.

OPTIMIZED VERSION:
- Vectorized edge/neighbor computation (NumPy)
- Union-Find for fast instance segmentation
- Shared skeleton_labels_vol computation
- GPU-accelerated CC labeling (CuPy)
"""

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial import KDTree
from tqdm import tqdm
import time
import gc

from .config import GPU_AVAILABLE, get_cupy

# GPU-accelerated CC labeling via CuPy (Windows compatible)
CUPY_LABEL_AVAILABLE = False
if GPU_AVAILABLE:
    try:
        cp = get_cupy()
        if cp is not None:
            from cupyx.scipy.ndimage import label as cupy_label
            CUPY_LABEL_AVAILABLE = True
            print("[OK] CuPy label loaded - GPU CC labeling available (Windows compatible)")
    except ImportError:
        pass

if not CUPY_LABEL_AVAILABLE:
    print("[INFO] GPU CC labeling not available - using CPU (scipy)")


def _cc_label_2d(mask, use_gpu=True):
    """
    2D Connected Component labeling with GPU acceleration if available
    """
    if use_gpu and CUPY_LABEL_AVAILABLE:
        cp = get_cupy()
        if cp is not None:
            try:
                mask_gpu = cp.asarray(mask)
                labeled_gpu, num_features = cupy_label(mask_gpu)
                labeled = cp.asnumpy(labeled_gpu)
                del mask_gpu, labeled_gpu
                return labeled, int(num_features)
            except Exception:
                pass  # Fall back to CPU
    
    # CPU fallback
    return cc_label(mask)


def compute_skeleton_labels(skeleton_binary, outer_masks_L2, label_names):
    """
    Compute skeleton label assignment (shared computation)
    
    This is called ONCE and the result is passed to all functions
    that need skeleton labels, avoiding redundant computation.
    
    Parameters:
    -----------
    skeleton_binary : np.ndarray
        Binary skeleton volume
    outer_masks_L2 : dict
        {label_name: mask_array}
    label_names : list
        Label names
    
    Returns:
    --------
    skeleton_labels : np.ndarray
        Label assignment for each skeleton voxel (0 = unassigned)
    label_counts : dict
        {label_name: voxel_count}
    """
    skeleton_labels = np.zeros_like(skeleton_binary, dtype=np.uint8)
    label_counts = {}
    
    for label_idx, label_name in enumerate(label_names, start=1):
        available = (skeleton_binary > 0) & (skeleton_labels == 0)
        mask = outer_masks_L2[label_name] > 0
        intersection = available & mask
        skeleton_labels[intersection] = label_idx
        label_counts[label_name] = int(np.sum(intersection))
    
    return skeleton_labels, label_counts


def preprocess_skeleton_intersections(skeleton_L2, outer_masks_L2, label_names, label_colors,
                                       skeleton_labels=None):
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
    skeleton_labels : np.ndarray or None
        Pre-computed skeleton labels (optional, will compute if None)
    
    Returns:
    --------
    results : dict
        {label_name: {voxels, proportion, color}}
    skeleton_labels : np.ndarray
        Computed skeleton labels (for reuse)
    """
    print("\n=== Simple Skeleton Intersection @ L2 ===")
    
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        return {}, None
    
    # Compute or reuse skeleton labels
    if skeleton_labels is None:
        skeleton_labels, label_counts = compute_skeleton_labels(
            skeleton_binary, outer_masks_L2, label_names
        )
    else:
        # Recompute counts from existing labels
        label_counts = {}
        for label_idx, label_name in enumerate(label_names, start=1):
            label_counts[label_name] = int(np.sum(skeleton_labels == label_idx))
    
    results = {}
    print("Processing labels...")
    for label_name in label_names:
        count = label_counts[label_name]
        proportion = 100.0 * count / total_voxels
        print(f"  {label_name}: {count:,} voxels ({proportion:.2f}%)")
        
        results[label_name] = {
            'voxels': count,
            'proportion': float(proportion),
            'color': label_colors[label_name]
        }
    
    unclassified = np.sum(skeleton_labels == 0)
    print(f"  Unclassified: {unclassified:,} voxels ({100.0*unclassified/total_voxels:.2f}%)")
    print("✓ Done")
    
    return results, skeleton_labels


def _build_edges_vectorized(coords_zyx, voxel_size_L2):
    """
    Vectorized edge building using NumPy broadcasting
    
    Instead of O(N × 26) Python loops, uses hash-based lookup with NumPy
    """
    n_voxels = len(coords_zyx)
    
    # Build coordinate → index lookup using structured array for fast hashing
    # Create a view of coords as a structured array for hashing
    coords_tuple = [tuple(c) for c in coords_zyx]
    voxel_to_node = {c: i for i, c in enumerate(coords_tuple)}
    
    # 26-connectivity offsets
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append([dz, dy, dx])
    offsets = np.array(offsets, dtype=np.int32)
    
    # Pre-compute all neighbor coordinates for all nodes at once
    # Shape: (n_voxels, 26, 3)
    all_neighbor_coords = coords_zyx[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    
    # Build neighbor lists and edges
    edges = []
    edge_set = set()
    neighbor_lists = [[] for _ in range(n_voxels)]
    
    # Process in batches for progress display
    batch_size = 10000
    num_batches = (n_voxels + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Finding edges (vectorized)", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_voxels)
        
        for node_id in range(start_idx, end_idx):
            for offset_idx in range(26):
                neighbor_coord = tuple(all_neighbor_coords[node_id, offset_idx])
                
                if neighbor_coord in voxel_to_node:
                    neighbor_id = voxel_to_node[neighbor_coord]
                    neighbor_lists[node_id].append(neighbor_id)
                    
                    # Add edge (avoid duplicates)
                    edge_key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        
                        # Calculate distance
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
    
    return edges, neighbor_lists


class UnionFind:
    """
    Disjoint Set Union (Union-Find) for fast connected component detection
    
    Much faster than BFS for finding connected components in a graph
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def _segment_instances_union_find(labels, neighbors_list, label_names):
    """
    OPTIMIZED: Instance segmentation using Union-Find
    
    Much faster than BFS for large graphs
    """
    n_points = len(labels)
    
    instance_ids = np.full(n_points, -1, dtype=np.int32)
    instances_info = []
    
    for label_idx, label_name in enumerate(label_names, start=1):
        label_point_indices = np.where(labels == label_idx)[0]
        
        if len(label_point_indices) == 0:
            print(f"  {label_name}: 0 points, skipping")
            continue
        
        print(f"  {label_name}: {len(label_point_indices)} points")
        
        # Create Union-Find for this label's points
        # Map global indices to local indices
        global_to_local = {g: l for l, g in enumerate(label_point_indices)}
        local_to_global = {l: g for l, g in enumerate(label_point_indices)}
        
        uf = UnionFind(len(label_point_indices))
        
        # Union neighbors within same label
        for local_idx, global_idx in enumerate(label_point_indices):
            for neighbor_global in neighbors_list[global_idx]:
                if labels[neighbor_global] == label_idx:
                    neighbor_local = global_to_local.get(neighbor_global)
                    if neighbor_local is not None:
                        uf.union(local_idx, neighbor_local)
        
        # Group by component
        components = {}
        for local_idx in range(len(label_point_indices)):
            root = uf.find(local_idx)
            if root not in components:
                components[root] = []
            components[root].append(local_to_global[local_idx])
        
        # Create instances
        label_instances = 0
        for component in components.values():
            instance_id = len(instances_info)
            
            for idx in component:
                instance_ids[idx] = instance_id
            
            instances_info.append({
                'id': instance_id,
                'label_idx': int(label_idx),
                'label_name': label_name,
                'point_count': len(component),
                'point_indices': component,
            })
            
            label_instances += 1
        
        print(f"    → {label_instances} instances")
    
    return instance_ids, instances_info


def build_skeleton_graph(skeleton_L2, outer_masks_L2, label_names, voxel_size_L2,
                         processing_level=2, target_max_points=50000,
                         skeleton_labels=None):
    """
    Build skeleton graph with connectivity, distances, and branch detection
    
    OPTIMIZED VERSION:
    - Vectorized edge building
    - Reuses pre-computed skeleton_labels
    
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
    skeleton_labels : np.ndarray or None
        Pre-computed skeleton labels (optional)
    
    Returns:
    --------
    skeleton_graph : dict
        Graph structure with nodes, edges, and summary
    """
    print("\n=== Building Skeleton Graph @ L2 ===")
    print("  Mode: Optimized (vectorized edges)")
    t_start = time.time()
    
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        print("⚠️ No skeleton voxels!")
        return None
    
    # Adaptive sampling (if needed)
    sampling_rate = 1
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
    
    # Get voxel coordinates
    coords_zyx = np.argwhere(skeleton_binary)  # (N, 3) in [z, y, x]
    n_voxels = len(coords_zyx)
    print(f"Skeleton voxels: {n_voxels:,}")
    
    # Compute or reuse skeleton labels
    if skeleton_labels is None:
        print("Assigning labels to skeleton voxels...")
        skeleton_labels, _ = compute_skeleton_labels(
            skeleton_binary, outer_masks_L2, label_names
        )
    else:
        print("Using pre-computed skeleton labels")
    
    # OPTIMIZED: Vectorized edge building
    edges, neighbor_lists = _build_edges_vectorized(coords_zyx, voxel_size_L2)
    print(f"Found {len(edges):,} edges")
    
    # Build nodes with degree info
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
    
    # Apply sampling if needed (keep keypoints!)
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
    
    # Calculate total length
    total_length_um = sum(e['distance_um'] for e in edges)
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


def _calculate_dynamic_cache_size(slice_shape, n_labels):
    """
    Dynamically calculate optimal cache size based on available memory
    
    Parameters:
    -----------
    slice_shape : tuple
        (height, width) of a single slice
    n_labels : int
        Number of unique labels
    
    Returns:
    --------
    max_cache_size : int
        Maximum number of Z slices to keep in cache
    """
    import gc
    
    # Estimate memory per slice (int32 labeled array)
    bytes_per_slice = slice_shape[0] * slice_shape[1] * 4  # int32 = 4 bytes
    # Each Z slice might have multiple label CC arrays
    avg_labels_per_slice = min(n_labels, 10)  # Assume ~10 labels per slice on average
    bytes_per_z = bytes_per_slice * avg_labels_per_slice
    
    # Check available memory
    try:
        import psutil
        available_mem = psutil.virtual_memory().available
        # Use at most 50% of available memory for cache
        target_mem = available_mem * 0.5
    except ImportError:
        # Fallback: assume 4GB available, use 2GB
        target_mem = 2 * (1024 ** 3)
    
    # Check GPU memory if available
    if GPU_AVAILABLE:
        try:
            cp = get_cupy()
            if cp is not None:
                device = cp.cuda.Device()
                free_gpu_mem = device.mem_info[0]
                # GPU memory is usually more constrained, use 30%
                gpu_target = free_gpu_mem * 0.3
                # Use the smaller of CPU and GPU limits
                target_mem = min(target_mem, gpu_target)
        except:
            pass
    
    # Calculate max cache size
    max_cache_size = max(8, int(target_mem / bytes_per_z))
    
    # Cap at reasonable maximum (don't cache more than 256 slices)
    max_cache_size = min(max_cache_size, 256)
    
    return max_cache_size


def _map_points_to_L0_cc_batch(coords_zyx, labels, labels_L0, factor, use_gpu=True):
    """
    Batch mapping of skeleton points to Level 0 CC
    
    Strategy:
    1. Group points by (label_idx, z_L2) for batch processing
    2. Pre-compute all needed L0 CC labels for each Z slice
    3. Vectorized lookup instead of per-point loops
    
    
    - Dynamic cache size based on available RAM/GPU memory
    - LRU eviction when cache is full
    """
    n_points = len(coords_zyx)
    L0_shape = labels_L0.shape
    
    cc_ids_L0 = np.full(n_points, -1, dtype=np.int32)
    z_L0_arr = np.zeros(n_points, dtype=np.int32)
    
    # Get unique (label, z_L2) combinations
    unique_labels = np.unique(labels[labels > 0])
    
    # Group points by z_L2 for processing
    z_L2_all = coords_zyx[:, 0]
    z_L2_unique = np.unique(z_L2_all)
    
    # DYNAMIC CACHE SIZE based on available memory
    slice_shape = (L0_shape[1], L0_shape[2])
    MAX_CACHE_SIZE = _calculate_dynamic_cache_size(slice_shape, len(unique_labels))
    
    print(f"    Processing {n_points:,} points across {len(z_L2_unique)} Z slices...")
    print(f"    Mode: On-demand CC labeling (dynamic cache)")
    print(f"    Cache size: {MAX_CACHE_SIZE} Z slices (auto-calculated based on available memory)")
    
    # LRU cache for recently used slices
    cc_cache = {}  # {z0: {label_idx: labeled_array}}
    cache_order = []
    
    def get_cc_label(z0, label_idx):
        """Get CC label for a specific (z0, label_idx), with caching"""
        nonlocal cc_cache, cache_order
        
        if z0 not in cc_cache:
            # Load slice data
            slice_data = labels_L0[z0]
            if hasattr(slice_data, 'compute'):
                slice_data = slice_data.compute()
            
            cc_cache[z0] = {}
            cache_order.append(z0)
            
            # Evict old cache entries if over limit
            while len(cc_cache) > MAX_CACHE_SIZE:
                oldest = cache_order.pop(0)
                del cc_cache[oldest]
                # Force garbage collection for large arrays
                import gc
                gc.collect()
        
        if label_idx not in cc_cache[z0]:
            # Compute CC for this label
            slice_data = labels_L0[z0]
            if hasattr(slice_data, 'compute'):
                slice_data = slice_data.compute()
            
            mask = slice_data == label_idx
            if np.any(mask):
                labeled, _ = _cc_label_2d(mask, use_gpu=use_gpu)
                cc_cache[z0][label_idx] = labeled
            else:
                cc_cache[z0][label_idx] = None
        
        return cc_cache[z0][label_idx]
    
    for z_L2 in tqdm(z_L2_unique, desc="    Mapping by Z slice", unit="z"):
        # Get all points at this z_L2
        point_mask = z_L2_all == z_L2
        point_indices = np.where(point_mask)[0]
        
        if len(point_indices) == 0:
            continue
        
        # L0 Z range for this L2 slice
        z0_start = z_L2 * factor
        z0_end = min(z0_start + factor, L0_shape[0])
        z0_mid = z0_start + factor // 2
        
        for i in point_indices:
            label_idx = labels[i]
            
            if label_idx == 0:
                z_L0_arr[i] = z0_mid
                continue
            
            y_L2, x_L2 = coords_zyx[i, 1], coords_zyx[i, 2]
            y0_start = y_L2 * factor
            x0_start = x_L2 * factor
            y0_end = min(y0_start + factor, L0_shape[1])
            x0_end = min(x0_start + factor, L0_shape[2])
            
            # Find first valid CC in the L0 region
            found = False
            for z0 in range(z0_start, z0_end):
                labeled = get_cc_label(z0, label_idx)
                if labeled is None:
                    continue
                
                # Check the center point first (most likely to hit)
                y0_center = (y0_start + y0_end) // 2
                x0_center = (x0_start + x0_end) // 2
                
                if 0 <= y0_center < labeled.shape[0] and 0 <= x0_center < labeled.shape[1]:
                    cc_id = labeled[y0_center, x0_center]
                    if cc_id > 0:
                        cc_ids_L0[i] = cc_id - 1
                        z_L0_arr[i] = z0
                        found = True
                        break
                
                # If center didn't hit, check the region
                region = labeled[y0_start:y0_end, x0_start:x0_end]
                if region.size > 0:
                    valid_ccs = region[region > 0]
                    if len(valid_ccs) > 0:
                        # Take the most common CC in this region
                        cc_id = np.bincount(valid_ccs).argmax()
                        cc_ids_L0[i] = cc_id - 1
                        z_L0_arr[i] = z0
                        found = True
                        break
            
            if not found:
                z_L0_arr[i] = z0_mid
        
        # Clear cache periodically to prevent memory buildup
        if len(cc_cache) > MAX_CACHE_SIZE // 2:
            # Free GPU memory
            if use_gpu and GPU_AVAILABLE:
                cp = get_cupy()
                if cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
    
    # Final cleanup
    cc_cache.clear()
    import gc
    gc.collect()
    
    if use_gpu and GPU_AVAILABLE:
        cp = get_cupy()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    
    return cc_ids_L0, z_L0_arr


def extract_skeleton_points_for_kdtree(skeleton_L2, outer_masks_L2, label_names, voxel_size_L2, 
                                        labels_L0, downsample_factor=4, processing_level=2,
                                        target_max_points=100000, skeleton_labels=None):
    """
    Extract skeleton points with mapping to Level 0 CC
    
    OPTIMIZED VERSION v2:
    - Batch pre-computation of L0 CC labels
    - Group processing by Z slice
    - Vectorized neighbor computation
    - Union-Find for instance segmentation
    - GPU-accelerated CC labeling
    
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
    skeleton_labels : np.ndarray or None
        Pre-computed skeleton labels (optional)
    
    Returns:
    --------
    skeleton_points : dict
        Skeleton points data for KD-Tree with neighbor and instance info
    """
    print("\n=== Extracting Skeleton Points for KD-Tree @ L2 ===")
    print(f"  CC Labeling: {'GPU (CuPy)' if CUPY_LABEL_AVAILABLE else 'CPU (scipy)'}")
    print(f"  Mode: Optimized (batch L0 CC, grouped by Z)")
    
    skeleton_binary = skeleton_L2 > 0
    total_voxels = np.sum(skeleton_binary)
    print(f"Total skeleton voxels: {total_voxels:,}")
    
    if total_voxels == 0:
        return None
    
    coords_zyx = np.argwhere(skeleton_binary)
    sampling_rate = 1
    
    if total_voxels > target_max_points:
        print(f"  ⚠ Large skeleton ({total_voxels:,} voxels)")
    
    n_points = len(coords_zyx)
    print(f"Sampled points: {n_points:,}")
    
    # Compute or reuse skeleton labels
    if skeleton_labels is None:
        print("Assigning labels...")
        skeleton_labels_vol, _ = compute_skeleton_labels(
            skeleton_binary, outer_masks_L2, label_names
        )
    else:
        print("Using pre-computed skeleton labels")
        skeleton_labels_vol = skeleton_labels
    
    # Extract labels for each point (vectorized)
    labels = skeleton_labels_vol[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]].astype(np.uint8)
    
    # === OPTIMIZED: Batch mapping to Level 0 CC ===
    print("Mapping skeleton points to Level 0 CC (batch mode)...")
    
    factor = downsample_factor
    L0_shape = labels_L0.shape
    print(f"  Level 0 shape: {L0_shape}")
    
    cc_ids_L0, z_L0_arr = _map_points_to_L0_cc_batch(
        coords_zyx, labels, labels_L0, factor, 
        use_gpu=CUPY_LABEL_AVAILABLE
    )
    
    # Cleanup GPU memory
    if GPU_AVAILABLE:
        cp = get_cupy()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    
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
    
    # ===        Vectorized neighbor computation       ===
    print("\n=== Computing 26-connectivity Neighbors (Vectorized) ===")
    
    _, neighbors_list = _build_edges_vectorized(coords_zyx, voxel_size_L2)
    
    avg_neighbors = np.mean([len(n) for n in neighbors_list])
    isolated_points = sum(1 for n in neighbors_list if len(n) == 0)
    print(f"  Average neighbors per point: {avg_neighbors:.1f}")
    print(f"  Isolated points (no neighbors): {isolated_points}")
    
    # ===       Instance segmentation using Union-Find       ===
    print("\n=== Segmenting Skeleton into Instances (Union-Find) ===")
    
    instance_ids, instances_info = _segment_instances_union_find(
        labels, neighbors_list, label_names
    )
    
    # Add z_range and z_cc_pairs to instances
    for inst in instances_info:
        component = inst['point_indices']
        component_coords = coords_zyx[component]
        z_min = int(component_coords[:, 0].min())
        z_max = int(component_coords[:, 0].max())
        inst['z_range_L2'] = [z_min, z_max]
        
        z_cc_pairs = []
        for idx in component:
            if cc_ids_L0[idx] >= 0:
                z_cc_pairs.append((int(z_L0_arr[idx]), int(cc_ids_L0[idx])))
        inst['z_cc_pairs'] = list(set(z_cc_pairs))
    
    print(f"\n  Total instances: {len(instances_info)}")
    
    # === Spatial CC: label-agnostic 3D connected components ===
    print("\n=== Computing Spatial CCs (label-agnostic, 26-connectivity) ===")
    from scipy.ndimage import generate_binary_structure, label as ndimage_label
    struct_26 = generate_binary_structure(3, 3)
    spatial_cc_vol, num_spatial_cc = ndimage_label(skeleton_binary, structure=struct_26)
    print(f"  Found {num_spatial_cc} spatial CCs")
    
    # Map each skeleton point to its spatial CC id
    spatial_cc_ids = spatial_cc_vol[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]].astype(np.int32)
    
    # Summary per spatial CC
    spatial_cc_sizes = np.bincount(spatial_cc_ids[spatial_cc_ids > 0])
    if len(spatial_cc_sizes) > 1:
        print(f"  Size range: {spatial_cc_sizes[1:].min()} — {spatial_cc_sizes[1:].max()} points")
        print(f"  Top 5: {sorted(spatial_cc_sizes[1:], reverse=True)[:5]}")

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
        'spatial_cc_ids': spatial_cc_ids.tolist(),
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
