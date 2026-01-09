"""
Mesh Building Functions

Prebuild meshes for different visualization modes:
- Morphology meshes for Skeleton Host
- Instance meshes for skeleton instances
- Image Host meshes for CC-based navigation
"""

import json
import numpy as np
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import pyvista as pv


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
            
            # Label 2D CCs, use seeds to find corresponding CC
            labeled, _ = ndimage.label(slice_mask)
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
    
    print(f"✓ Skeleton instance meshes complete")


def prebuild_image_host_meshes(outer_masks_L2, inner_masks_L2, label_names, 
                                voxel_size_L2, output_dir, processing_level=2):
    """
    Prebuild meshes for Image Host mode (outer + inner per 3D CC)
    
    For each label:
    1. 3D connected component analysis
    2. Build outer + inner mesh for each 3D CC
    3. Build (z, 2d_cc_id) → 3d_cc_id lookup table
    
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
    
    Returns:
    --------
    result : dict
        Mesh info with lookup tables
    """
    mesh_dir = Path(output_dir) / "image_host_meshes"
    mesh_dir.mkdir(exist_ok=True)
    
    print("\n=== Prebuilding Image Host Meshes @ L2 ===")
    
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
        
        # 3D connected component analysis (using outer mask)
        labeled_3d, num_cc = ndimage.label(outer_mask)
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
        
        for cc_id in tqdm(range(1, num_cc + 1), desc=f"    Building {label_name} meshes"):
            cc_mask_outer = (labeled_3d == cc_id)
            voxel_count = np.sum(cc_mask_outer)
            
            if voxel_count < 10:
                continue
            
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
            
            # === Outer Mesh ===
            try:
                padded = np.zeros((nz + 2*pad, ny + 2*pad, nx + 2*pad), dtype=np.uint8)
                padded[pad:pad+nz, pad:pad+ny, pad:pad+nx] = cc_mask_outer
                
                # Transpose to (X, Y, Z) for PyVista
                data_t = np.transpose(padded.astype(float), (2, 1, 0))
                
                origin = (
                    -pad * voxel_size_L2[2],
                    -pad * voxel_size_L2[1],
                    -pad * voxel_size_L2[0]
                )
                
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
                print(f"      CC {cc_id} outer failed: {e}")
            
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
                            
                    except Exception as e:
                        print(f"      CC {cc_id} inner failed: {e}")
            
            if mesh_info['outer_mesh']:
                cc_info['meshes'].append(mesh_info)
        
        # === Build z → 2d_cc → 3d_cc lookup table ===
        print(f"    Building lookup table...")
        for z in range(outer_mask.shape[0]):
            slice_mask = outer_mask[z]
            if not slice_mask.any():
                continue
            
            labeled_2d, _ = ndimage.label(slice_mask)
            
            z_lookup = {}
            for cc_2d in range(1, labeled_2d.max() + 1):
                # Find which 3D CC this 2D CC belongs to
                cc_2d_mask = (labeled_2d == cc_2d)
                # Take first point's 3D CC ID
                y, x = np.argwhere(cc_2d_mask)[0]
                cc_3d = labeled_3d[z, y, x]
                if cc_3d > 0:
                    z_lookup[str(cc_2d - 1)] = int(cc_3d)  # 0-indexed key as string
            
            if z_lookup:
                cc_info['z_2dcc_to_3dcc'][str(z)] = z_lookup
        
        result['labels'][label_name] = cc_info
        print(f"    ✓ {len(cc_info['meshes'])} meshes saved")
    
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
