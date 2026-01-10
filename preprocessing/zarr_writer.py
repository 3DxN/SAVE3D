"""
Zarr Output Writer

Write preprocessed data to Zarr format with OME-NGFF metadata.
- Parallel writing of pyramid levels using ThreadPoolExecutor
- Chunked writing for large arrays
"""

import json
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _write_array_to_zarr(z_arr, arr, desc="Writing"):
    """Write array to zarr dataset with progress"""
    z_arr[:] = arr
    return arr.shape


def write_zarr_pathology(zarr_path, img_pyramid, lab_pyramid, 
                         outer_masks_L2, inner_masks_L2, skeleton_metadata,
                         voxel_size_L0, label_colors, label_names, legend_labels,
                         processing_level, downsample_factor, voxel_size_L2,
                         max_workers=4):
    """
    Write output Zarr with all data
    
    Uses parallel writing for pyramid levels
    
    Parameters:
    -----------
    zarr_path : str
        Output Zarr path
    img_pyramid : list
        Image pyramid (L0 + downsampled levels)
    lab_pyramid : list
        Label pyramid (L0 + downsampled levels)
    outer_masks_L2 : dict
        Outer masks @ Level 2
    inner_masks_L2 : dict
        Inner masks @ Level 2 (can be empty)
    skeleton_metadata : dict
        Skeleton intersection data
    voxel_size_L0 : tuple
        Voxel size @ Level 0
    label_colors : dict
        {label_name: hex_color}
    label_names : list
        Ordered list of label names
    legend_labels : list
        Labels to show in legend (user-named labels only)
    processing_level : int
        Processing level (e.g., 2)
    downsample_factor : int
        Downsample factor (e.g., 4)
    voxel_size_L2 : tuple
        Voxel size @ Level 2
    max_workers : int
        Max parallel workers (default 4)
    """
    import zarr
    import numcodecs
    
    print("\n=== Writing Zarr ===")
    print(f"  Mode: Optimized (parallel writing)")
    zarr_path = Path(zarr_path)
    
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    
    is_rgb = len(img_pyramid[0].shape) == 4 and img_pyramid[0].shape[-1] == 3
    store = zarr.open_group(str(zarr_path), mode='w')
    
    # Compression
    img_compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    lab_compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    
    # 1. Write image pyramid @ L0
    print("Writing image pyramid @ L0...")
    
    # Create all datasets first
    img_datasets = []
    for i, arr in enumerate(img_pyramid):
        if is_rgb:
            chunks = (min(16, arr.shape[0]), min(256, arr.shape[1]), 
                     min(256, arr.shape[2]), arr.shape[3])
        else:
            chunks = (min(16, arr.shape[0]), min(256, arr.shape[1]), min(256, arr.shape[2]))
        
        z_arr = store.create_dataset(str(i), shape=arr.shape, dtype=arr.dtype, 
                                    chunks=chunks, compressor=img_compressor)
        img_datasets.append((z_arr, arr, i))
    
    # Write in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_write_array_to_zarr, z_arr, arr, f"Level {i}"): i 
            for z_arr, arr, i in img_datasets
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Image pyramid", unit="level"):
            level = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  Level {level} failed: {e}")
    
    # 2. Write combined labels pyramid @ L0
    print("Writing combined labels pyramid @ L0...")
    seg_group = store.create_group("segmentation")
    
    lab_datasets = []
    for i, arr in enumerate(lab_pyramid):
        chunks = (min(16, arr.shape[0]), min(256, arr.shape[1]), min(256, arr.shape[2]))
        z_arr = seg_group.create_dataset(str(i), shape=arr.shape, dtype=arr.dtype, 
                                        chunks=chunks, compressor=lab_compressor)
        lab_datasets.append((z_arr, arr, i))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_write_array_to_zarr, z_arr, arr, f"Level {i}"): i 
            for z_arr, arr, i in lab_datasets
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Labels pyramid", unit="level"):
            level = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  Level {level} failed: {e}")
    
    # 3. Write outer masks @ L2
    print("Writing outer masks @ L2...")
    outer_group = store.create_group("outer_masks_L2")
    
    for label_name in tqdm(list(outer_masks_L2.keys()), desc="Outer masks", unit="label"):
        mask_data = outer_masks_L2[label_name]
        chunks = (min(16, mask_data.shape[0]), min(256, mask_data.shape[1]), min(256, mask_data.shape[2]))
        z_arr = outer_group.create_dataset(label_name, shape=mask_data.shape, dtype=mask_data.dtype,
                                          chunks=chunks, compressor=lab_compressor)
        z_arr[:] = mask_data
    
    # 4. Write inner masks @ L2 (if available)
    has_inner_mask = len(inner_masks_L2) > 0
    if has_inner_mask:
        print("Writing inner masks @ L2...")
        inner_group = store.create_group("inner_masks_L2")
        
        for label_name in tqdm(list(inner_masks_L2.keys()), desc="Inner masks", unit="label"):
            mask_data = inner_masks_L2[label_name]
            chunks = (min(16, mask_data.shape[0]), min(256, mask_data.shape[1]), min(256, mask_data.shape[2]))
            z_arr = inner_group.create_dataset(label_name, shape=mask_data.shape, dtype=mask_data.dtype,
                                              chunks=chunks, compressor=lab_compressor)
            z_arr[:] = mask_data
    else:
        print("No inner masks to write")
    
    # 5. Metadata
    store.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ] + ([{"name": "c", "type": "channel"}] if is_rgb else []),
        "datasets": [{"path": str(i)} for i in range(len(img_pyramid))]
    }]
    
    seg_group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": [{"path": str(i)} for i in range(len(lab_pyramid))]
    }]
    
    # 6. Save metadata.json
    metadata = {
        'resolution_levels': {
            'L0': {
                'description': 'Input image resolution (Level 0)',
                'voxel_size_um': list(voxel_size_L0)
            },
            'L2': {
                'description': f'Processing level (Level {processing_level} = {downsample_factor}x downsampled)',
                'voxel_size_um': list(voxel_size_L2),
                'downsample_factor': downsample_factor
            }
        },
        'processing_level': processing_level,
        'downsample_factor': downsample_factor,
        'label_colors': label_colors,
        'label_names': label_names,
        'legend_labels': legend_labels,  # Labels to show in control panel legend
        'has_inner_mask': has_inner_mask,
        'skeleton_intersections': skeleton_metadata
    }
    
    metadata_path = zarr_path.parent / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Zarr writing complete")
