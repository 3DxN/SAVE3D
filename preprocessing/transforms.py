"""
Image Transform Functions

Upsampling and other spatial transformations for masks.
Supports GPU acceleration via CuPy when available.
"""

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from .config import GPU_AVAILABLE, get_cupy


def upsample_mask_nearest(mask_L2, target_shape_L0, factor):
    """
    Nearest neighbor upsampling from Level 2 to Level 0 with exact size matching
    
    Uses GPU acceleration when available for faster processing.
    
    Parameters:
    -----------
    mask_L2 : np.array
        Input mask at Level 2 (processing level)
    target_shape_L0 : tuple
        Target shape (nz, ny, nx) at Level 0
    factor : int
        Upsampling factor (e.g., 4 for L2→L0)
    
    Returns:
    --------
    mask_L0 : np.array
        Upsampled mask with exact target_shape_L0
    """
    print(f"Upsampling L2 {mask_L2.shape} → L0 {target_shape_L0}...")
    print(f"  Mode: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU'}")
    
    # Use fast repeat-based upsampling instead of zoom
    # This is much faster for nearest neighbor
    if GPU_AVAILABLE:
        return _upsample_mask_gpu(mask_L2, target_shape_L0, factor)
    else:
        return _upsample_mask_cpu_fast(mask_L2, target_shape_L0, factor)


def _upsample_mask_gpu(mask_L2, target_shape_L0, factor):
    """GPU-accelerated nearest neighbor upsampling using repeat"""
    cp = get_cupy()
    
    nz_L2, ny_L2, nx_L2 = mask_L2.shape
    nz_L0, ny_L0, nx_L0 = target_shape_L0
    
    # Process in chunks to avoid GPU memory issues
    chunk_size_z = 32  # Process 32 L2 slices at a time
    num_chunks = (nz_L2 + chunk_size_z - 1) // chunk_size_z
    
    # Allocate output
    result = np.zeros(target_shape_L0, dtype=mask_L2.dtype)
    
    for chunk_idx in tqdm(range(num_chunks), desc="GPU upsampling", unit="chunk"):
        z_start_L2 = chunk_idx * chunk_size_z
        z_end_L2 = min(z_start_L2 + chunk_size_z, nz_L2)
        
        # Get chunk
        chunk = mask_L2[z_start_L2:z_end_L2]
        
        # Transfer to GPU
        chunk_gpu = cp.asarray(chunk)
        
        # Fast repeat-based upsampling
        upsampled_gpu = cp.repeat(chunk_gpu, factor, axis=0)
        upsampled_gpu = cp.repeat(upsampled_gpu, factor, axis=1)
        upsampled_gpu = cp.repeat(upsampled_gpu, factor, axis=2)
        
        # Transfer back
        upsampled = cp.asnumpy(upsampled_gpu)
        
        # Calculate output range
        z_start_L0 = z_start_L2 * factor
        z_end_L0 = min(z_start_L0 + upsampled.shape[0], nz_L0)
        
        # Trim if needed
        actual_nz = z_end_L0 - z_start_L0
        actual_ny = min(upsampled.shape[1], ny_L0)
        actual_nx = min(upsampled.shape[2], nx_L0)
        
        result[z_start_L0:z_end_L0, :actual_ny, :actual_nx] = \
            upsampled[:actual_nz, :actual_ny, :actual_nx]
        
        # Free GPU memory
        del chunk_gpu, upsampled_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    print(f"✓ Final upsampled shape: {result.shape}")
    return result


def _upsample_mask_cpu_fast(mask_L2, target_shape_L0, factor):
    """Fast CPU nearest neighbor upsampling using repeat"""
    nz_L2, ny_L2, nx_L2 = mask_L2.shape
    nz_L0, ny_L0, nx_L0 = target_shape_L0
    
    # Process in chunks to reduce memory usage
    chunk_size_z = 32
    num_chunks = (nz_L2 + chunk_size_z - 1) // chunk_size_z
    
    result = np.zeros(target_shape_L0, dtype=mask_L2.dtype)
    
    for chunk_idx in tqdm(range(num_chunks), desc="CPU upsampling", unit="chunk"):
        z_start_L2 = chunk_idx * chunk_size_z
        z_end_L2 = min(z_start_L2 + chunk_size_z, nz_L2)
        
        chunk = mask_L2[z_start_L2:z_end_L2]
        
        # Fast repeat-based upsampling (much faster than zoom)
        upsampled = np.repeat(chunk, factor, axis=0)
        upsampled = np.repeat(upsampled, factor, axis=1)
        upsampled = np.repeat(upsampled, factor, axis=2)
        
        # Calculate output range
        z_start_L0 = z_start_L2 * factor
        z_end_L0 = min(z_start_L0 + upsampled.shape[0], nz_L0)
        
        # Trim if needed
        actual_nz = z_end_L0 - z_start_L0
        actual_ny = min(upsampled.shape[1], ny_L0)
        actual_nx = min(upsampled.shape[2], nx_L0)
        
        result[z_start_L0:z_end_L0, :actual_ny, :actual_nx] = \
            upsampled[:actual_nz, :actual_ny, :actual_nx]
    
    print(f"✓ Final upsampled shape: {result.shape}")
    return result


# Legacy function for backward compatibility
def upsample_mask_nearest_legacy(mask_L2, target_shape_L0, factor):
    """Original zoom-based upsampling (slower, kept for reference)"""
    print(f"Upsampling L2 {mask_L2.shape} → L0 {target_shape_L0}...")
    
    expected_shape = tuple(s * factor for s in mask_L2.shape)
    
    size_mismatch = False
    for i, (exp, tgt) in enumerate(zip(expected_shape, target_shape_L0)):
        if exp != tgt:
            size_mismatch = True
            diff = exp - tgt
            axis_name = ['Z', 'Y', 'X'][i]
            print(f"  {axis_name}: expected {exp}, target {tgt}, diff {diff:+d}")
    
    if size_mismatch:
        print(f"  Will trim to exact target size after upsampling")
    
    nz = mask_L2.shape[0]
    mask_L0 = np.zeros(expected_shape, dtype=mask_L2.dtype)
    
    chunk_size = 10
    num_chunks = (nz + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(num_chunks), desc="Upsampling", unit="chunk"):
        z_start = chunk_idx * chunk_size
        z_end = min(z_start + chunk_size, nz)
        
        chunk = mask_L2[z_start:z_end]
        chunk_up = zoom(chunk, factor, order=0)  # order=0 = nearest neighbor
        
        z_out_start = z_start * factor
        z_out_end = z_out_start + chunk_up.shape[0]
        
        # Be careful not to exceed expected_shape
        z_out_end = min(z_out_end, expected_shape[0])
        actual_chunk_size = z_out_end - z_out_start
        mask_L0[z_out_start:z_out_end] = chunk_up[:actual_chunk_size]
    
    # Trim or pad to exact target size if needed
    if mask_L0.shape != target_shape_L0:
        print(f"  Adjusting from {mask_L0.shape} to {target_shape_L0}...")
        
        result = np.zeros(target_shape_L0, dtype=mask_L0.dtype)
        
        # Copy overlapping region
        nz_copy = min(mask_L0.shape[0], target_shape_L0[0])
        ny_copy = min(mask_L0.shape[1], target_shape_L0[1])
        nx_copy = min(mask_L0.shape[2], target_shape_L0[2])
        
        result[:nz_copy, :ny_copy, :nx_copy] = mask_L0[:nz_copy, :ny_copy, :nx_copy]
        mask_L0 = result
    
    print(f"✓ Final upsampled shape: {mask_L0.shape}")
    return mask_L0
