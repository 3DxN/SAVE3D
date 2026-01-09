"""
Image Transform Functions

Upsampling and other spatial transformations for masks.
"""

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def upsample_mask_nearest(mask_L2, target_shape_L0, factor):
    """
    Nearest neighbor upsampling from Level 2 to Level 0 with exact size matching
    
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
    
    # Calculate expected output shape from upsampling
    expected_shape = tuple(s * factor for s in mask_L2.shape)
    
    # Check if sizes match
    size_mismatch = False
    for i, (exp, tgt) in enumerate(zip(expected_shape, target_shape_L0)):
        if exp != tgt:
            size_mismatch = True
            diff = exp - tgt
            axis_name = ['Z', 'Y', 'X'][i]
            print(f"  {axis_name}: expected {exp}, target {tgt}, diff {diff:+d}")
    
    if size_mismatch:
        print(f"  Will trim to exact target size after upsampling")
    
    # Perform upsampling chunk by chunk
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
        
        # Pad if needed (replicate last slice/row/col)
        if mask_L0.shape[0] < target_shape_L0[0]:
            # Pad Z: replicate last slice
            for z in range(mask_L0.shape[0], target_shape_L0[0]):
                result[z, :ny_copy, :nx_copy] = mask_L0[-1, :ny_copy, :nx_copy]
            print(f"    Z: padded {target_shape_L0[0] - mask_L0.shape[0]} slices")
        
        if mask_L0.shape[1] < target_shape_L0[1]:
            # Pad Y: replicate last row
            for y in range(mask_L0.shape[1], target_shape_L0[1]):
                result[:, y, :nx_copy] = result[:, mask_L0.shape[1]-1, :nx_copy]
            print(f"    Y: padded {target_shape_L0[1] - mask_L0.shape[1]} rows")
        
        if mask_L0.shape[2] < target_shape_L0[2]:
            # Pad X: replicate last column
            for x in range(mask_L0.shape[2], target_shape_L0[2]):
                result[:, :, x] = result[:, :, mask_L0.shape[2]-1]
            print(f"    X: padded {target_shape_L0[2] - mask_L0.shape[2]} cols")
        
        mask_L0 = result
        print(f"  ✓ Adjusted to exact target size")
    
    print(f"✓ Final upsampled shape: {mask_L0.shape}")
    return mask_L0
