"""
Image Pyramid Building

GPU-accelerated (CuPy) or CPU fallback for multi-resolution pyramid construction.
Uses local mean downsampling for high-quality results.
"""

import gc
import numpy as np
from tqdm import tqdm
from skimage.transform import downscale_local_mean

from .config import GPU_AVAILABLE, get_cupy


def gpu_downscale_local_mean(image, factors, cval=0, clip=True):
    """
    GPU implementation of downscale_local_mean using CuPy
    100% compatible with skimage.transform.downscale_local_mean
    
    Parameters:
    -----------
    image : ndarray
        Input image (numpy or cupy array)
    factors : int or tuple
        Downsampling factors per dimension
    cval : float
        Constant value for padding
    clip : bool
        Whether to clip output to input dtype range
    
    Returns:
    --------
    result : ndarray
        Downsampled image (same type as input)
    """
    cp = get_cupy()
    
    if not GPU_AVAILABLE or cp is None:
        return downscale_local_mean(image, factors, cval=cval, clip=clip)
    
    if isinstance(image, np.ndarray):
        image_gpu = cp.asarray(image)
        return_numpy = True
    else:
        image_gpu = image
        return_numpy = False
    
    if isinstance(factors, int):
        factors = (factors,) * image_gpu.ndim
    
    factors = tuple(factors)
    
    if len(factors) != image_gpu.ndim:
        raise ValueError(f"factors must have {image_gpu.ndim} elements, got {len(factors)}")
    
    output_shape = tuple(
        int(np.ceil(s / f)) for s, f in zip(image_gpu.shape, factors)
    )
    
    result = image_gpu
    current_shape = list(image_gpu.shape)
    
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        
        factor = int(factor)
        old_size = current_shape[axis]
        new_size = int(np.ceil(old_size / factor))
        
        pad_size = new_size * factor - old_size
        if pad_size > 0:
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, pad_size)
            result = cp.pad(result, pad_width, mode='constant', constant_values=cval)
            current_shape[axis] = new_size * factor
        
        shape_before_mean = (
            current_shape[:axis] + 
            [new_size, factor] + 
            current_shape[axis+1:]
        )
        
        result = result.reshape(shape_before_mean)
        result = cp.mean(result, axis=axis+1)
        
        current_shape[axis] = new_size
    
    if clip and image_gpu.dtype.kind in 'ui':
        if image_gpu.dtype.kind == 'u':
            result = cp.clip(result, 0, np.iinfo(image_gpu.dtype).max)
        else:
            result = cp.clip(result, np.iinfo(image_gpu.dtype).min, 
                           np.iinfo(image_gpu.dtype).max)
    
    if result.dtype != image_gpu.dtype:
        result = result.astype(image_gpu.dtype)
    
    if return_numpy:
        result = cp.asnumpy(result)
    
    return result


def build_pyramid_with_gpu_local_mean(image, levels):
    """
    Build multi-resolution pyramid using GPU local_mean
    
    Parameters:
    -----------
    image : ndarray
        Input image at Level 0
    levels : int
        Number of pyramid levels to generate
    
    Returns:
    --------
    pyramid : list
        List of images at each resolution level
    """
    pyramid = [image]
    
    is_rgb = len(image.shape) == 4 and (image.shape[-1] == 3 or image.shape[1] == 3)
    print(f"Building {levels} level pyramid...")
    
    for level in tqdm(range(1, levels), desc="Building pyramid", unit="level"):
        if min(image.shape[:3]) // (2**level) < 16:
            print(f"Stopped at level {level-1} (too small)")
            break
        
        factor = 2 ** level
        
        # GPU downsampling
        if GPU_AVAILABLE:
            downsampled = gpu_downscale_local_mean(
                image, 
                (factor, factor, factor) if not is_rgb else (factor, factor, factor, 1)
            )
        else:
            downsampled = downscale_local_mean(
                image, 
                (factor, factor, factor) if not is_rgb else (factor, factor, factor, 1)
            )
        
        if image.dtype == np.uint16:
            downsampled = np.clip(downsampled, 0, 65535).astype(np.uint16)
        elif image.dtype == np.uint8:
            downsampled = downsampled.astype(np.uint8)
        
        pyramid.append(downsampled)
        gc.collect()
    
    print(f"✓ Pyramid complete: {len(pyramid)} levels")
    return pyramid
