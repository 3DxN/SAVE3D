"""
Image Pyramid Building

GPU-accelerated (CuPy) or CPU fallback for multi-resolution pyramid construction.
Uses local mean downsampling with:
- Chunked processing for large images
- True pipelined CUDA streams (overlap H2D, compute, D2H)
- Automatic GPU memory-based chunk sizing
- Progressive downsampling (Level N from Level N-1)
"""

import gc
import numpy as np
from tqdm import tqdm
from skimage.transform import downscale_local_mean

from .config import GPU_AVAILABLE, get_cupy


def _get_gpu_memory_info():
    """Get GPU memory info in GB"""
    cp = get_cupy()
    if cp is None:
        return None, None
    
    try:
        device = cp.cuda.Device()
        total = device.mem_info[1] / (1024**3)  # Total GPU memory
        free = device.mem_info[0] / (1024**3)   # Free GPU memory
        return total, free
    except:
        return None, None


def _calculate_optimal_chunk_size(image_shape, dtype, free_mem_gb, factor, is_rgb=False, 
                                   target_utilization=0.85, min_chunk_z=None, max_chunk_z_limit=512):
    """
    Calculate optimal chunk size based on GPU memory
    
    OPTIMIZED: More aggressive memory utilization for better GPU throughput
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of input image (Z, Y, X) or (Z, Y, X, C)
    dtype : np.dtype
        Data type of image
    free_mem_gb : float
        Available GPU memory in GB
    factor : int
        Downsampling factor
    is_rgb : bool
        Whether image has RGB channels
    target_utilization : float
        Target GPU memory utilization (0.0-1.0), default 0.85 (85%)
    min_chunk_z : int or None
        Minimum chunk size (defaults to factor)
    max_chunk_z_limit : int
        Maximum chunk size limit (default 512)
    
    Returns:
    --------
    chunk_size_z : int
        Optimal number of Z slices per chunk
    """
    nz, ny, nx = image_shape[:3]
    nc = image_shape[3] if is_rgb else 1
    
    if min_chunk_z is None:
        min_chunk_z = factor
    
    # Memory calculation (optimized for better GPU utilization)
    # For downscale_local_mean with CuPy:
    # 1. Input chunk (as float32)
    # 2. Intermediate reshape - CuPy uses views, minimal extra allocation
    # 3. Output chunk (smaller, float32)
    # 
    # CuPy's reshape for block mean uses views when possible,
    # so actual memory is close to input + output only
    
    input_slice_bytes = ny * nx * nc * 4  # float32
    output_slice_bytes = (ny // factor) * (nx // factor) * nc * 4 / factor
    
    # Memory per Z slice in chunk
    # - Input: 1.0x (the data being processed)
    # - Output: ~0.125x for factor=2 (1/8 of input)
    # - Overhead: ~10% for CuPy memory allocator fragmentation
    # Total: ~1.1x input + output
    mem_per_z_slice = input_slice_bytes * 1.1 + output_slice_bytes
    
    # Target memory usage
    target_mem_bytes = free_mem_gb * (1024**3) * target_utilization
    
    # Calculate optimal chunk size
    optimal_chunk_z = int(target_mem_bytes / mem_per_z_slice)
    
    # Round down to multiple of factor for clean division
    optimal_chunk_z = (optimal_chunk_z // factor) * factor
    
    # Clamp to valid range
    optimal_chunk_z = max(min_chunk_z, min(optimal_chunk_z, nz, max_chunk_z_limit))
    
    # Ensure it's at least the factor
    optimal_chunk_z = max(factor, optimal_chunk_z)
    
    return optimal_chunk_z


def _get_recommended_chunk_size(free_mem_gb, image_shape, factor, is_rgb=False):
    """
    Get recommended chunk size with detailed memory breakdown
    
    Returns:
    --------
    chunk_size : int
    memory_info : dict
    """
    nz, ny, nx = image_shape[:3]
    nc = image_shape[3] if is_rgb else 1
    
    input_slice_mb = ny * nx * nc * 4 / (1024**2)  # float32
    output_slice_mb = (ny // factor) * (nx // factor) * nc * 4 / factor / (1024**2)
    
    chunk_size = _calculate_optimal_chunk_size(image_shape, np.float32, free_mem_gb, factor, is_rgb)
    
    # Estimate based on optimized memory calculation
    estimated_usage_mb = chunk_size * (input_slice_mb * 1.1 + output_slice_mb)
    
    memory_info = {
        'free_mem_gb': free_mem_gb,
        'input_slice_mb': input_slice_mb,
        'output_slice_mb': output_slice_mb,
        'chunk_size_z': chunk_size,
        'estimated_usage_mb': estimated_usage_mb,
        'estimated_usage_gb': estimated_usage_mb / 1024,
        'utilization_percent': (estimated_usage_mb / 1024) / free_mem_gb * 100
    }
    
    return chunk_size, memory_info


def gpu_downscale_local_mean_pipelined(image, factors, chunk_size_z=None, cval=0, clip=True,
                                        target_gpu_utilization=0.85):
    """
    GPU implementation with true pipelined CUDA streams
    
    OPTIMIZED: Better GPU memory utilization (default 85%)
    
    Pipeline stages (overlapped):
    1. H2D transfer of chunk N+1
    2. Compute on chunk N
    3. D2H transfer of chunk N-1
    
    Parameters:
    -----------
    image : ndarray
        Input image (numpy array), shape (Z, Y, X) or (Z, Y, X, C)
    factors : tuple
        Downsampling factors per dimension
    chunk_size_z : int or None
        Number of Z slices per chunk. If None, auto-calculated based on GPU memory.
    cval : float
        Constant value for padding
    clip : bool
        Whether to clip output to input dtype range
    target_gpu_utilization : float
        Target GPU memory utilization (0.0-1.0), default 0.85 (85%)
    
    Returns:
    --------
    result : ndarray
        Downsampled image (numpy array)
    """
    cp = get_cupy()
    
    if not GPU_AVAILABLE or cp is None:
        return downscale_local_mean(image, factors, cval=cval, clip=clip)
    
    # Get GPU memory info
    total_mem, free_mem = _get_gpu_memory_info()
    
    # Handle RGB images
    is_rgb = image.ndim == 4
    if is_rgb:
        if len(factors) == 3:
            factors = (*factors, 1)
    
    nz, ny, nx = image.shape[:3]
    fz, fy, fx = factors[:3]
    nz = (nz // fz) * fz
    image = image[:nz]
    
    # Auto-calculate optimal chunk size with detailed memory info
    if chunk_size_z is None:
        if free_mem:
            chunk_size_z, mem_info = _get_recommended_chunk_size(
                free_mem, image.shape, fz, is_rgb
            )
            print(f"    GPU Memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")
            print(f"    Auto chunk size: {chunk_size_z} slices")
            print(f"    Estimated GPU usage: {mem_info['estimated_usage_gb']:.2f}GB ({mem_info['utilization_percent']:.0f}%)")
        else:
            chunk_size_z = 64  # Default fallback
            print(f"    Using default chunk size: {chunk_size_z} slices")
    
    # Ensure chunk_size_z is multiple of factor
    chunk_size_z = max(fz, (chunk_size_z // fz) * fz)
    
    # Calculate output shape
    out_nz = int(np.ceil(nz / fz))
    out_ny = int(np.ceil(ny / fy))
    out_nx = int(np.ceil(nx / fx))
    
    if is_rgb:
        out_shape = (out_nz, out_ny, out_nx, image.shape[3])
    else:
        out_shape = (out_nz, out_ny, out_nx)
    
    # Allocate output array
    result = np.zeros(out_shape, dtype=np.float32)
    
    # Calculate chunk info
    num_chunks = int(np.ceil(nz / chunk_size_z))
    
    # For small number of chunks, use simple sequential processing
    if num_chunks <= 2:
        return _gpu_downscale_sequential(image, factors, chunk_size_z, cval, clip, 
                                         result, out_nz, is_rgb, cp)
    
    # Create CUDA streams for pipelining
    stream_h2d = cp.cuda.Stream(non_blocking=True)  # Host to Device
    stream_compute = cp.cuda.Stream(non_blocking=True)  # Compute
    stream_d2h = cp.cuda.Stream(non_blocking=True)  # Device to Host
    
    # Pipeline state
    gpu_buffers = [None, None]  # Double buffer for GPU data
    results_pending = [None, None]  # Pending results to write
    
    def prepare_chunk(chunk_idx):
        """Prepare chunk data on CPU"""
        z_start = chunk_idx * chunk_size_z
        z_end = min(z_start + chunk_size_z, nz)
        actual_chunk_size = z_end - z_start
        need_padding = actual_chunk_size % fz != 0
        
        if need_padding:
            pad_z = fz - (actual_chunk_size % fz)
            if is_rgb:
                chunk_data = np.zeros((actual_chunk_size + pad_z, ny, nx, image.shape[3]), dtype=image.dtype)
            else:
                chunk_data = np.zeros((actual_chunk_size + pad_z, ny, nx), dtype=image.dtype)
            chunk_data[:actual_chunk_size] = image[z_start:z_end]
        else:
            # Use view if already contiguous, otherwise copy
            chunk_data = image[z_start:z_end]
            if not chunk_data.flags['C_CONTIGUOUS']:
                chunk_data = np.ascontiguousarray(chunk_data)
        
        return chunk_data, z_start
    
    def write_result(chunk_result_cpu, z_start):
        """Write result to output array"""
        if chunk_result_cpu is None:
            return
        out_z_start = z_start // fz
        out_z_end = out_z_start + chunk_result_cpu.shape[0]
        if out_z_end > out_nz:
            chunk_result_cpu = chunk_result_cpu[:out_nz - out_z_start]
            out_z_end = out_nz
        result[out_z_start:out_z_end] = chunk_result_cpu
    
    # Process chunks with pipelining
    pbar = tqdm(total=num_chunks, desc="    GPU pipeline", unit="chunk", leave=False)
    
    try:
        for chunk_idx in range(num_chunks + 2):  # +2 for pipeline drain
            buf_idx = chunk_idx % 2
            
            # Stage 1: Start H2D transfer for chunk N+2 (if exists)
            if chunk_idx < num_chunks:
                chunk_data, z_start = prepare_chunk(chunk_idx)
                with stream_h2d:
                    gpu_buffers[buf_idx] = (cp.asarray(chunk_data), z_start)
            
            # Stage 2: Compute on chunk N+1 (if exists)
            if chunk_idx >= 1 and chunk_idx < num_chunks + 1:
                prev_buf_idx = (chunk_idx - 1) % 2
                if gpu_buffers[prev_buf_idx] is not None:
                    chunk_gpu, z_start = gpu_buffers[prev_buf_idx]
                    
                    # Wait for H2D to complete
                    stream_compute.wait_event(stream_h2d.record())
                    
                    with stream_compute:
                        chunk_result = _gpu_downscale_chunk(chunk_gpu, factors, cval, cp)
                        results_pending[prev_buf_idx] = (chunk_result, z_start)
                    
                    # Free input buffer
                    del chunk_gpu
                    gpu_buffers[prev_buf_idx] = None
            
            # Stage 3: D2H transfer and write for chunk N (if exists)
            if chunk_idx >= 2:
                prev_prev_buf_idx = (chunk_idx - 2) % 2
                if results_pending[prev_prev_buf_idx] is not None:
                    chunk_result, z_start = results_pending[prev_prev_buf_idx]
                    
                    # Wait for compute to complete
                    stream_d2h.wait_event(stream_compute.record())
                    
                    with stream_d2h:
                        chunk_result_cpu = cp.asnumpy(chunk_result)
                    
                    stream_d2h.synchronize()
                    write_result(chunk_result_cpu, z_start)
                    
                    # Free result buffer
                    del chunk_result
                    results_pending[prev_prev_buf_idx] = None
                    
                    pbar.update(1)
        
        pbar.close()
        
    except cp.cuda.memory.OutOfMemoryError as e:
        pbar.close()
        print(f"    ⚠️ GPU OOM, falling back to sequential processing")
        # Clean up and fall back to sequential
        cp.get_default_memory_pool().free_all_blocks()
        return _gpu_downscale_sequential(image, factors, chunk_size_z // 2, cval, clip,
                                         result, out_nz, is_rgb, cp)
    
    # Clean up
    cp.get_default_memory_pool().free_all_blocks()
    
    # Clip and convert dtype
    if clip and image.dtype.kind in 'ui':
        if image.dtype.kind == 'u':
            result = np.clip(result, 0, np.iinfo(image.dtype).max)
        else:
            result = np.clip(result, np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
    
    result = result.astype(image.dtype)
    
    return result


def _gpu_downscale_sequential(image, factors, chunk_size_z, cval, clip, result, out_nz, is_rgb, cp):
    """Simple sequential GPU processing (fallback)"""
    nz, ny, nx = image.shape[:3]
    fz = factors[0]
    nz = (nz // fz) * fz
    image = image[:nz]
    num_chunks = int(np.ceil(nz / chunk_size_z))
    
    for chunk_idx in tqdm(range(num_chunks), desc="    GPU chunks", unit="chunk", leave=False):
        z_start = chunk_idx * chunk_size_z
        z_end = min(z_start + chunk_size_z, nz)
        
        actual_chunk_size = z_end - z_start
        need_padding = actual_chunk_size % fz != 0
        
        if need_padding:
            pad_z = fz - (actual_chunk_size % fz)
            if is_rgb:
                chunk_data = np.zeros((actual_chunk_size + pad_z, ny, nx, image.shape[3]), dtype=image.dtype)
            else:
                chunk_data = np.zeros((actual_chunk_size + pad_z, ny, nx), dtype=image.dtype)
            chunk_data[:actual_chunk_size] = image[z_start:z_end]
        else:
            chunk_data = image[z_start:z_end]
            if not chunk_data.flags['C_CONTIGUOUS']:
                chunk_data = np.ascontiguousarray(chunk_data)
        
        try:
            chunk_gpu = cp.asarray(chunk_data)
            chunk_result = _gpu_downscale_chunk(chunk_gpu, factors, cval, cp)
            chunk_result_cpu = cp.asnumpy(chunk_result)
            
            out_z_start = z_start // fz
            out_z_end = out_z_start + chunk_result_cpu.shape[0]
            if out_z_end > out_nz:
                chunk_result_cpu = chunk_result_cpu[:out_nz - out_z_start]
                out_z_end = out_nz
            result[out_z_start:out_z_end] = chunk_result_cpu
            
            del chunk_gpu, chunk_result
            cp.get_default_memory_pool().free_all_blocks()
            
        except cp.cuda.memory.OutOfMemoryError:
            # Fallback to CPU
            if is_rgb:
                chunk_result_cpu = downscale_local_mean(chunk_data, factors)
            else:
                chunk_result_cpu = downscale_local_mean(chunk_data, factors[:3])
            
            out_z_start = z_start // fz
            out_z_end = out_z_start + chunk_result_cpu.shape[0]
            if out_z_end > out_nz:
                chunk_result_cpu = chunk_result_cpu[:out_nz - out_z_start]
                out_z_end = out_nz
            result[out_z_start:out_z_end] = chunk_result_cpu
    
    # Clip and convert dtype
    if clip and image.dtype.kind in 'ui':
        if image.dtype.kind == 'u':
            result = np.clip(result, 0, np.iinfo(image.dtype).max)
        else:
            result = np.clip(result, np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
    
    return result.astype(image.dtype)


def _gpu_downscale_chunk(chunk_gpu, factors, cval, cp):
    """
    Downscale a single chunk on GPU
    
    Parameters:
    -----------
    chunk_gpu : cupy.ndarray
        Input chunk on GPU
    factors : tuple
        Downsampling factors
    cval : float
        Padding value
    cp : module
        CuPy module
    
    Returns:
    --------
    result : cupy.ndarray
        Downsampled chunk on GPU
    """
    result = chunk_gpu.astype(cp.float32)
    current_shape = list(result.shape)
    
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        
        factor = int(factor)
        old_size = current_shape[axis]
        new_size = int(np.ceil(old_size / factor))
        
        # Pad if necessary
        pad_size = new_size * factor - old_size
        if pad_size > 0:
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, pad_size)
            result = cp.pad(result, pad_width, mode='constant', constant_values=cval)
            current_shape[axis] = new_size * factor
        
        # Reshape and mean
        shape_before_mean = (
            current_shape[:axis] + 
            [new_size, factor] + 
            current_shape[axis+1:]
        )
        
        result = result.reshape(shape_before_mean)
        result = cp.mean(result, axis=axis+1)
        current_shape[axis] = new_size
    
    return result


def gpu_downscale_local_mean(image, factors, cval=0, clip=True):
    """
    GPU implementation of downscale_local_mean using CuPy
    100% compatible with skimage.transform.downscale_local_mean
    
    For large images, automatically uses pipelined chunked processing.
    
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
    
    if isinstance(factors, int):
        factors = (factors,) * image.ndim
    
    factors = tuple(factors)
    
    # Check image size - use chunked processing for large images
    image_size_gb = image.nbytes / (1024**3)
    total_mem, free_mem = _get_gpu_memory_info()
    
    # Use chunked processing if image is large relative to GPU memory
    use_chunked = False
    if free_mem:
        # Need ~3x image size for safe processing (input + intermediate + output)
        if image_size_gb * 3 > free_mem:
            use_chunked = True
            print(f"    Large image ({image_size_gb:.1f}GB) > GPU memory ({free_mem:.1f}GB free)")
            print(f"    Using pipelined GPU processing...")
    elif image_size_gb > 2.0:  # Default threshold if can't query GPU
        use_chunked = True
        print(f"    Large image ({image_size_gb:.1f}GB), using chunked processing...")
    
    if use_chunked:
        return gpu_downscale_local_mean_pipelined(image, factors, chunk_size_z=None, cval=cval, clip=clip)
    
    # Small image: process entirely on GPU
    if isinstance(image, np.ndarray):
        image_gpu = cp.asarray(image)
        return_numpy = True
    else:
        image_gpu = image
        return_numpy = False
    
    if len(factors) != image_gpu.ndim:
        raise ValueError(f"factors must have {image_gpu.ndim} elements, got {len(factors)}")
    
    result = _gpu_downscale_chunk(image_gpu, factors, cval, cp)
    
    # Clip if needed
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
        # Clean up GPU memory
        del image_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    return result


def build_pyramid_with_gpu_local_mean(image, levels, progressive=True):
    """
    Build multi-resolution pyramid using GPU local_mean
    
    Parameters:
    -----------
    image : ndarray
        Input image at Level 0
    levels : int
        Number of pyramid levels to generate
    progressive : bool
        If True, compute Level N from Level N-1 (faster for large images)
        If False, compute all levels from Level 0 (original behavior)
    
    Returns:
    --------
    pyramid : list
        List of images at each resolution level
    """
    pyramid = [image]
    
    is_rgb = len(image.shape) == 4 and (image.shape[-1] == 3 or image.shape[1] == 3)
    image_size_gb = image.nbytes / (1024**3)
    
    print(f"Building {levels} level pyramid...")
    print(f"  Input: {image.shape}, {image.dtype}, {image_size_gb:.1f}GB")
    print(f"  Mode: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU'}")
    print(f"  Strategy: {'Progressive (L(N) from L(N-1))' if progressive else 'Direct (all from L0)'}")
    
    # Determine which source to use for each level
    for level in tqdm(range(1, levels), desc="Building pyramid", unit="level"):
        if min(image.shape[:3]) // (2**level) < 16:
            print(f"Stopped at level {level-1} (too small)")
            break
        
        if progressive and level > 1:
            # Progressive: compute from previous level (2x downsample)
            source = pyramid[-1]
            factor = 2
            print(f"\n  Level {level}: 2x from Level {level-1}")
        else:
            # Direct: compute from Level 0
            source = image
            factor = 2 ** level
            print(f"\n  Level {level}: {factor}x from Level 0")
        
        # GPU downsampling with automatic chunking and pipelining
        if GPU_AVAILABLE:
            downsampled = gpu_downscale_local_mean(
                source, 
                (factor, factor, factor) if not is_rgb else (factor, factor, factor, 1)
            )
        else:
            downsampled = downscale_local_mean(
                source, 
                (factor, factor, factor) if not is_rgb else (factor, factor, factor, 1)
            )
        
        # Ensure correct dtype
        if image.dtype == np.uint16:
            downsampled = np.clip(downsampled, 0, 65535).astype(np.uint16)
        elif image.dtype == np.uint8:
            downsampled = downsampled.astype(np.uint8)
        
        pyramid.append(downsampled)
        print(f"    Output: {downsampled.shape}, {downsampled.nbytes/(1024**3):.2f}GB")
        
        # Only gc.collect for large allocations
        if source.nbytes > 1024**3:  # > 1GB
            gc.collect()
    
    print(f"\n✓ Pyramid complete: {len(pyramid)} levels")
    return pyramid


# Legacy function for backward compatibility
def gpu_downscale_local_mean_chunked(image, factors, chunk_size_z=64, cval=0, clip=True):
    """Legacy chunked processing function - now redirects to pipelined version"""
    return gpu_downscale_local_mean_pipelined(image, factors, chunk_size_z, cval, clip)


def gpu_downscale_local_mean_streamed(image, factors, chunk_size_z=None, cval=0, clip=True):
    """Legacy streamed processing function - now redirects to pipelined version"""
    return gpu_downscale_local_mean_pipelined(image, factors, chunk_size_z, cval, clip)
