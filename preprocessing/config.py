"""
Configuration and GPU Detection

Handles CuPy/GPU availability and provides fallback to CPU mode.
"""

import gc

# GPU detection
FORCE_CPU_MODE = False
GPU_AVAILABLE = False

try:
    if not FORCE_CPU_MODE:
        import cupy as cp
        GPU_AVAILABLE = True
        print("[OK] CuPy loaded - GPU acceleration enabled")
    else:
        cp = None
        print("[INFO] Forced CPU mode")
except ImportError:
    cp = None
    print("[INFO] GPU not available - CPU mode")


def get_cupy():
    """Get CuPy module if available, else None"""
    if GPU_AVAILABLE:
        import cupy as cp
        return cp
    return None


def cleanup_gpu():
    """Free GPU memory"""
    if GPU_AVAILABLE:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
