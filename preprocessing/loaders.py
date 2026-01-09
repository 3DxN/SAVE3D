"""
Data Loading Functions

Load volumes from various formats (TIFF, NPY) and label volumes.
"""

from pathlib import Path
import numpy as np
import tifffile


def load_volume(path):
    """
    Load volume from various formats
    
    Supported formats:
    - .npy: NumPy array
    - .tif/.tiff: TIFF image stack
    
    For 4D TIFF images, automatically handles channel ordering.
    """
    path = Path(path)
    s = path.suffix.lower()
    
    if s == ".npy":
        return np.load(str(path))
    
    if s in [".tif", ".tiff"]:
        arr = tifffile.imread(str(path))
        
        # Format handling for 4D arrays
        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                # Channel-last: (Z, Y, X, C) -> keep RGB
                return arr[..., :3]
            elif arr.shape[1] in (3, 4):
                # Channel-second: (Z, C, Y, X) -> transpose to (Z, Y, X, C)
                return np.transpose(arr, (0, 2, 3, 1))[..., :3]
        
        return arr
    
    raise ValueError(f"Unsupported format: {path}")


def load_labels_from_volume(label_path, label_dict=None):
    """
    Load labels from a single volume file
    
    Scans ALL unique values in the volume (except 0=background).
    Uses label_dict for custom names where specified, auto-generates for others.
    
    Parameters:
    -----------
    label_path : str or Path
        Path to label volume (values: 0=background, 1,2,3,...=labels)
    label_dict : dict or None
        {value: label_name} mapping for custom names.
        Values not in dict will be auto-named as "Label_N".
        If None or empty {}, all labels get auto-names.
    
    Returns:
    --------
    masks : dict
        {label_name: binary_mask}
    label_names : list
        Ordered list of label names
    user_named_labels : list
        Labels that have user-specified names (for legend display)
    """
    print(f"Loading label volume: {label_path}")
    
    label_vol = tifffile.imread(str(label_path))
    print(f"  Shape: {label_vol.shape}, dtype: {label_vol.dtype}")
    
    # Get ALL unique values except 0 (background)
    unique_values = np.unique(label_vol)
    unique_values = unique_values[unique_values > 0]
    print(f"  Found {len(unique_values)} unique labels (excl. background)")
    
    if label_dict is None:
        label_dict = {}
    
    masks = {}
    label_names = []
    user_named_labels = []  # Track which labels have user-specified names
    
    # Process ALL unique values
    for value in sorted(unique_values):
        value = int(value)
        
        # Use user-specified name or auto-generate
        if value in label_dict:
            name = label_dict[value]
            user_named_labels.append(name)
            source = "user"
        else:
            name = f"Label_{value}"
            source = "auto"
        
        mask = (label_vol == value).astype(np.uint8)
        voxel_count = np.sum(mask)
        
        if voxel_count > 0:
            masks[name] = mask
            label_names.append(name)
            print(f"  {name} (value={value}): {voxel_count:,} voxels [{source}]")
    
    return masks, label_names, user_named_labels
