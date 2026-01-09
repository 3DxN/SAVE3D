"""
Mesh Building Module

"""

import numpy as np
from scipy.ndimage import gaussian_filter
import pyvista as pv


class MeshBuilder:
    """
    Mesh creation utilities
    
    """
    
    def __init__(self, voxel_size_L2):
        """
        Initialize with voxel size
        
        Args:
            voxel_size_L2: tuple (vz, vy, vx) in micrometers
        """
        self.voxel_size_L2 = voxel_size_L2
    
    def _create_surface(self, mask, xy_bbox, z_min, pad=2, smooth=False):
        """
        Create surface from mask
        
        Args:
            mask: 3D binary mask
            xy_bbox: dict with y_min, y_max, x_min, x_max
            z_min: minimum z index
            pad: padding size (default 2)
            smooth: whether to apply gaussian blur + mesh smoothing
        
        Returns:
            pv.PolyData or None
        """
        if mask.sum() == 0:
            return None
        
        try:
            nz, ny, nx = mask.shape
            padded_shape = (nz + 2*pad, ny + 2*pad, nx + 2*pad)
            padded_mask = np.zeros(padded_shape, dtype=np.uint8)
            padded_mask[pad:pad+nz, pad:pad+ny, pad:pad+nx] = mask
            
            # Optional blur
            if smooth:
                mask_float = gaussian_filter(padded_mask.astype(float), sigma=0.8)
            else:
                mask_float = padded_mask.astype(float)
            
            # Transpose for PyVista
            data_transposed = np.transpose(mask_float, (2, 1, 0))
            
            # Origin
            origin_x = (xy_bbox['x_min'] - pad) * self.voxel_size_L2[2]
            origin_y = (xy_bbox['y_min'] - pad) * self.voxel_size_L2[1]
            origin_z = (z_min - pad) * self.voxel_size_L2[0]
            
            # Create grid
            grid = pv.ImageData(
                dimensions=data_transposed.shape,
                spacing=self.voxel_size_L2,
                origin=(origin_x, origin_y, origin_z)
            )
            
            grid.point_data['values'] = data_transposed.ravel(order='F')
            mesh = grid.contour(isosurfaces=[0.5])
            
            # Optional smooth
            if smooth and mesh.n_points > 0:
                mesh = mesh.smooth(n_iter=10, relaxation_factor=0.1)
            
            return mesh if mesh.n_points > 0 else None
            
        except Exception as e:
            print(f"    ✗ Surface failed: {e}")
            return None