"""
CC Path Propagation Module

"""

import numpy as np
from scipy import ndimage


class CCPathTracker:
    """
    Connected Component path tracking and propagation
    
    """
    
    def __init__(self, cc_metadata, outer_masks_L2, inner_masks_L2, label_names, voxel_size_L2):
        """
        Initialize with required data
        """
        self.cc_metadata = cc_metadata
        self.outer_masks_L2 = outer_masks_L2
        self.inner_masks_L2 = inner_masks_L2
        self.label_names = label_names
        self.voxel_size_L2 = voxel_size_L2
    
    # =========================================================================
    # CC PATH PROPAGATION
    # =========================================================================
       
    def _find_max_overlap_cc(self, label_id, z_from, cc_from, z_to, same_label_only=True):
        """
        Find CC with max overlap
        
        Args:
            label_id: source label ID
            z_from: source z index
            cc_from: source CC ID
            z_to: target z index
            same_label_only: if True, only search within same label
                            if False, search across all labels
        
        Returns:
            if same_label_only: cc_id (int) or None
            if not same_label_only: (label_id, cc_id) tuple or None
        """
        label_str = str(label_id)
        
        if label_str not in self.cc_metadata['labels']:
            return None
        
        label_data = self.cc_metadata['labels'][label_str]
        
        if str(z_from) not in label_data['layers']:
            return None
        
        source_layer = label_data['layers'][str(z_from)]
        
        if cc_from >= len(source_layer['outer']):
            return None
        
        source_cc = source_layer['outer'][cc_from]
        
        # Determine direction
        if z_to > z_from:
            overlaps = source_cc.get('overlaps_next', [])
        else:
            overlaps = source_cc.get('overlaps_prev', [])
        
        if not overlaps:
            return None
        
        # Filter if same_label_only
        if same_label_only:
            overlaps = [o for o in overlaps if o['label'] == label_id]
            if not overlaps:
                return None
        
        # Find max overlap
        max_overlap = max(overlaps, key=lambda x: x['area'])
        cc_id = max_overlap['cc_id'] - 1  # 1-indexed → 0-indexed
        
        if same_label_only:
            return cc_id
        else:
            return (max_overlap['label'], cc_id)


    def _propagate_cc_path(self, start_label_id, start_z, start_cc_id, cross_label=True):
        """
        Propagate CC path using overlap table
        
        Args:
            start_label_id: starting label ID
            start_z: starting z index
            start_cc_id: starting CC ID
            cross_label: if True, can jump across labels
                        if False, stay within same label
        
        Returns:
            path_3d: dict {z: [(label_id, cc_id), ...]}
        """
        mode_str = "CROSS LABEL" if cross_label else "SAME LABEL"
        print(f"\n[PROPAGATE {mode_str}] label={start_label_id}, z={start_z}, cc={start_cc_id}")
        
        path_3d = {}
        max_z = self.outer_masks_L2[self.label_names[0]].shape[0]
        
        path_3d[start_z] = [(start_label_id, start_cc_id)]
        
        # Propagate downward (z -> 0)
        print(f"  Propagating downward...")
        current_label = start_label_id
        current_cc = start_cc_id
        
        for z in range(start_z - 1, -1, -1):
            result = self._find_max_overlap_cc(
                current_label, z + 1, current_cc, z,
                same_label_only=not cross_label
            )
            
            if result is None:
                print(f"    Stop @ z={z+1} (no overlap)")
                break
            
            if cross_label:
                next_label, next_cc = result
                path_3d[z] = [(next_label, next_cc)]
                current_label, current_cc = next_label, next_cc
            else:
                path_3d[z] = [(start_label_id, result)]
                current_cc = result
        
        print(f"    Reached z={min(path_3d.keys())}")
        
        # Propagate upward (z -> max_z)
        print(f"  Propagating upward...")
        current_label = start_label_id
        current_cc = start_cc_id
        
        for z in range(start_z + 1, max_z):
            result = self._find_max_overlap_cc(
                current_label, z - 1, current_cc, z,
                same_label_only=not cross_label
            )
            
            if result is None:
                print(f"    Stop @ z={z-1} (no overlap)")
                break
            
            if cross_label:
                next_label, next_cc = result
                path_3d[z] = [(next_label, next_cc)]
                current_label, current_cc = next_label, next_cc
            else:
                path_3d[z] = [(start_label_id, result)]
                current_cc = result
        
        print(f"    Reached z={max(path_3d.keys())}")
        
        # Summary
        z_range = (min(path_3d.keys()), max(path_3d.keys()))
        print(f"  ✓ Path spans z=[{z_range[0]}:{z_range[1]}] ({z_range[1] - z_range[0] + 1} slices)")
        
        if cross_label:
            labels_in_path = set()
            for ccs in path_3d.values():
                for lid, _ in ccs:
                    labels_in_path.add(lid)
            label_names_in_path = [self.label_names[lid - 1] for lid in sorted(labels_in_path)]
            print(f"  ✓ Path contains {len(labels_in_path)} labels: {label_names_in_path}")
        
        return path_3d

    def _calculate_union_xy_bbox_from_path(self, path_3d):
        """
        Calculate union XY bbox from all CCs in path
        """
        print(f"\n[UNION BBOX]")
        
        all_y_coords = []
        all_x_coords = []
        
        for z, ccs in path_3d.items():
            for label_id, cc_id in ccs:
                label_str = str(label_id)
                
                if label_str not in self.cc_metadata['labels']:
                    continue
                
                label_data = self.cc_metadata['labels'][label_str]
                
                if str(z) not in label_data['layers']:
                    continue
                
                layer_data = label_data['layers'][str(z)]
                
                if cc_id >= len(layer_data['outer']):
                    continue
                
                cc_info = layer_data['outer'][cc_id]
                bbox_2d = cc_info['bbox']  # [y_min, y_max, x_min, x_max]
                
                all_y_coords.extend([bbox_2d[0], bbox_2d[1]])
                all_x_coords.extend([bbox_2d[2], bbox_2d[3]])
        
        if not all_y_coords:
            print("  ✗ No valid CCs in path")
            return None
        
        # Add margin
        margin = 20
        mask_shape = self.outer_masks_L2[self.label_names[0]].shape
        
        xy_bbox = {
            'y_min': max(0, min(all_y_coords) - margin),
            'y_max': min(mask_shape[1], max(all_y_coords) + margin),
            'x_min': max(0, min(all_x_coords) - margin),
            'x_max': min(mask_shape[2], max(all_x_coords) + margin)
        }
        
        print(f"  Union XY bbox:")
        print(f"    Y: [{xy_bbox['y_min']}:{xy_bbox['y_max']}] ({xy_bbox['y_max'] - xy_bbox['y_min']} pixels)")
        print(f"    X: [{xy_bbox['x_min']}:{xy_bbox['x_max']}] ({xy_bbox['x_max'] - xy_bbox['x_min']} pixels)")
        
        return xy_bbox

    def _extract_path_ccs_in_bbox(self, path_3d, xy_bbox):
        """
        Extract ONLY CCs on path within bbox (filter out junk)
        
        Returns:
            label_masks: {label_name: {'outer': 3D_mask, 'inner': 3D_mask}}
        """
        # Get Z range from path
        z_min = min(path_3d.keys())
        z_max = max(path_3d.keys()) + 1
        nz = z_max - z_min
        ny = xy_bbox['y_max'] - xy_bbox['y_min']
        nx = xy_bbox['x_max'] - xy_bbox['x_min']
        
        # Initialize masks for each label
        label_masks = {}
        for label_name in self.label_names:
            label_masks[label_name] = {
                'outer': np.zeros((nz, ny, nx), dtype=np.uint8),
                'inner': np.zeros((nz, ny, nx), dtype=np.uint8)
            }
        
        # Extract CCs layer by layer
        print(f"  Extracting path CCs (z=[{z_min}:{z_max}])...")
        
        for z in range(z_min, z_max):
            if z not in path_3d:
                continue
            
            for label_id, cc_id in path_3d[z]:
                label_name = self.label_names[label_id - 1]
                
                # Extract outer CC
                outer_slice = self.outer_masks_L2[label_name][z].compute()
                
                from scipy import ndimage
                labeled_outer, _ = ndimage.label(outer_slice > 0)
                
                # CC is 0-indexed in path, but 1-indexed in labeled array
                cc_mask = (labeled_outer == (cc_id + 1))
                
                # Crop to bbox
                cc_cropped = cc_mask[
                    xy_bbox['y_min']:xy_bbox['y_max'],
                    xy_bbox['x_min']:xy_bbox['x_max']
                ]
                
                # Add to mask
                z_rel = z - z_min
                label_masks[label_name]['outer'][z_rel] |= cc_cropped
                
                # Extract inner CC (within outer)
                if label_name in self.inner_masks_L2:
                    inner_slice = self.inner_masks_L2[label_name][z].compute()
                    inner_in_outer = (inner_slice > 0) & cc_mask
                    
                    inner_cropped = inner_in_outer[
                        xy_bbox['y_min']:xy_bbox['y_max'],
                        xy_bbox['x_min']:xy_bbox['x_max']
                    ]
                    
                    label_masks[label_name]['inner'][z_rel] |= inner_cropped
        
        # Report
        for label_name, masks in label_masks.items():
            outer_voxels = np.sum(masks['outer'])
            inner_voxels = np.sum(masks['inner'])
            if outer_voxels > 0:
                print(f"    {label_name}: outer={outer_voxels:,}, inner={inner_voxels:,}")
        
        return label_masks

    def _crossed_instance_boundary(self, from_z, to_z, label_id, cc_path_3d):
        """
        Check if we crossed an instance boundary (same label, but disconnected)
        
        Parameters:
        -----------
        from_z : int
            Previous Z
        to_z : int
            Current Z
        label_id : int
            Current label ID
        cc_path_3d : dict
            Current CC path (passed from caller)
        
        Returns:
        --------
        crossed : bool
            True if crossed instance boundary
        """
        if cc_path_3d is None:
            return False
        
        # Determine direction
        if to_z > from_z:
            z_range = range(from_z + 1, to_z + 1)
        else:
            z_range = range(from_z - 1, to_z - 1, -1)
        
        # Check each layer in between
        for z in z_range:
            if z not in cc_path_3d:
                return True
            
            ccs = cc_path_3d[z]
            if len(ccs) == 0:
                return True
            
            first_label_id, _ = ccs[0]
            
            if first_label_id != label_id:
                print(f"    Found boundary @ z={z}: {self.label_names[first_label_id-1]} (target: {self.label_names[label_id-1]})")
                return True
        
        return False

    def _get_best_overlap(self, z, label_id, cc_id, direction='next'):
        """ 
        Get best overlap for selection propagation       
        direction: 'next' = overlaps_next, 'prev' = overlaps_prev
        """
        label_str = str(label_id)
        
        if label_str not in self.cc_metadata['labels']:
            return None
        
        label_data = self.cc_metadata['labels'][label_str]
        
        if str(z) not in label_data['layers']:
            return None
        
        layer_data = label_data['layers'][str(z)]
        
        if cc_id >= len(layer_data['outer']):
            return None
        
        cc_info = layer_data['outer'][cc_id]
        
        overlap_key = 'overlaps_next' if direction == 'next' else 'overlaps_prev'
        
        best_cc = None
        best_area = 0
        
        for overlap in cc_info.get(overlap_key, []):
            if overlap['label'] == label_id and overlap['area'] > best_area:
                best_area = overlap['area']
                best_cc = overlap['cc_id'] - 1  # 1-indexed → 0-indexed
        
        if best_cc is not None and best_cc >= 0:
            return (label_id, best_cc)
        
        return None
                   
    # =========================================================================
    # SHARED FUNCTIONS (used by both modes)
    # =========================================================================
    
    def _find_clicked_cc(self, label_id, z_L2, y_L2, x_L2):
        """ Find CC at clicked position@L2 (data query) """
        print(f"\n    [CC DETECTION]")
        print(f"    Input: label_id={label_id}, coords @ L2: ({z_L2}, {y_L2}, {x_L2})")
        
        label_str = str(label_id)
        
        if label_str not in self.cc_metadata['labels']:
            print(f"    ✗ Label {label_id} not in metadata")
            return None
        
        label_data = self.cc_metadata['labels'][label_str]
        label_name = label_data['name']
        
        if str(z_L2) not in label_data['layers']:
            print(f"    ✗ Layer z={z_L2} not in metadata")
            return None
        
        if label_name not in self.outer_masks_L2:
            print(f"    ✗ No outer mask for {label_name}")
            return None
        
        outer_mask = self.outer_masks_L2[label_name]
        
        if (z_L2 < 0 or y_L2 < 0 or x_L2 < 0 or
            z_L2 >= outer_mask.shape[0] or
            y_L2 >= outer_mask.shape[1] or
            x_L2 >= outer_mask.shape[2]):
            print(f"    ✗ Coords out of bounds")
            return None
        
        slice_mask = outer_mask[z_L2].compute()
        
        if slice_mask[y_L2, x_L2] == 0:
            print(f"    ✗ Click not on mask")
            return None
        
        print(f"    ✓ Click IS on mask (value={slice_mask[y_L2, x_L2]})")
        
        from scipy import ndimage
        labeled, num_cc = ndimage.label(slice_mask > 0)
        
        print(f"    Found {num_cc} CCs")
        
        cc_label = labeled[y_L2, x_L2]
        
        if cc_label == 0:
            print(f"    ✗ CC label is 0")
            return None
        
        cc_id = cc_label - 1
        
        layer_data = label_data['layers'][str(z_L2)]
        if cc_id >= len(layer_data['outer']):
            print(f"    ✗ CC ID out of range")
            return None
        
        print(f"    ✓ Found CC #{cc_id}")
        return cc_id
    
    def _get_marker_position(self, label_id, z_L2, cc_id):
        """Get marker position from CC metadata"""
        print(f"\n  [GET MARKER]")
        print(f"    Input: label_id={label_id}, z_L2={z_L2}, cc_id={cc_id}")
        
        label_str = str(label_id)
        
        if label_str not in self.cc_metadata['labels']:
            print(f"    ✗ Label {label_id} not in metadata")
            return None, False
        
        label_data = self.cc_metadata['labels'][label_str]
        print(f"    ✓ Found label data")
        
        if str(z_L2) not in label_data['layers']:
            print(f"    ✗ Layer z={z_L2} not in metadata")
            available = sorted([int(z) for z in label_data['layers'].keys()])
            print(f"    Available layers: {available[:5]}...{available[-5:]}")
            return None, False
        
        layer_data = label_data['layers'][str(z_L2)]
        print(f"    ✓ Found layer data: {len(layer_data['outer'])} CCs")
        
        if cc_id >= len(layer_data['outer']):
            print(f"    ✗ CC ID {cc_id} out of range (max={len(layer_data['outer'])-1})")
            return None, False
        
        cc_info = layer_data['outer'][cc_id]
        print(f"    ✓ Found CC info")
        
        if 'skeleton_marker' not in cc_info:
            print(f"    ✗ No skeleton_marker in CC info")
            print(f"    CC keys: {list(cc_info.keys())}")
            return None, False
        
        marker_info = cc_info['skeleton_marker']
        print(f"    ✓ Found skeleton_marker")
        print(f"    Marker info: {marker_info}")
        
        position = tuple(marker_info['position'])
        has_skeleton = marker_info['has_skeleton']
        
        print(f"    → Position: {position}")
        print(f"    → Has skeleton: {has_skeleton}")
        
        return position, has_skeleton
    