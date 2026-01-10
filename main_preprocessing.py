#!/usr/bin/env python3
"""
SAVE-3D Preprocessing Pipeline

This script processes 3D pathology data and generates all required
outputs for the SAVE-3D visualization application.

Usage:
    python main_preprocessing.py

Modify the CONFIGURATION section below for your data.

Resolution Level System:
========================
- Level 0 (L0): Input image resolution (highest resolution)
- Level N (LN): 2^N times downsampled from Level 0
  - Level 1 = Level 0 / 2
  - Level 2 = Level 0 / 4
  - Level 3 = Level 0 / 8

Input Formats Supported:
========================
Option A (Legacy): Separate mask files per label
    OUTER_MASKS_L2 = {'Benign': 'benign_outer.tiff', ...}
    INNER_MASKS_L2 = {'Benign': 'benign_inner.tiff', ...}

Option B (New): Single label volume
    LABEL_OUTER_PATH = 'outer_labels.tiff'  # Values: 0=bg, 1=label1, 2=label2, ...
    LABEL_INNER_PATH = 'inner_labels.tiff'  # Optional, can be None
    LABEL_DICT = {1: 'Benign', 2: 'HGPIN', ...}

Color Assignment:
=================
- LABEL_COLORS can be partial or empty
- Unspecified labels get auto-assigned rainbow colors
"""

# Configure PyVista/VTK for offscreen rendering BEFORE any imports
# This prevents "wglMakeCurrent failed" OpenGL errors on Windows
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN'] = '1'

# Suppress VTK warnings (non-fatal OpenGL context errors)
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

import gc
import json
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Import preprocessing modules
from preprocessing import (
    GPU_AVAILABLE,
    SimpleProgress,
    load_volume,
    load_labels_from_volume,
    assign_colors,
    build_pyramid_with_gpu_local_mean,
    upsample_mask_nearest,
    preprocess_skeleton_intersections,
    build_skeleton_graph,
    extract_skeleton_points_for_kdtree,
    precompute_2d_cc_and_adjacency,
    prebuild_morphology_meshes,
    prebuild_instance_meshes,
    prebuild_image_host_meshes,
    write_zarr_pathology,
)
from preprocessing.config import cleanup_gpu
from preprocessing.skeleton import compute_skeleton_labels


def main():
    """
    Main preprocessing pipeline
    """
    progress = SimpleProgress()
    
    print("="*80)
    print("SAVE-3D Preprocessing Pipeline")
    print("="*80)
    
    # =========================================================================
    # CONFIGURATION - Modify these settings for your data
    # =========================================================================
    
    # Resolution Level System
    PROCESSING_LEVEL = 2
    DOWNSAMPLE_FACTOR = 2 ** PROCESSING_LEVEL  # = 4
    
    # Input @ Level 0
    # IMAGE_PATH = "IDC-P_8x_111325.tif"
    IMAGE_PATH = "fc_2x_all_30_to_1200_cropped_092325.tif"
    VOXEL_SIZE_L0 = (0.9667 * 2, 0.9667 * 2, 0.9667 * 2)  # μm
    PYRAMID_LEVELS = 4
    
    # Skeleton @ Level 2
    # SKELETON_L2 = 'output_skeleton_32x_092625.tiff'
    SKELETON_L2 = 'Crypt_mask_cropped_mask_4x_010926-skeleton.tif'
    
    # Output
    OUT_ZARR = 'prostate_pathology.zarr'
    
    # ========= INPUT FORMAT SELECTION =========
    # Choose ONE of the following input formats:
    
    # --- Format A: Separate mask files (Legacy) ---
    USE_SEPARATE_MASKS = False  # Set to True for Format A, False for Format B
    
    if USE_SEPARATE_MASKS:
        # Outer masks @ Level 2 (required)
        OUTER_MASKS_L2 = {
            'Benign': 'Outer_mask_Benign_major_cc-32x_XYZ.tif',
            'HGPIN': 'Outer_mask_HGPIN_major_cc-32x_XYZ.tif',
            'AIP': 'Outer_mask_AIP_major_cc-32x_XYZ.tif',
            'IDCP': 'Outer_mask_IDCP_major_cc-32x_XYZ.tif'
        }

        INNER_MASKS_L2 = {
            'Benign': 'Inner_Benign_major_cc_32x_XYZ.tif',
            'HGPIN': 'Inner_HGPIN_major_cc_32x_XYZ.tif',
            'AIP': 'Inner_AIP_major_cc_32x_XYZ.tif',
            'IDCP': 'Inner_IDCP_major_cc_32x_XYZ.tif'
        }
        # Set to {} for no inner masks:
        # INNER_MASKS_L2 = {}
        
        # Optional: User-specified colors (can be partial or empty)
        USER_LABEL_COLORS = {
            'Benign': '#10B981',
            'HGPIN': '#F59E0B',
            'AIP': '#FB7185',
            'IDCP': '#DC2626'
        }
    
    else:
        # --- Format B: Single label volume (New) ---
        # LABEL_OUTER_PATH = '8x_mask-lbl.tif'  # Values: 0=bg, 1, 2, 3, ...
        LABEL_OUTER_PATH = 'Crypt_mask_cropped_mask_4x_010926-1-lbl.tif'  # Values: 0=bg, 1, 2, 3, ...
        LABEL_INNER_PATH = None  # Optional: set to None if no inner
        
        LABEL_DICT = {}  # {value: name} mapping, empty for auto-naming

        # Optional: User-specified colors (can be partial or empty)
        USER_LABEL_COLORS = {}
    
    # =========================================================================
    # END CONFIGURATION
    # =========================================================================
    
    # Compute L2 voxel size
    VOXEL_SIZE_L2 = tuple(v * DOWNSAMPLE_FACTOR for v in VOXEL_SIZE_L0)
    
    print(f"\nResolution Level System:")
    print(f"  Level 0 (L0): Input image resolution")
    print(f"  Level {PROCESSING_LEVEL} (L2): Processing level ({DOWNSAMPLE_FACTOR}x downsampled)")
    print(f"\n[INPUT] Image @ L0: {IMAGE_PATH}")
    print(f"[INPUT] Voxel size @ L0: {VOXEL_SIZE_L0} μm")
    print(f"[INPUT] Voxel size @ L2: {VOXEL_SIZE_L2} μm")
    print(f"[INPUT] Skeleton @ L2: {SKELETON_L2}")
    print(f"[OUTPUT] {OUT_ZARR}")
    print("="*80)
    
    try:
        # Step 1: Load image @ Level 0
        progress.update("Step 1/7: Loading image @ L0...")
        image_L0 = load_volume(IMAGE_PATH)
        progress.update(f"Image @ L0: {image_L0.shape}, {image_L0.dtype}, {image_L0.nbytes/1024**3:.1f}GB")
        
        # Store shape for upsampling
        if image_L0.ndim == 4:
            target_shape_L0 = image_L0.shape[:3]  # (nz, ny, nx) - ignore channel
        else:
            target_shape_L0 = image_L0.shape
        
        print(f"Target L0 shape for masks: {target_shape_L0}")
        
        # Step 2: Build image pyramid
        progress.update("Step 2/7: Building image pyramid...")
        img_pyramid = build_pyramid_with_gpu_local_mean(image_L0, PYRAMID_LEVELS)
        
        # Step 3: Load masks @ Level 2
        progress.update("Step 3/7: Loading masks @ L2...")
        
        if USE_SEPARATE_MASKS:
            # Format A: Load separate mask files
            outer_masks_L2 = {}
            inner_masks_L2 = {}
            label_names = list(OUTER_MASKS_L2.keys())
            
            for label_name in tqdm(label_names, desc="Loading outer masks", unit="label"):
                outer_masks_L2[label_name] = load_volume(OUTER_MASKS_L2[label_name])
            
            if INNER_MASKS_L2:
                for label_name in tqdm(label_names, desc="Loading inner masks", unit="label"):
                    if label_name in INNER_MASKS_L2:
                        inner_masks_L2[label_name] = load_volume(INNER_MASKS_L2[label_name])
            
            print(f"Loaded {len(outer_masks_L2)} outer masks, {len(inner_masks_L2)} inner masks")
            
            # Format A: all labels are user-named (for legend display)
            user_named_labels = label_names.copy()
        
        else:
            # Format B: Load single label volumes (auto-scan all unique values)
            outer_masks_L2, label_names, user_named_labels = load_labels_from_volume(
                LABEL_OUTER_PATH, LABEL_DICT
            )
            
            if LABEL_INNER_PATH is not None:
                inner_masks_L2, _, _ = load_labels_from_volume(LABEL_INNER_PATH, LABEL_DICT)
            else:
                inner_masks_L2 = {}
                print("No inner mask provided")
        
        has_inner_mask = len(inner_masks_L2) > 0
        print(f"\n[INFO] Inner mask: {'Yes' if has_inner_mask else 'No'}")
        print(f"[INFO] Labels: {len(label_names)} total")
        print(f"[INFO] User-named labels (for legend): {len(user_named_labels)}")
        
        # Assign colors (auto-fill unspecified)
        progress.update("Assigning label colors...")
        label_colors = assign_colors(label_names, USER_LABEL_COLORS)
        print(f"\n[COLORS] Label color assignment:")
        for name in label_names:
            source = "user" if name in USER_LABEL_COLORS else "auto"
            print(f"  {name}: {label_colors[name]} ({source})")
        
        # Verify mask sizes
        first_outer = outer_masks_L2[label_names[0]]
        expected_L2_shape = tuple(s // DOWNSAMPLE_FACTOR for s in target_shape_L0)
        print(f"\nExpected L2 shape: {expected_L2_shape}")
        print(f"Actual outer mask shape: {first_outer.shape}")
        
        if first_outer.shape != expected_L2_shape:
            print(f" Warning: Mask shape mismatch!")
            print(f" Expected: {expected_L2_shape}")
            print(f" Got: {first_outer.shape}")
            print(f" Will handle size mismatch during upsampling")
        
        # Step 3.5: Load skeleton & compute intersections @ L2
        progress.update("Step 3.5/7: Loading skeleton & computing intersections @ L2...")
        skeleton_L2 = load_volume(SKELETON_L2)
        progress.update(f"Skeleton @ L2: {skeleton_L2.shape}")
        
        # Verify skeleton size
        if skeleton_L2.shape != first_outer.shape:
            print(f" Warning: Skeleton shape {skeleton_L2.shape} != mask shape {first_outer.shape}")
        
        # ===         Pre-compute skeleton labels ONCE        ===
        # This avoids redundant computation in multiple functions
        progress.update("Pre-computing skeleton labels (shared)...")
        skeleton_binary = skeleton_L2 > 0
        skeleton_labels_vol, label_counts = compute_skeleton_labels(
            skeleton_binary, outer_masks_L2, label_names
        )
        # for name, count in label_counts.items():
        #     print(f"  {name}: {count:,} voxels")
        
        # Skeleton intersection (NO mesh generation) - reuses skeleton_labels_vol
        skeleton_metadata, _ = preprocess_skeleton_intersections(
            skeleton_L2, outer_masks_L2, label_names, label_colors,
            skeleton_labels=skeleton_labels_vol
        )
        
        # Step 4: Create combined labels
        progress.update("Step 4/7: Creating combined labels...")
        
        # Merge outer masks @ L2
        combined_L2 = np.zeros_like(outer_masks_L2[label_names[0]], dtype=np.uint8)
        for label_idx, label_name in enumerate(label_names, start=1):
            combined_L2[outer_masks_L2[label_name] > 0] = label_idx
        
        progress.update(f"Combined labels @ L2: {combined_L2.shape}, unique labels: {np.unique(combined_L2)}")
        
        # Upsample to L0 with exact size matching
        progress.update("  Upsampling combined labels L2 → L0...")
        combined_L0 = upsample_mask_nearest(combined_L2, target_shape_L0, factor=DOWNSAMPLE_FACTOR)
        
        # Verify final size
        if combined_L0.shape != target_shape_L0:
            raise ValueError(f"Combined labels size mismatch! Got {combined_L0.shape}, expected {target_shape_L0}")
        
        progress.update(f"  ✓ Combined labels @ L0: {combined_L0.shape} (exact match)")
        
        # Build combined labels pyramid
        progress.update("  Building combined labels pyramid...")
        lab_pyramid = [combined_L0.astype(np.uint32)]
        
        for level in range(1, PYRAMID_LEVELS):
            factor = 2 ** level
            if min(combined_L0.shape) // factor < 16:
                break
            
            lab_down = combined_L0[::factor, ::factor, ::factor].astype(np.uint32)
            lab_pyramid.append(lab_down)
            progress.update(f"  Level {level}: {lab_down.shape}")
        
        # Step 4.1: Extract skeleton points (reuses skeleton_labels_vol)
        progress.update("Step 4.1/7: Extracting skeleton points for KD-Tree...")
        skeleton_points = extract_skeleton_points_for_kdtree(
            skeleton_L2, outer_masks_L2, label_names, VOXEL_SIZE_L2,
            labels_L0=combined_L0,
            downsample_factor=DOWNSAMPLE_FACTOR,
            processing_level=PROCESSING_LEVEL,
            target_max_points=50000,
            skeleton_labels=skeleton_labels_vol
        )
        
        # Step 4.2: Build skeleton graph (reuses skeleton_labels_vol)
        progress.update("Step 4.2/7: Building skeleton graph...")
        skeleton_graph = build_skeleton_graph(
            skeleton_L2, outer_masks_L2, label_names, VOXEL_SIZE_L2,
            processing_level=PROCESSING_LEVEL,
            target_max_points=50000,
            skeleton_labels=skeleton_labels_vol
        )
        
        # Save skeleton graph
        skeleton_graph_path = Path(OUT_ZARR).parent / "skeleton_graph.json"
        with open(skeleton_graph_path, 'w') as f:
            json.dump(skeleton_graph, f)
        progress.update(f"✓ Skeleton graph saved to {skeleton_graph_path}")
        
        # Step 4.3: Prebuild morphology meshes
        progress.update("Step 4.3/7: Prebuilding morphology meshes...")
        morph_mesh_info = prebuild_morphology_meshes(
            outer_masks_L2, inner_masks_L2, label_names, label_colors,
            VOXEL_SIZE_L2, Path(OUT_ZARR).parent,
            processing_level=PROCESSING_LEVEL,
            target_faces=100000
        )
        progress.update(f"✓ Morphology meshes: {len(morph_mesh_info['meshes'])} files")
        
        # Step 4.4: Prebuild skeleton instance meshes
        progress.update("Step 4.4/7: Prebuilding skeleton instance meshes...")
        prebuild_instance_meshes(
            skeleton_points.get('instances', []),
            skeleton_points,
            outer_masks_L2,
            inner_masks_L2,
            label_names,
            VOXEL_SIZE_L2,
            Path(OUT_ZARR).parent
        )
        progress.update(f"✓ Skeleton instance meshes complete")
        
        # Save skeleton points
        skeleton_points_path = Path(OUT_ZARR).parent / "skeleton_points.json"
        with open(skeleton_points_path, 'w') as f:
            json.dump(skeleton_points, f)
        progress.update(f"✓ Skeleton points saved to {skeleton_points_path}")
        
        # Step 4.5: Prebuild image host meshes
        progress.update("Step 4.5/7: Prebuilding image host meshes...")
        image_host_mesh_info = prebuild_image_host_meshes(
            outer_masks_L2,
            inner_masks_L2,
            label_names,
            VOXEL_SIZE_L2,
            Path(OUT_ZARR).parent,
            processing_level=PROCESSING_LEVEL
        )
        
        # Add has_inner_mask to mesh info
        image_host_mesh_info['has_inner_mask'] = has_inner_mask
        
        # Save image host mesh info
        image_host_mesh_path = Path(OUT_ZARR).parent / "image_host_meshes.json"
        with open(image_host_mesh_path, 'w') as f:
            json.dump(image_host_mesh_info, f, indent=2)
        progress.update(f"✓ Image host meshes saved to {image_host_mesh_path}")
        
        # Step 5: 2D CC precomputation
        progress.update("Step 5/7: Precomputing 2D CC and adjacency @ L2...")
        
        cc_metadata = precompute_2d_cc_and_adjacency(
            outer_masks_L2, inner_masks_L2, skeleton_L2, label_names, 
            VOXEL_SIZE_L2, label_colors,
            processing_level=PROCESSING_LEVEL,
            downsample_factor=DOWNSAMPLE_FACTOR
        )
        
        # Save CC metadata
        cc_metadata_path = Path(OUT_ZARR).parent / "cc_metadata.json"
        progress.update(f"Saving CC metadata to {cc_metadata_path}...")
        with open(cc_metadata_path, 'w') as f:
            json.dump(cc_metadata, f, indent=2)
        progress.update("✓ CC metadata saved")
        
        # Step 6: Write Zarr
        progress.update("Step 6/7: Writing Zarr...")
        write_zarr_pathology(
            OUT_ZARR, img_pyramid, lab_pyramid, 
            outer_masks_L2, inner_masks_L2, skeleton_metadata,
            VOXEL_SIZE_L0, label_colors, label_names, user_named_labels,
            PROCESSING_LEVEL, DOWNSAMPLE_FACTOR, VOXEL_SIZE_L2
        )
        
        total_time = time.time() - progress.start_time
        print("\n" + "="*80)
        print(f"[OK] Preprocessing Complete! Total time: {total_time/60:.1f} minutes")
        print(f"[OUTPUT] {OUT_ZARR}")
        print(f"[OUTPUT] {cc_metadata_path}")
        print("="*80)
        
        # Summary
        print("\nOutput Summary:")
        print(f"  Resolution Levels:")
        print(f"    L0 (input): voxel = {VOXEL_SIZE_L0} μm")
        print(f"    L2 (processing): voxel = {VOXEL_SIZE_L2} μm ({DOWNSAMPLE_FACTOR}x)")
        print(f"  Image pyramid: {len(img_pyramid)} levels @ L0")
        print(f"  Combined labels pyramid: {len(lab_pyramid)} levels @ L0")
        print(f"  Outer masks: {len(outer_masks_L2)} labels @ L2")
        print(f"  Inner masks: {len(inner_masks_L2)} labels @ L2 ({'enabled' if has_inner_mask else 'disabled'})")
        print(f"  Skeleton metadata: {len(skeleton_metadata)} components @ L2")
        print(f"  CC metadata: {sum(len(cc_metadata['labels'][lid]['layers']) for lid in cc_metadata['labels'])} total layers")
        print(f"\nLabel Colors:")
        for name in label_names:
            print(f"  {name}: {label_colors[name]}")
        
        print(f"\nNext: python main_app.py {OUT_ZARR}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup_gpu()
        gc.collect()


if __name__ == "__main__":
    main()
