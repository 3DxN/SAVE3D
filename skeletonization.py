import numpy as np
import kimimaro
import tifffile
import skimage.transform

# ========== 1️⃣ Load TIFF Image ==========
tiff_path = "example/Crypt_mask_cropped_mask_4x_010926-label_and_skeleton/Crypt_mask_cropped_mask_4x_010926-mask.tif"  # Replace with your image path
labels = tifffile.imread(tiff_path).astype(np.uint8)  # Ensure uint8 format

# Ensure image shape is (Z, Y, X)
print(f"Original image dimensions: {labels.shape}")

# ========== 2️⃣ Downscale Resolution (XY reduced by 50%) ==========
#labels_resized = skimage.transform.rescale(labels, (1, 0.25, 0.25), order=0, preserve_range=True).astype(np.uint8)

#print(f"Image dimensions after downscaling: {labels_resized.shape}")

# ========== 3️⃣ Set Anisotropy (3-axis resolution) ==========
# Original resolution (XY = 3.867, Z = 10)
voxel_spacing = (1, 1, 1)  # XY resolution becomes 2x (due to 50% XY downscaling)

# ========== 4️⃣ Perform Skeletonization ==========
skels = kimimaro.skeletonize(
    labels,  
    teasar_params={
        "scale": 1.0,  # No scaling
        "const": 100,  # Preserve fine connections in skeleton
        "pdrf_scale": 10000,
        "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3500,
        "soma_detection_threshold": 750,
        "soma_invalidation_const": 300,
        "soma_invalidation_scale": 2,
        "max_paths": 300,
    },
    dust_threshold=50,  # Filter out small fragments, preserve fine connections
    anisotropy=voxel_spacing,  # Specify actual XY/Z resolution
    fix_branching=True,
    fix_borders=True,
    fill_holes=True,  # Ensure skeleton completeness
    progress=True,
    parallel=1,  # Speed up computation
)

print("✅ Skeletonization complete!")

# ========== 5️⃣ Save as TIFF File ==========
def save_skeleton_as_tiff(skeletons, output_tiff_path, volume_shape):
    """Save Kimimaro Skeleton results as TIFF"""
    skeleton_volume = np.zeros(volume_shape, dtype=np.uint8)  # Create empty 3D volume (all zeros)

    for obj_id, skel in skeletons.items():
        for vertex in skel.vertices:  # ✅ Fixed: use `vertices` instead of `nodes`
            z, y, x = map(int, vertex)  # Convert to integer indices
            if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                skeleton_volume[z, y, x] = 255  # Set to 255 to mark skeleton points

    # Save as TIFF file
    tifffile.imwrite(output_tiff_path, skeleton_volume)
    print(f"✅ TIFF file saved: {output_tiff_path}")


# Specify output TIFF file path
output_tiff_path = "example/Crypt_mask_cropped_mask_4x_010926-label_and_skeleton/Crypt_mask_cropped_mask_4x_010926-skeleton.tif"

# Save skeleton image
save_skeleton_as_tiff(skels, output_tiff_path, labels.shape)
