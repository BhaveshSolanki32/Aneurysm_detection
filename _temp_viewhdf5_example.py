import h5py
import numpy as np

HDF5_FILE_PATH = r'processed_data_mra_v1.hdf5'

def get_patch_from_hdf5(series_uid, center_z, center_y, center_x, patch_size=96):
    """
    Extracts a 96^3 patch from the HDF5 file.
    """
    half_patch = patch_size // 2
    
    # Calculate the start and end coordinates for the slice
    start_z = center_z - half_patch
    end_z = center_z + half_patch
    start_y = center_y - half_patch
    end_y = center_y + half_patch
    start_x = center_x - half_patch
    end_x = center_x + half_patch
    
    with h5py.File(HDF5_FILE_PATH, 'r') as f:
        # Get the dataset for the specific scan
        scan_dataset = f[series_uid]
        
        # --- THIS IS THE EFFICIENT READ ---
        # Only this specific slice is read from disk into memory.
        patch = scan_dataset[start_z:end_z, start_y:end_y, start_x:end_x]
        
        # The data is stored as float16, so you might want to convert it
        # for training, especially if not using mixed-precision.
        patch = patch.astype(np.float32)
        
    return patch

# --- Example Usage ---
# Assume your new_localization_mra.csv tells you there's an aneurysm for
# patient '1.2.826.0.1.3680043.20571' at coordinates (110, 150, 200)

# The UID of the scan you want to access
target_uid = '1.2.826.0.1.3680043.20571' 

# The center coordinate of the patch you want to extract
# This would come from your localization CSV or a random sampling strategy
z, y, x = 110, 150, 200

# Get the patch
try:
    aneurysm_patch = get_patch_from_hdf5(target_uid, z, y, x)
    print(f"Successfully extracted a patch of shape: {aneurysm_patch.shape}")
    print(f"Data type of the patch: {aneurysm_patch.dtype}")
except KeyError:
    print(f"Error: Scan with UID '{target_uid}' not found in the HDF5 file.")