# Save this as create_patch_manifest_hdf5_train_test.py

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import ast

# --- 1. CONFIGURATION ---
# Path to your new localization CSV with both positive and negative cases
LOCALIZATION_CSV_PATH = r'processed_data_unified\all_scan_csv_label.csv' 
# Path to the HDF5 file containing all preprocessed 3D scans
HDF5_DATA_PATH = r'processed_data_unified\processed_scans.hdf5'
# Directory to save the output manifest CSV files
OUTPUT_DIR = r'aneurysm_dataset_manifests_hdf5' 
PATCH_SIZE = 96
STRIDE = 40
# The data will be split into training and testing sets based on this ratio.
# For example, 0.8 means 80% for training and 20% for testing.
TRAIN_SIZE = 0.8
# Using almost all available CPU cores for processing
NUM_PROCESSES = max(1, cpu_count() - 2)

# --- 2. HELPER & WORKER FUNCTIONS ---

def generate_start_coords(scan_dim, patch_size, stride):
    """
    Calculates the starting coordinates for patch extraction along one dimension.
    This ensures that the entire scan is covered.
    """
    last_possible_start = scan_dim - patch_size
    if last_possible_start < 0: return [0]
    coords = list(range(0, last_possible_start, stride))
    # Ensure the very last possible patch is included for full coverage
    if not coords or coords[-1] < last_possible_start:
        coords.append(last_possible_start)
    return coords

def process_scan_for_manifest(args):
    """
    Processes a single scan to generate a list of manifest rows.
    This function does NOT save patch files; it only creates the manifest entries.
    """
    series_uid, aneurysms_in_scan, base_scan_info, config = args
    hdf5_path = config['hdf5_path']
    patch_size = config['patch_size']
    stride = config['stride']
    location_cols = config['location_cols']

    try:
        # Open the HDF5 file and get the shape of the specific scan dataset
        with h5py.File(hdf5_path, 'r') as f:
            scan_shape = f[series_uid].shape
            scan_d, scan_h, scan_w = scan_shape
    except (FileNotFoundError, KeyError):
        # If the HDF5 file or the specific scan UID is not found, skip it
        print(f"Warning: Could not find scan for series_uid {series_uid} in {hdf5_path}")
        return []

    # Generate all possible patch starting coordinates
    z_coords = generate_start_coords(scan_d, patch_size, stride)
    y_coords = generate_start_coords(scan_h, patch_size, stride)
    x_coords = generate_start_coords(scan_w, patch_size, stride)

    manifest_rows_for_this_scan = []
    
    # Iterate through every possible patch in the 3D scan
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                patch_start = (z, y, x)
                patch_end = (z + patch_size, y + patch_size, x + patch_size)
                
                aneurysms_in_patch = []
                # Check if any of the known aneurysms fall within this patch
                for an in aneurysms_in_scan:
                    an_coord = (an['coord_z'], an['coord_y'], an['coord_x'])
                    if (patch_start[0] <= an_coord[0] < patch_end[0] and
                        patch_start[1] <= an_coord[1] < patch_end[1] and
                        patch_start[2] <= an_coord[2] < patch_end[2]):
                        aneurysms_in_patch.append(an)

                is_present = 1 if len(aneurysms_in_patch) > 0 else 0
                artery_labels = np.zeros(len(location_cols), dtype=int)
                relative_coords = []

                if is_present:
                    # If aneurysms are present, aggregate their artery locations and calculate relative coordinates
                    for an in aneurysms_in_patch:
                        current_artery_labels = [an[col] for col in location_cols]
                        artery_labels = np.bitwise_or(artery_labels, current_artery_labels)
                        
                        relative_z = an['coord_z'] - patch_start[0]
                        relative_y = an['coord_y'] - patch_start[1]
                        relative_x = an['coord_x'] - patch_start[2]
                        relative_coords.append((relative_z, relative_y, relative_x))
                
                # Create the manifest row, starting with the base info for the scan
                manifest_row = base_scan_info.copy()
                manifest_row.update({
                    'start_z': z, 'start_y': y, 'start_x': x,
                    'Aneurysm Present': is_present,
                    'relative_coords': str(relative_coords) if is_present else '[]'
                })
                
                # Add the artery location columns
                for i, col in enumerate(location_cols):
                    manifest_row[col] = artery_labels[i]
                
                manifest_rows_for_this_scan.append(manifest_row)
    
    return manifest_rows_for_this_scan

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print("--- Starting HDF5 Patch Manifest Creation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_master = pd.read_csv(LOCALIZATION_CSV_PATH)
    
    # Define the artery location columns based on your CSV
    location_cols = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
        'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
    ]
    
    # These columns from the master CSV will be copied to every patch row for that scan
    base_info_cols = ['SeriesInstanceUID', 'Modality'] + location_cols

    print("Step 1: Preparing tasks for parallel processing...")
    # Separate aneurysm locations from the main dataframe
    df_aneurysms = df_master[df_master['Aneurysm Present'] == 1].dropna(subset=['coord_z']).copy()
    df_aneurysms[['coord_z', 'coord_y', 'coord_x']] = df_aneurysms[['coord_z', 'coord_y', 'coord_x']].astype(int)
    
    # Prepare tasks for the multiprocessing pool
    tasks = []
    config = {
        'hdf5_path': HDF5_DATA_PATH, 
        'patch_size': PATCH_SIZE, 
        'stride': STRIDE, 
        'location_cols': location_cols
    }
    
    # Get unique scans to process, dropping duplicates
    df_scans = df_master.drop_duplicates(subset=['SeriesInstanceUID'])

    for _, row in df_scans.iterrows():
        series_uid = row['SeriesInstanceUID']
        # Get all aneurysm annotations for the current scan
        aneurysms_in_scan = [an_row.to_dict() for _, an_row in df_aneurysms[df_aneurysms['SeriesInstanceUID'] == series_uid].iterrows()]
        # Get the base information (UID, Modality, etc.) for this scan
        base_scan_info = row.to_dict()
        tasks.append((series_uid, aneurysms_in_scan, base_scan_info, config))

    print(f"Prepared {len(tasks)} scans for processing.")
    print(f"\nStep 2: Generating manifest with {NUM_PROCESSES} processes...")
    
    all_manifest_rows = []
    # Use a multiprocessing pool to process scans in parallel
    with Pool(processes=NUM_PROCESSES) as pool:
        for result_list in tqdm(pool.imap_unordered(process_scan_for_manifest, tasks), total=len(tasks)):
            all_manifest_rows.extend(result_list)

    print("\nStep 3: Performing patient-level train-test split and saving manifest CSVs...")
    df_manifest = pd.DataFrame(all_manifest_rows)
    
    # Ensure correct column order
    ordered_cols = ['SeriesInstanceUID', 'Modality', 'start_z', 'start_y', 'start_x', 
                    'Aneurysm Present', 'relative_coords'] + location_cols
    df_manifest = df_manifest[ordered_cols]

    # Perform a patient-level split to ensure no patient data leaks between sets
    all_uids = df_manifest['SeriesInstanceUID'].unique()
    train_uids, test_uids = train_test_split(
        all_uids, 
        train_size=TRAIN_SIZE, 
        random_state=42, 
        shuffle=True
    )
    
    uid_splits = {'train': train_uids, 'test': test_uids}

    for split_name, uids in uid_splits.items():
        # Filter the manifest dataframe for the UIDs in the current split
        split_df = df_manifest[df_manifest['SeriesInstanceUID'].isin(uids)]
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}_manifest.csv")
        split_df.to_csv(output_path, index=False)
        print(f"Saved {split_name}_manifest.csv with {len(split_df):,} patches.")
        
        # Print statistics for the created manifest
        if not split_df.empty:
            pos_count = split_df['Aneurysm Present'].sum()
            neg_count = len(split_df) - pos_count
            pos_percentage = (pos_count / len(split_df)) * 100 if len(split_df) > 0 else 0
            print(f" -> Contains {pos_count:,} positive patches and {neg_count:,} negative patches ({pos_percentage:.2f}% positive).")

    print("\n--- Manifest Creation Complete ---")
    print(f"All manifest files are saved in the '{OUTPUT_DIR}' directory.")