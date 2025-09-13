# Save this as create_dataset_patches.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---

# --- INPUT PATHS ---
# Path to your master CSV with all scan info and new coordinate columns
MASTER_CSV_PATH = 'master_df_positive_ct.csv' 
# Path to the folder containing your preprocessed .npy scan files
NPY_DATA_DIR = 'processed_data_npy'

# --- OUTPUT PATHS ---
# A base folder where all the output will be saved
OUTPUT_DIR = 'aneurysm_patched_dataset_padded'

# --- PATCHING PARAMETERS ---
PATCH_SIZE = 96
STRIDE = 33

# --- DATA SPLIT RATIOS ---
TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1 # Must sum to 1.0 with the others

# --- 2. SCRIPT LOGIC ---

def generate_start_coords(scan_dim, patch_size, stride):
    """
    Calculates all possible start coordinates for a given dimension,
    ensuring the final patch that covers the edge is included.
    """
    last_possible_start = scan_dim - patch_size
    if last_possible_start < 0:
        # If scan is smaller than patch, we can only take one patch from the start.
        return [0]
    
    coords = list(range(0, last_possible_start + 1, stride))
    
    # If the stride doesn't perfectly land on the last possible start position, add it.
    # This guarantees the final corner is always covered by a patch.
    if coords[-1] != last_possible_start:
        coords.append(last_possible_start)
        
    return coords

def create_patched_dataset_with_padding():
    """
    Main function to perform patient-level split, generate patches via a padded sliding window,
    save each patch to disk, and create corresponding metadata CSVs.
    """
    print("--- Starting Padded Dataset Creation ---")

    # --- A. SETUP AND DATA LOADING ---
    try:
        df_master = pd.read_csv(MASTER_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Master CSV not found at {MASTER_CSV_PATH}")
        return

    for split in ['train', 'test', 'validation']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'patches'), exist_ok=True)

    location_cols = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
        'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
    ]

    # --- B. PATIENT-LEVEL DATA SPLIT ---
    print("Step 1: Performing patient-level (SeriesInstanceUID) data split...")
    all_uids = df_master['SeriesInstanceUID'].unique()
    train_uids, temp_uids = train_test_split(all_uids, train_size=TRAIN_SIZE, random_state=42)
    relative_test_size = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    test_uids, val_uids = train_test_split(temp_uids, test_size=relative_test_size, random_state=42)
    uid_splits = {'train': list(train_uids), 'test': list(test_uids), 'validation': list(val_uids)}
    print(f"Split complete: {len(train_uids)} train, {len(test_uids)} test, {len(val_uids)} validation scans.")

    # --- C. PATCH GENERATION AND SAVING ---
    all_manifest_rows = []
    pad_width = PATCH_SIZE // 2  # This is 48
    
    df_aneurysms = df_master.dropna(subset=['coord_z']).copy()
    df_aneurysms[['coord_z', 'coord_y', 'coord_x']] = df_aneurysms[['coord_z', 'coord_y', 'coord_x']].astype(int)

    print("\nStep 2: Generating and saving patches for all scans...")
    for series_uid in tqdm(all_uids, desc="Processing Scans"):
        try:
            full_scan = np.load(os.path.join(NPY_DATA_DIR, f"{series_uid}.npy"))
            scan_d, scan_h, scan_w = full_scan.shape
        except FileNotFoundError:
            print(f"Warning: NPY file for {series_uid} not found. Skipping.")
            continue
            
        current_split = [k for k, v in uid_splits.items() if series_uid in v][0]

        # --- i. Pad the scan ---
        padded_scan = np.pad(full_scan, pad_width=pad_width, mode='constant', constant_values=full_scan.min())

        # --- ii. Get aneurysm info and update coordinates to be relative to the PADDED scan ---
        aneurysms_in_scan = []
        aneurysm_rows = df_aneurysms[df_aneurysms['SeriesInstanceUID'] == series_uid]
        for _, row in aneurysm_rows.iterrows():
            an_dict = row.to_dict()
            # Shift original coordinates by the padding amount
            an_dict['padded_coord_z'] = an_dict['coord_z'] + pad_width
            an_dict['padded_coord_y'] = an_dict['coord_y'] + pad_width
            an_dict['padded_coord_x'] = an_dict['coord_x'] + pad_width
            aneurysms_in_scan.append(an_dict)

        # --- iii. Generate start coordinates for the sliding window ---
        # These coordinates are for the top-left corner of the crop in the PADDED array
        z_coords = generate_start_coords(scan_d, PATCH_SIZE, STRIDE)
        y_coords = generate_start_coords(scan_h, PATCH_SIZE, STRIDE)
        x_coords = generate_start_coords(scan_w, PATCH_SIZE, STRIDE)

        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    patch_start = (z, y, x)
                    patch_end = (z + PATCH_SIZE, y + PATCH_SIZE, x + PATCH_SIZE)
                    
                    aneurysms_in_patch = []
                    for an in aneurysms_in_scan:
                        an_coord = (an['padded_coord_z'], an['padded_coord_y'], an['padded_coord_x'])
                        if (patch_start[0] <= an_coord[0] < patch_end[0] and
                            patch_start[1] <= an_coord[1] < patch_end[1] and
                            patch_start[2] <= an_coord[2] < patch_end[2]):
                            aneurysms_in_patch.append(an)

                    is_present = 1 if len(aneurysms_in_patch) > 0 else 0
                    artery_labels = np.zeros(len(location_cols), dtype=int)
                    relative_coords = []

                    if is_present:
                        for an in aneurysms_in_patch:
                            current_artery_labels = [an[col] for col in location_cols]
                            artery_labels = np.bitwise_or(artery_labels, current_artery_labels)
                            # Calculate coordinates relative to the patch's top-left corner
                            relative_z = an['padded_coord_z'] - patch_start[0]
                            relative_y = an['padded_coord_y'] - patch_start[1]
                            relative_x = an['padded_coord_x'] - patch_start[2]
                            relative_coords.append((relative_z, relative_y, relative_x))

                    # Crop from the PADDED scan
                    patch_array = padded_scan[z:z+PATCH_SIZE, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patch_filename = f"{series_uid}_z{z}_y{y}_x{x}.npy"
                    save_path = os.path.join(OUTPUT_DIR, current_split, 'patches', patch_filename)
                    np.save(save_path, patch_array)

                    manifest_row = {
                        'series_uid': series_uid,
                        'patch_filename': patch_filename,
                        'split': current_split,
                        'Aneurysm Present': is_present,
                        'relative_coords': str(relative_coords) if is_present else '[]'
                    }
                    for i, col in enumerate(location_cols):
                        manifest_row[col] = artery_labels[i]
                    
                    all_manifest_rows.append(manifest_row)

    # --- D. SAVE THE FINAL MANIFEST CSVs ---
    print("\nStep 3: Saving manifest CSV files...")
    df_manifest = pd.DataFrame(all_manifest_rows)
    for split in ['train', 'test', 'validation']:
        split_df = df_manifest[df_manifest['split'] == split].drop(columns=['split'])
        output_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
        split_df.to_csv(output_path, index=False)
        print(f"Saved {split}_manifest.csv with {len(split_df)} patches.")
        if not split_df.empty:
            pos_count = split_df['Aneurysm Present'].sum()
            print(f" -> {pos_count} positive patches.")

    print("\n--- Dataset Creation Complete ---")
    print(f"All files saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    create_patched_dataset_with_padding()