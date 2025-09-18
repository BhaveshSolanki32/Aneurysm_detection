# Save this as create_patch_manifest.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

# --- 1. CONFIGURATION ---
MASTER_CSV_PATH = 'master_df_positive_ct.csv' 
NPY_DATA_DIR = 'processed_data_npy'
OUTPUT_DIR = 'aneurysm_dataset_manifests' # New folder for just the CSVs
PATCH_SIZE = 96
STRIDE = 34
TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
NUM_PROCESSES = max(1, cpu_count() - 1)

# --- 2. WORKER FUNCTION ---
def process_scan_for_manifest(args):
    """
    Processes a single scan to generate a list of manifest rows.
    IT DOES NOT SAVE PATCH FILES.
    """
    series_uid, aneurysms_in_scan, config = args
    npy_data_dir, patch_size, stride, location_cols = config['npy_data_dir'], config['patch_size'], config['stride'], config['location_cols']

    try:
        # We only need the shape, so we can use memmap to avoid loading the whole file
        scan_shape = np.load(os.path.join(npy_data_dir, f"{series_uid}.npy"), mmap_mode='r').shape
        scan_d, scan_h, scan_w = scan_shape
    except FileNotFoundError:
        return []

    z_coords = generate_start_coords(scan_d, patch_size, stride)
    y_coords = generate_start_coords(scan_h, patch_size, stride)
    x_coords = generate_start_coords(scan_w, patch_size, stride)

    manifest_rows_for_this_scan = []
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                patch_start = (z, y, x)
                patch_end = (z + patch_size, y + patch_size, x + patch_size)
                
                aneurysms_in_patch = []
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
                    for an in aneurysms_in_patch:
                        current_artery_labels = [an[col] for col in location_cols]
                        artery_labels = np.bitwise_or(artery_labels, current_artery_labels)
                        relative_z = an['coord_z'] - patch_start[0]
                        relative_y = an['coord_y'] - patch_start[1]
                        relative_x = an['coord_x'] - patch_start[2]
                        relative_coords.append((relative_z, relative_y, relative_x))

                # The manifest row contains the location of the patch, not a filename
                manifest_row = {
                    'series_uid': series_uid,
                    'start_z': z, 'start_y': y, 'start_x': x,
                    'Aneurysm Present': is_present,
                    'relative_coords': str(relative_coords) if is_present else '[]'
                }
                for i, col in enumerate(location_cols):
                    manifest_row[col] = artery_labels[i]
                
                manifest_rows_for_this_scan.append(manifest_row)
    
    return manifest_rows_for_this_scan

def generate_start_coords(scan_dim, patch_size, stride):
    """Helper function to calculate start coordinates, ensuring full coverage."""
    last_possible_start = scan_dim - patch_size
    if last_possible_start < 0: return [0]
    coords = list(range(0, last_possible_start + 1, stride))
    if coords[-1] != last_possible_start: coords.append(last_possible_start)
    return coords

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print("--- Starting Patch Manifest Creation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_master = pd.read_csv(MASTER_CSV_PATH)
    location_cols = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
        'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation'
    ]

    print("Step 1: Preparing tasks for parallel processing...")
    df_aneurysms = df_master.dropna(subset=['coord_z']).copy()
    df_aneurysms[['coord_z', 'coord_y', 'coord_x']] = df_aneurysms[['coord_z', 'coord_y', 'coord_x']].astype(int)
    
    tasks = []
    config = {
        'npy_data_dir': NPY_DATA_DIR, 'patch_size': PATCH_SIZE, 
        'stride': STRIDE, 'location_cols': location_cols
    }
    
    for series_uid in df_master['SeriesInstanceUID'].unique():
        if not os.path.exists(os.path.join(NPY_DATA_DIR, f"{series_uid}.npy")):
            continue
        aneurysms_in_scan = [row.to_dict() for _, row in df_aneurysms[df_aneurysms['SeriesInstanceUID'] == series_uid].iterrows()]
        tasks.append((series_uid, aneurysms_in_scan, config))

    print(f"\nStep 2: Generating manifest with {NUM_PROCESSES} processes...")
    all_manifest_rows = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result_list in tqdm(pool.imap_unordered(process_scan_for_manifest, tasks), total=len(tasks)):
            all_manifest_rows.extend(result_list)

    print("\nStep 3: Performing patient-level split and saving manifest CSVs...")
    df_manifest = pd.DataFrame(all_manifest_rows)
    
    all_uids = df_manifest['series_uid'].unique()
    train_uids, temp_uids = train_test_split(all_uids, train_size=TRAIN_SIZE, random_state=42)
    relative_test_size = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    test_uids, val_uids = train_test_split(temp_uids, test_size=relative_test_size, random_state=42)
    
    uid_splits = {'train': train_uids, 'test': test_uids, 'validation': val_uids}

    for split_name, uids in uid_splits.items():
        split_df = df_manifest[df_manifest['series_uid'].isin(uids)]
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}_manifest.csv")
        split_df.to_csv(output_path, index=False)
        print(f"Saved {split_name}_manifest.csv with {len(split_df)} patches.")
        if not split_df.empty:
            pos_count = split_df['Aneurysm Present'].sum()
            print(f" -> {pos_count} positive patches.")

    print("\n--- Manifest Creation Complete ---")
    print(f"All manifest files saved in: {OUTPUT_DIR}")