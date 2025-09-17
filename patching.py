# Save this as create_patch_manifest.py (FINAL - Generates CENTER Coordinates)

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
MASTER_CSV_PATH = 'master_df_positive_ct.csv' 
NPY_DATA_DIR = 'processed_data_npy'
OUTPUT_DIR = 'aneurysm_dataset_manifests_padded' # New, clear name
PATCH_SIZE = 96
STRIDE =  30# Using your stride
TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
NUM_PROCESSES = max(1, cpu_count() - 1)

# --- WORKER FUNCTION ---
def process_scan_for_centers(args):
    """
    Processes a single scan to generate a manifest of CENTER coordinates for patches.
    This prepares the data for an on-the-fly padding Dataloader.
    """
    series_uid, aneurysms_in_scan, config = args
    npy_data_dir, patch_size, stride = config['npy_data_dir'], config['patch_size'], config['stride']

    try:
        scan_shape = np.load(os.path.join(npy_data_dir, f"{series_uid}.npy"), mmap_mode='r').shape
        scan_d, scan_h, scan_w = scan_shape
    except FileNotFoundError:
        return []

    manifest_rows = []
    # Loop through all possible CENTER coordinates in the original scan
    for z in range(0, scan_d, stride):
        for y in range(0, scan_h, stride):
            for x in range(0, scan_w, stride):
                # This (z,y,x) is the CENTER of our conceptual patch
                
                # Define the patch boundaries relative to the original scan's coordinate system
                patch_start = (z - patch_size // 2, y - patch_size // 2, x - patch_size // 2)
                patch_end = (z + patch_size // 2, y + patch_size // 2, x + patch_size // 2)

                aneurysms_in_patch = []
                for an in aneurysms_in_scan:
                    an_coord = (an['coord_z'], an['coord_y'], an['coord_x'])
                    if (patch_start[0] <= an_coord[0] < patch_end[0] and
                        patch_start[1] <= an_coord[1] < patch_end[1] and
                        patch_start[2] <= an_coord[2] < patch_end[2]):
                        aneurysms_in_patch.append(an)

                is_present = 1 if len(aneurysms_in_patch) > 0 else 0
                
                # The manifest stores the CENTER coordinate
                manifest_row = {
                    'series_uid': series_uid,
                    'center_z': z, 'center_y': y, 'center_x': x,
                    'Aneurysm Present': is_present
                }
                manifest_rows.append(manifest_row)
    return manifest_rows

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print("--- Starting PADDED Manifest Creation (Generates Center Coords) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_master = pd.read_csv(MASTER_CSV_PATH)
    df_aneurysms = df_master.dropna(subset=['coord_z']).copy()
    df_aneurysms[['coord_z', 'coord_y', 'coord_x']] = df_aneurysms[['coord_z', 'coord_y', 'coord_x']].astype(int)
    
    tasks = []
    config = { 'npy_data_dir': NPY_DATA_DIR, 'patch_size': PATCH_SIZE, 'stride': STRIDE }
    
    for series_uid in df_master['SeriesInstanceUID'].unique():
        if not os.path.exists(os.path.join(NPY_DATA_DIR, f"{series_uid}.npy")):
            continue
        aneurysms_in_scan = [row.to_dict() for _, row in df_aneurysms[df_aneurysms['SeriesInstanceUID'] == series_uid].iterrows()]
        tasks.append((series_uid, aneurysms_in_scan, config))

    all_manifest_rows = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result_list in tqdm(pool.imap_unordered(process_scan_for_centers, tasks), total=len(tasks)):
            all_manifest_rows.extend(result_list)

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

    print(f"\n--- Manifest Creation Complete ---")