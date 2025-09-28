# Save this as save_unified.py
import os
import time
import pandas as pd
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast
import h5py

# --- UNIFIED IMPORTS ---
# Make sure both preprocessing scripts are in the same directory
from prep_mr import preprocess_mri_scan
from preprocess_ct import preprocess_cta_scan

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)

# --- UNIFIED CONFIGURATION ---
BASE_PATH = r'rsna-intracranial-aneurysm-detection\series'
OUTPUT_DIR = 'processed_data_unified' # A single directory for all outputs
OUTPUT_HDF5_PATH = os.path.join(OUTPUT_DIR, 'processed_scans.hdf5')
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'preprocessing_log.csv')
NEW_LOCALIZATION_CSV_PATH = os.path.join(OUTPUT_DIR, 'localization_manifest.csv')

MAX_SCANS_TO_PROCESS = None # Set to a number (e.g., 100) for testing, or None to run all
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv'
NUM_PROCESSES = 8 # Adjust based on your CPU cores


def process_and_save_scan(args):
    """
    A unified wrapper function that calls the correct preprocessing pipeline
    based on the scan's modality (CTA vs. MRA/other).
    """
    series_uid, base_path, hdf5_path, coords_list, modality = args
    
    # 1. EFFICIENT SKIP: Check if this scan is already in the HDF5 file.
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if series_uid in f:
                return {'SeriesInstanceUID': series_uid, 'status': 'Skipped', 'shape_z_y_x': f[series_uid].shape, 'error': 'Dataset already exists in HDF5', 'final_coords_zyx': None}
    except (FileNotFoundError, OSError):
        pass # File doesn't exist yet, which is fine.
        
    try:
        folder_path = os.path.join(base_path, series_uid)

        # 2. DISPATCHER: Call the correct preprocessing function based on modality.
        print(f"Processing {series_uid} (Modality: {modality})")
        if modality == 'CTA':
            processed_array, final_spacing, final_coords_list = preprocess_cta_scan(
                folder_path,
                initial_coords_list=coords_list
            )
        else: # Handle MRA and any other non-CTA modalities
            processed_array, final_spacing, final_coords_list = preprocess_mri_scan(
                folder_path,
                modality=modality,
                initial_coords_list=coords_list
            )
        
        if processed_array.size == 0:
             raise ValueError("Preprocessing returned an empty array.")
        
        processed_array_fp16 = processed_array.astype(np.float16)

        # 3. ROBUST LOCK: Use a file-based lock to prevent write collisions.
        lock_file_path = hdf5_path + ".lock"
        while True:
            try:
                fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.5)

        try:
            # This section is now guaranteed to be atomic
            with h5py.File(hdf5_path, 'a') as f:
                f.create_dataset(
                    name=series_uid, data=processed_array_fp16, shape=processed_array_fp16.shape,
                    dtype='f2', chunks=(32, 32, 32), compression="gzip"
                )
        finally:
            # CRITICAL: Always release the lock
            os.remove(lock_file_path)
            
        return {'SeriesInstanceUID': series_uid, 'status': 'Success', 'shape_z_y_x': processed_array.shape, 'error': None, 'final_coords_zyx': final_coords_list}

    except Exception as e:
        return {'SeriesInstanceUID': series_uid, 'status': 'Failed', 'shape_z_y_x': None, 'error': str(e), 'final_coords_zyx': None}

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        df_train = pd.read_csv(r'rsna-intracranial-aneurysm-detection\train.csv')
        df_loc = pd.read_csv(ORIGINAL_LOCALIZATION_CSV)
        
        df_train_unique = df_train.drop_duplicates(subset=['SeriesInstanceUID'])
        df_merged = pd.merge(df_train_unique, df_loc, on='SeriesInstanceUID', how='left')
        
        uids_to_process = df_merged['SeriesInstanceUID'].unique().tolist()
        print(f"Found {len(uids_to_process)} unique SeriesInstanceUIDs to process (CTA & non-CTA).")

        if MAX_SCANS_TO_PROCESS is not None:
            uids_to_process = uids_to_process[:MAX_SCANS_TO_PROCESS]
            print(f"--- LIMITING to processing {len(uids_to_process)} scans for this run. ---")
            
        grouped = df_merged.groupby('SeriesInstanceUID')
        
        tasks = []
        for uid, group in tqdm(grouped, desc="Preparing tasks"):
            if uid not in uids_to_process:
                continue
            
            modality = group['Modality'].iloc[0]
            coords_list_for_series = []
            if not group['SOPInstanceUID'].isnull().all():
                for _, row in group.iterrows():
                    try:
                        coords_dict = ast.literal_eval(row['coordinates'])
                        coords_list_for_series.append({'sop_uid': row['SOPInstanceUID'], 'coords_xy': coords_dict, 'location': row['location']})
                    except (ValueError, SyntaxError, TypeError):
                        continue
            
            final_coords_arg = coords_list_for_series if coords_list_for_series else None
            tasks.append((uid, BASE_PATH, OUTPUT_HDF5_PATH, final_coords_arg, modality))

    except FileNotFoundError as e:
        print(f"Error reading CSV files: {e}")
        exit()
    
    print(f"Starting preprocessing with {NUM_PROCESSES} parallel processes...")
    
    results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        async_results = []
        for task in tasks:
            series_uid = task[0]
            job = pool.apply_async(process_and_save_scan, args=(task,))
            async_results.append((series_uid, job))

        for series_uid, job in tqdm(async_results, total=len(tasks), desc="Processing Scans"):
            try:
                result = job.get(timeout=600) 
                results.append(result)
            except Exception as e:
                print(f"\n---!!! HANG DETECTED OR CRITICAL ERROR !!! ---")
                print(f"Process failed on SeriesInstanceUID: {series_uid} | Error: {e}")
                print(f"---------------------------------------------")
                results.append({'SeriesInstanceUID': series_uid, 'status': 'Failed_Hang_Or_Error', 'shape_z_y_x': None, 'error': str(e), 'final_coords_zyx': None})

    # --- COMPREHENSIVE CSV SAVING LOGIC (MODIFIED TO INCLUDE MODALITY) ---
    print("\nPreprocessing complete. Generating comprehensive localization file for ALL successful scans...")
    results_df = pd.DataFrame(results)
    
    # --- CHANGE 1: MERGE WITH ORIGINAL DATAFRAME TO GET MODALITY ---
    # We use the df_train_unique which has the UID and Modality for every scan.
    results_df = pd.merge(results_df, df_train_unique[['SeriesInstanceUID', 'Modality']], on='SeriesInstanceUID', how='left')
    
    results_df.to_csv(CSV_LOG_PATH, index=False)
    print(f"Detailed log file saved to: {CSV_LOG_PATH}")

    all_successful_df = results_df[results_df['status'] == 'Success'].copy()
    location_cols = ['Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery', 'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery', 'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery', 'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation']
    
    # --- CHANGE 2: ADD 'Modality' TO THE LIST OF FINAL COLUMNS ---
    final_cols = ['SeriesInstanceUID', 'Modality', 'coord_z', 'coord_y', 'coord_x'] + location_cols + ['Aneurysm Present']

    positive_df = all_successful_df[(all_successful_df['final_coords_zyx'].notna()) & (all_successful_df['final_coords_zyx'].apply(lambda x: isinstance(x, list) and len(x) > 0))].copy()

    if not positive_df.empty:
        positive_df = positive_df.explode('final_coords_zyx')
        extracted_data = positive_df['final_coords_zyx'].apply(pd.Series)
        # --- CHANGE 3: CARRY 'Modality' THROUGH THE POSITIVE DATAFRAME ---
        positive_df = positive_df[['SeriesInstanceUID', 'Modality']].join(extracted_data)
        coords = pd.DataFrame(positive_df['final_coords_zyx'].tolist(), index=positive_df.index, columns=['coord_z', 'coord_y', 'coord_x'])
        positive_df = pd.concat([positive_df, coords], axis=1)
        positive_df['Aneurysm Present'] = 1
        location_dummies = pd.get_dummies(positive_df['location'])
        for col in location_cols:
            if col not in location_dummies.columns:
                location_dummies[col] = 0
        location_dummies = location_dummies[location_cols].astype(int)
        positive_final_df = pd.concat([positive_df, location_dummies], axis=1)
        positive_final_df = positive_final_df[final_cols]
    else:
        positive_final_df = pd.DataFrame(columns=final_cols)

    negative_df = all_successful_df[all_successful_df['final_coords_zyx'].isna()].copy()
    # --- CHANGE 4: CARRY 'Modality' THROUGH THE NEGATIVE DATAFRAME ---
    negative_final_df = negative_df[['SeriesInstanceUID', 'Modality']].copy()
    negative_final_df['Aneurysm Present'] = 0

    final_loc_df = pd.concat([positive_final_df, negative_final_df], ignore_index=True)
    final_loc_df[location_cols] = final_loc_df[location_cols].fillna(0).astype(int)
    final_loc_df = final_loc_df.sort_values(by='SeriesInstanceUID').reset_index(drop=True)
    # Ensure the final column order is correct
    final_loc_df = final_loc_df[final_cols]
    
    final_loc_df.drop_duplicates(inplace=True)
    final_loc_df.to_csv(NEW_LOCALIZATION_CSV_PATH, index=False)
    print(f"Final training manifest saved to: {NEW_LOCALIZATION_CSV_PATH}")
    
    status_counts = results_df['status'].value_counts()
    aneurysm_counts = final_loc_df['Aneurysm Present'].value_counts()
    modality_counts = final_loc_df['Modality'].value_counts()
    
    print("\n--- Summary ---")
    print("Processing Status Counts:")
    print(status_counts)
    print("\nModality Counts in Final Manifest:")
    print(modality_counts)
    print("\nAneurysm Presence in Final Manifest:")
    print(aneurysm_counts)
    print(f"A total of {final_loc_df.shape[0]} records were saved.")
    print("-----------------")