# Save this as save_mra_hdf5_fix.py
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
# --- MODIFIED: Import Manager from multiprocessing ---
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import ast
import h5py

from prep_mr import preprocess_mri_scan 
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)

# --- CONFIGURATION (Identical to before) ---
BASE_PATH = r'rsna-intracranial-aneurysm-detection\series'
OUTPUT_HDF5_PATH = os.path.join('processed_data_mra_v1', 'processed_data_mra_v1.hdf5')
CHUNK_SHAPE = (32, 32, 32) 
CSV_LOG_PATH = os.path.join('processed_data_mra_v1', 'preprocessing_log_mra.csv')
NEW_LOCALIZATION_CSV_PATH = os.path.join('processed_data_mra_v1', 'new_localization_mra.csv')
OUTPUT_DIR_FOR_CSVS = 'processed_data_mra_v1'

MAX_SCANS_TO_PROCESS = 60
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv'
NUM_PROCESSES = 8

import time # Add the time import for sleeping

def process_and_save_scan(args):
    """
    MODIFIED: This version uses a robust file-based lock to prevent write collisions.
    """
    # NOTE: We no longer need the 'lock' object in the arguments
    series_uid, base_path, hdf5_path, coords_list, modality = args
    
    # Read-only check can still happen without a lock
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if series_uid in f:
                return {'SeriesInstanceUID': series_uid, 'status': 'Skipped', 'shape_z_y_x': f[series_uid].shape, 'error': 'Dataset already exists in HDF5', 'final_coords_zyx': None}
    except (FileNotFoundError, OSError):
        pass
        
    try:
        # --- Heavy processing happens here, in parallel ---
        folder_path = os.path.join(base_path, series_uid)
        processed_array, final_spacing, final_coords_list = preprocess_mri_scan(
            folder_path,
            modality=modality,
            initial_coords_list=coords_list
        )
        if processed_array.size == 0:
             raise ValueError("Preprocessing returned an empty array.")
        processed_array_fp16 = processed_array.astype(np.float16)

        # --- BULLETPROOF FILE LOCK IMPLEMENTATION ---
        lock_file_path = hdf5_path + ".lock"
        
        while True: # Loop until we acquire the lock
            try:
                # O_CREAT | O_EXCL is an atomic "create if not exists" operation
                fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL)
                os.close(fd) # We just needed to create it, not keep it open
                break # Lock acquired, exit the loop
            except FileExistsError:
                time.sleep(0.5) # Lock is held by another process, wait and try again

        try:
            # --- This section is now guaranteed to only be run by one process at a time ---
            with h5py.File(hdf5_path, 'a') as f:
                f.create_dataset(
                    name=series_uid, data=processed_array_fp16, shape=processed_array_fp16.shape,
                    dtype='f2', chunks=CHUNK_SHAPE, compression="gzip"
                )
        finally:
            # --- CRITICAL: Always release the lock, even if the write fails ---
            os.remove(lock_file_path)
            
        return {'SeriesInstanceUID': series_uid, 'status': 'Success', 'shape_z_y_x': processed_array.shape, 'error': None, 'final_coords_zyx': final_coords_list}

    except Exception as e:
        return {'SeriesInstanceUID': series_uid, 'status': 'Failed', 'shape_z_y_x': None, 'error': str(e), 'final_coords_zyx': None}

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_FOR_CSVS, exist_ok=True)

    # --- NO MORE MANAGER OR LOCK OBJECT NEEDED ---
    
    try:
        df_train = pd.read_csv(r'rsna-intracranial-aneurysm-detection\train.csv')
        df_loc = pd.read_csv(ORIGINAL_LOCALIZATION_CSV)
        
        df_train_unique = df_train.drop_duplicates(subset=['SeriesInstanceUID'])
        df_mra = df_train_unique[df_train_unique['Modality'] != 'CTA'].copy()
        df_merged = pd.merge(df_mra, df_loc, on='SeriesInstanceUID', how='left')
        
        uids_to_process = df_merged['SeriesInstanceUID'].unique().tolist()
        print(f"Found {len(uids_to_process)} unique non-CTA SeriesInstanceUIDs.")

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
            
            # --- MODIFIED: The task tuple no longer contains the lock object ---
            tasks.append((
                uid,
                BASE_PATH,
                OUTPUT_HDF5_PATH,
                final_coords_arg,
                modality
            ))

    except FileNotFoundError as e:
        print(f"Error reading CSV files: {e}")
        exit()
    
    print(f"Starting preprocessing with {NUM_PROCESSES} parallel processes (with Robust File Lock and Hang Detection)...")
    
    results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        # We are keeping the superior hang-detection logic from the last fix
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
                print(f"The process is stuck or failed on SeriesInstanceUID: {series_uid}")
                print(f"Error: {e}")
                print(f"---------------------------------------------")
                results.append({'SeriesInstanceUID': series_uid, 'status': 'Failed_Hang_Or_Error', 'shape_z_y_x': None, 'error': str(e), 'final_coords_zyx': None})

    # --- The rest of the script is unchanged ---
    print("\nPreprocessing complete. Generating log and new localization files...")    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='SeriesInstanceUID').reset_index(drop=True)
    log_df = results_df[['SeriesInstanceUID', 'status', 'shape_z_y_x', 'error']]
    log_df.to_csv(CSV_LOG_PATH, index=False)
    print(f"Log file saved to: {CSV_LOG_PATH}")
    # ... (rest of your CSV saving logic)
    
    # --- CSV SAVING LOGIC (This part remains unchanged) ---
    print("\nGenerating comprehensive localization file for ALL successful scans...")

    # 1. Start with ALL scans that were successfully processed.
    all_successful_df = results_df[results_df['status'] == 'Success'].copy()

    # Define the final columns ahead of time for consistency
    location_cols = ['Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery', 'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery', 'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery', 'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation']
    final_cols = ['SeriesInstanceUID', 'coord_z', 'coord_y', 'coord_x'] + location_cols + ['Aneurysm Present']

    # 2. Isolate and process the POSITIVE scans (those with aneurysms)
    # This is the same logic as before, but applied to a subset
    positive_df = all_successful_df[
        (all_successful_df['final_coords_zyx'].notna()) &
        (all_successful_df['final_coords_zyx'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ].copy()

    if not positive_df.empty:
        positive_df = positive_df.explode('final_coords_zyx')
        extracted_data = positive_df['final_coords_zyx'].apply(pd.Series)
        positive_df = positive_df[['SeriesInstanceUID']].join(extracted_data)
        
        coords = pd.DataFrame(positive_df['final_coords_zyx'].tolist(), index=positive_df.index, columns=['coord_z', 'coord_y', 'coord_x'])
        positive_df = pd.concat([positive_df, coords], axis=1)
        
        positive_df['Aneurysm Present'] = 1 # Label these as 1
        
        location_dummies = pd.get_dummies(positive_df['location'])
        for col in location_cols:
            if col not in location_dummies.columns:
                location_dummies[col] = 0
        location_dummies = location_dummies[location_cols].astype(int)
                
        positive_final_df = pd.concat([positive_df, location_dummies], axis=1)
        positive_final_df = positive_final_df[final_cols]
    else:
        # If there are no positive scans, create an empty dataframe with the right columns
        positive_final_df = pd.DataFrame(columns=final_cols)

    # 3. Isolate and process the NEGATIVE scans (no aneurysms)
    negative_df = all_successful_df[all_successful_df['final_coords_zyx'].isna()].copy()
    
    if not negative_df.empty:
        negative_final_df = negative_df[['SeriesInstanceUID']].copy()
        negative_final_df['Aneurysm Present'] = 0 # Label these as 0
    else:
        negative_final_df = pd.DataFrame(columns=['SeriesInstanceUID', 'Aneurysm Present'])


    # 4. Combine the positive and negative dataframes
    final_loc_df = pd.concat([positive_final_df, negative_final_df], ignore_index=True)

    # 5. Clean up the final dataframe
    # The artery location columns will be NaN for negative scans, so fill them with 0
    final_loc_df[location_cols] = final_loc_df[location_cols].fillna(0).astype(int)
    
    # Sort by UID for a clean and organized file
    final_loc_df = final_loc_df.sort_values(by='SeriesInstanceUID').reset_index(drop=True)
    
    final_loc_df.drop_duplicates(inplace=True)
    final_loc_df.to_csv(NEW_LOCALIZATION_CSV_PATH, index=False)
    print(f"New COMPREHENSIVE localization file saved to: {NEW_LOCALIZATION_CSV_PATH}")
    
    status_counts = results_df['status'].value_counts()
    aneurysm_counts = final_loc_df['Aneurysm Present'].value_counts()
    
    print("\n--- Summary ---")
    print(status_counts)
    print("\nAneurysm Presence in Final CSV:")
    print(aneurysm_counts)
    print(f"A total of {final_loc_df.shape[0]} records were saved.")
    print("-----------------")