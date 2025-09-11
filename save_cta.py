# Save this as save_cta.py
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast # To safely evaluate the string representation of the dictionary

# IMPORTANT: Make sure your preprocessing script is in the same directory
from preprocess_ct import preprocess_cta_scan
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(5) 

# --- CONFIGURATION ---
BASE_PATH = r'rsna-intracranial-aneurysm-detection\series'
# --- It's highly recommended to use a new output directory for a fresh run ---
OUTPUT_DIR = r'processed_data_v3' 
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'preprocessing_log.csv')
NEW_LOCALIZATION_CSV_PATH = os.path.join(OUTPUT_DIR, 'new_localization.csv')

# --- NEW CONFIGURATION OPTIONS ---
MAX_SCANS_TO_PROCESS = 50
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv' 
NUM_PROCESSES = 3


def process_and_save_scan(args):
    """
    A wrapper function for a single scan. Takes a tuple of arguments.
    This function will be called by each parallel process.
    """
    # MODIFIED: Unpack the new `coords_list` argument
    series_uid, base_path, output_dir, coords_list = args
    
    folder_path = os.path.join(base_path, series_uid)
    output_path = os.path.join(output_dir, f"{series_uid}.nii.gz")
    
    if os.path.exists(output_path):
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Skipped',
            'shape_z_y_x': None, 'error': 'File already exists',
            'final_coords_zyx': None
        }
        
    try:
        # MODIFIED: Pass the list of coordinates to the main function
        processed_array, final_spacing, final_coords_list = preprocess_cta_scan(
            folder_path, 
            initial_coords_list=coords_list
        )
        
        if processed_array.size == 0:
             raise ValueError("Preprocessing returned an empty array.")

        sitk_image = sitk.GetImageFromArray(processed_array)
        sitk_image.SetSpacing(final_spacing[::-1]) 
        sitk.WriteImage(sitk_image, output_path)
        
        # MODIFIED: Return the list of final coordinates
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Success',
            'shape_z_y_x': processed_array.shape, 'error': None,
            'final_coords_zyx': final_coords_list
        }

    except Exception as e:
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Failed',
            'shape_z_y_x': None, 'error': str(e),
            'final_coords_zyx': None
        }

# This guard is ESSENTIAL for multiprocessing to work correctly
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df_train = pd.read_csv(r'rsna-intracranial-aneurysm-detection\train.csv')
        df_loc = pd.read_csv(ORIGINAL_LOCALIZATION_CSV)
        
        df_cta = df_train[df_train['Modality'] == 'CTA'].copy()
        
        # Merge is correct, no changes needed here.
        df_merged = pd.merge(df_cta, df_loc, on='SeriesInstanceUID', how='left')
        
        uids_to_process = df_merged['SeriesInstanceUID'].unique().tolist()
        print(f"Found {len(uids_to_process)} unique CTA SeriesInstanceUIDs.")

        if MAX_SCANS_TO_PROCESS is not None:
            uids_to_process = uids_to_process[:MAX_SCANS_TO_PROCESS]
            print(f"--- LIMITING to processing {len(uids_to_process)} scans for this run. ---")
            
        # --- MODIFICATION START: Correctly prepare tasks for all aneurysms ---
        
        # Group the merged dataframe by series UID to handle multiple aneurysms per series
        grouped = df_merged.groupby('SeriesInstanceUID')
        
        tasks = []
        for uid, group in tqdm(grouped, desc="Preparing tasks"):
             # We only care about processing UIDs that are in our target list
            if uid not in uids_to_process:
                continue
                
            coords_list_for_series = []
            # Check if there are any valid localizations for this group
            if not group['SOPInstanceUID'].isnull().all():
                for _, row in group.iterrows():
                    # Safely evaluate coordinates
                    try:
                        coords_dict = ast.literal_eval(row['coordinates'])
                        coords_list_for_series.append({
                            'sop_uid': row['SOPInstanceUID'],
                            'coords_xy': coords_dict
                        })
                    except (ValueError, SyntaxError, TypeError):
                        continue # Skip malformed or NaN coordinates
            
            # If the list is empty after checking all rows, pass None.
            # Otherwise, pass the populated list.
            final_coords_arg = coords_list_for_series if coords_list_for_series else None
            
            tasks.append((
                uid,
                BASE_PATH,
                OUTPUT_DIR,
                final_coords_arg # This is now a list of dicts, or None
            ))
        # --- MODIFICATION END ---

    except FileNotFoundError as e:
        print(f"Error reading CSV files: {e}")
        exit()
    
    print(f"Starting preprocessing with {NUM_PROCESSES} parallel processes...")

    results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(process_and_save_scan, tasks), total=len(tasks)):
            results.append(result)

    print("\nPreprocessing complete. Generating log and new localization files...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='SeriesInstanceUID').reset_index(drop=True)
    
    log_df = results_df[['SeriesInstanceUID', 'status', 'shape_z_y_x', 'error']]
    log_df.to_csv(CSV_LOG_PATH, index=False)
    print(f"Log file saved to: {CSV_LOG_PATH}")

    # --- MODIFICATION START: Correctly save the new localization data ---
    
    # Filter for successful results that actually have coordinate data
    loc_df = results_df[
        (results_df['status'] == 'Success') & 
        (results_df['final_coords_zyx'].notna()) &
        (results_df['final_coords_zyx'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ].copy()

    # The 'final_coords_zyx' column now contains lists of tuples.
    # We use pandas' 'explode' to create a new row for each item in the list.
    if not loc_df.empty:
        loc_df = loc_df.explode('final_coords_zyx')
    
    final_loc_df = loc_df[['SeriesInstanceUID', 'final_coords_zyx']]
    final_loc_df.to_csv(NEW_LOCALIZATION_CSV_PATH, index=False)
    print(f"New localization file saved to: {NEW_LOCALIZATION_CSV_PATH}")
    
    # --- MODIFICATION END ---
    
    status_counts = results_df['status'].value_counts()
    print("\n--- Summary ---")
    print(status_counts)
    print(f"{final_loc_df.shape[0]} aneurysm locations were successfully transformed.")
    print("-----------------")
