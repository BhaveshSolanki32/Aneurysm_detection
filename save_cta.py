# Save this as run_preprocessing_v2.py
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
OUTPUT_DIR = r'processed_data_v2' # Use a new folder to avoid confusion
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'preprocessing_log.csv')
NEW_LOCALIZATION_CSV_PATH = os.path.join(OUTPUT_DIR, 'new_localization.csv')

# --- NEW CONFIGURATION OPTIONS ---
# Set to an integer (e.g., 50) to process only that many scans for a test run
# Set to None to process all scans
MAX_SCANS_TO_PROCESS = 50

# Path to your original localization CSV
# Example: r'path\to\your\localization.csv'
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv' 

# Use one less than your total cores to keep your system responsive
NUM_PROCESSES =8 #max(1, cpu_count() - 3)


def process_and_save_scan(args):
    """
    A wrapper function for a single scan. Takes a tuple of arguments.
    This function will be called by each parallel process.
    """
    series_uid, base_path, output_dir, sop_uid, coords_str = args
    
    folder_path = os.path.join(base_path, series_uid)
    output_path = os.path.join(output_dir, f"{series_uid}.nii.gz")
    
    # Prepare coordinates if they exist
    coords_dict = None
    if pd.notna(coords_str):
        try:
            # Safely convert the string "{'x': ...}" to a dictionary
            coords_dict = ast.literal_eval(coords_str)
        except (ValueError, SyntaxError):
            return {
                'SeriesInstanceUID': series_uid, 'status': 'Failed',
                'shape_z_y_x': None, 'error': 'Invalid coordinate format',
                'final_coords_zyx': None
            }
            
    if os.path.exists(output_path):
        # Even if file exists, we might need to recalculate coords.
        # For simplicity in this run, we skip if the image is already processed.
        # To re-calculate labels for existing files, you'd need to modify this logic.
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Skipped',
            'shape_z_y_x': None, 'error': 'File already exists',
            'final_coords_zyx': None
        }
        
    try:
        processed_array, final_spacing, final_coords = preprocess_cta_scan(
            folder_path, 
            initial_coords_xy=coords_dict, 
            sop_instance_uid=sop_uid
        )
        
        if processed_array.size == 0:
             raise ValueError("Preprocessing returned an empty array.")

        sitk_image = sitk.GetImageFromArray(processed_array)
        # SimpleITK spacing is (x, y, z), our target_spacing is (z, y, x)
        sitk_image.SetSpacing(final_spacing[::-1]) 
        sitk.WriteImage(sitk_image, output_path)
        
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Success',
            'shape_z_y_x': processed_array.shape, 'error': None,
            'final_coords_zyx': final_coords
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
        
        # Keep only CTA scans
        df_cta = df_train[df_train['Modality'] == 'CTA'].copy()
        
        # Merge the localization data. Use a left merge to keep all CTA scans.
        df_merged = pd.merge(df_cta, df_loc, on='SeriesInstanceUID', how='left')
        
        uids_to_process = df_merged['SeriesInstanceUID'].unique().tolist()
        print(f"Found {len(uids_to_process)} unique CTA SeriesInstanceUIDs.")

        # Apply the processing limit if specified
        if MAX_SCANS_TO_PROCESS is not None:
            uids_to_process = uids_to_process[:MAX_SCANS_TO_PROCESS]
            print(f"--- LIMITING to processing {len(uids_to_process)} scans for this run. ---")

        # Create a dictionary for quick lookup
        lookup_df = df_merged.set_index('SeriesInstanceUID')
        
        tasks = []
        for uid in uids_to_process:
            record = lookup_df.loc[[uid]].iloc[0] # Get the first row for this UID
            tasks.append((
                uid,
                BASE_PATH,
                OUTPUT_DIR,
                record['SOPInstanceUID'],
                record['coordinates']
            ))

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
    
    # --- Save the detailed processing log ---
    log_df = results_df[['SeriesInstanceUID', 'status', 'shape_z_y_x', 'error']]
    log_df.to_csv(CSV_LOG_PATH, index=False)
    print(f"Log file saved to: {CSV_LOG_PATH}")

    # --- Save the new localization data ---
    loc_df = results_df[results_df['final_coords_zyx'].notna()].copy()
    loc_df = loc_df[['SeriesInstanceUID', 'final_coords_zyx']]
    loc_df.to_csv(NEW_LOCALIZATION_CSV_PATH, index=False)
    print(f"New localization file saved to: {NEW_LOCALIZATION_CSV_PATH}")
    
    status_counts = results_df['status'].value_counts()
    print("\n--- Summary ---")
    print(status_counts)
    print(f"{loc_df.shape[0]} aneurysm locations were successfully transformed.")
    print("-----------------")