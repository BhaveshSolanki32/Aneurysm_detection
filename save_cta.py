# Save this as run_preprocessing_v2.py
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# IMPORTANT: Make sure your preprocessing script is in the same directory
from preprocess_ct import preprocess_cta_scan

# --- CONFIGURATION ---
BASE_PATH = r'rsna-intracranial-aneurysm-detection\series'
OUTPUT_DIR = r'processed_data'
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'preprocessing_log.csv')

# Use one less than your total cores to keep your system responsive
NUM_PROCESSES = max(1, cpu_count() - 2)

def process_and_save_scan(args):
    """
    A wrapper function for a single scan. Takes a tuple of arguments.
    This function will be called by each parallel process.
    """
    series_uid, base_path, output_dir = args
    
    folder_path = os.path.join(base_path, series_uid)
    output_path = os.path.join(output_dir, f"{series_uid}.nii.gz")
    
    if os.path.exists(output_path):
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Skipped',
            'shape_z_y_x': None, 'error': 'File already exists'
        }
        
    try:
        processed_array, final_spacing_zyx = preprocess_cta_scan(folder_path)
        
        if processed_array.size == 0:
             raise ValueError("Preprocessing returned an empty array.")

        sitk_image = sitk.GetImageFromArray(processed_array)
        sitk_image.SetSpacing(final_spacing_zyx[::-1]) # (x, y, z) order
        sitk.WriteImage(sitk_image, output_path)
        
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Success',
            'shape_z_y_x': processed_array.shape, 'error': None
        }

    except Exception as e:
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Failed',
            'shape_z_y_x': None, 'error': str(e)
        }

# This guard is ESSENTIAL for multiprocessing to work correctly
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df_orig_train = pd.read_csv(r'rsna-intracranial-aneurysm-detection\train.csv')
        uids_to_process = df_orig_train[df_orig_train['Modality'] == 'CTA']['SeriesInstanceUID'].tolist()
        print(f"Found {len(uids_to_process)} CTA SeriesInstanceUIDs to process.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()
    
    tasks = [(uid, BASE_PATH, OUTPUT_DIR) for uid in uids_to_process]

    print(f"Starting preprocessing with {NUM_PROCESSES} parallel processes...")

    results = []
    # This is the robust way to run multiprocessing with a progress bar
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(process_and_save_scan, tasks), total=len(tasks)):
            results.append(result)

    print("\nPreprocessing complete. Generating log file...")
    
    results_df = pd.DataFrame(results)
    
    # Sort by UID to have a consistent log file order
    results_df = results_df.sort_values(by='SeriesInstanceUID').reset_index(drop=True)
    results_df = results_df[['SeriesInstanceUID', 'status', 'shape_z_y_x', 'error']]
    results_df.to_csv(CSV_LOG_PATH, index=False)

    print(f"Log file saved to: {CSV_LOG_PATH}")
    
    status_counts = results_df['status'].value_counts()
    print("\n--- Summary ---")
    print(status_counts)
    print("-----------------")