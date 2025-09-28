# Save this as save_mra.py
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast # To safely evaluate the string representation of the dictionary

# IMPORTANT: Make sure your MRA preprocessing script is in the same directory
# This assumes you are using a preprocess_mr.py that accepts and returns location data
from prep_mr import preprocess_mri_scan 
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)

# --- CONFIGURATION ---
BASE_PATH = r'rsna-intracranial-aneurysm-detection\series'
# --- It's highly recommended to use a new output directory for a fresh run ---
OUTPUT_DIR = r'processed_data_mra_v1' 
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'preprocessing_log_mra.csv')
NEW_LOCALIZATION_CSV_PATH = os.path.join(OUTPUT_DIR, 'new_localization_mra.csv')

# --- NEW CONFIGURATION OPTIONS ---
MAX_SCANS_TO_PROCESS = 60
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv'
NUM_PROCESSES = 4


def process_and_save_scan(args):
    """
    A wrapper function for a single scan. Takes a tuple of arguments.
    This function will be called by each parallel process.
    """
    # MODIFIED: Unpack the new `coords_list` and `modality` arguments
    series_uid, base_path, output_dir, coords_list, modality = args
    
    folder_path = os.path.join(base_path, series_uid)
    output_path = os.path.join(output_dir, f"{series_uid}.nii.gz")
    
    if os.path.exists(output_path):
        return {
            'SeriesInstanceUID': series_uid, 'status': 'Skipped',
            'shape_z_y_x': None, 'error': 'File already exists',
            'final_coords_zyx': None
        }
        
    try:
        # MODIFIED: Pass the list of coordinates and modality to the MRA processing function
        processed_array, final_spacing, final_coords_list = preprocess_mri_scan(
            folder_path,
            modality=modality,
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
        
        # ---- DATA PREPARATION LOGIC ----
        # Drop duplicates from train.csv to prevent the N x N join problem
        df_train_unique = df_train.drop_duplicates(subset=['SeriesInstanceUID'])

        # --- Filter for all modalities that are NOT CTA ---
        df_mra = df_train_unique[df_train_unique['Modality'] != 'CTA'].copy()
        
        # This merge is now safe because df_mra has unique SeriesInstanceUIDs
        df_merged = pd.merge(df_mra, df_loc, on='SeriesInstanceUID', how='left')
        
        uids_to_process = df_merged['SeriesInstanceUID'].unique().tolist()
        print(f"Found {len(uids_to_process)} unique non-CTA SeriesInstanceUIDs.")

        if MAX_SCANS_TO_PROCESS is not None:
            uids_to_process = uids_to_process[:MAX_SCANS_TO_PROCESS]
            print(f"--- LIMITING to processing {len(uids_to_process)} scans for this run. ---")
            
        # Group the merged dataframe by series UID to handle multiple aneurysms per series
        grouped = df_merged.groupby('SeriesInstanceUID')
        
        tasks = []
        for uid, group in tqdm(grouped, desc="Preparing tasks"):
             # We only care about processing UIDs that are in our target list
            if uid not in uids_to_process:
                continue
            
            # --- FIX 1: DEFINE MODALITY FOR EVERY GROUP ---
            # Get the modality from the first row of the group. This ensures it's always defined.
            modality = group['Modality'].iloc[0]
                
            coords_list_for_series = []
            # Check if there are any valid localizations for this group
            if not group['SOPInstanceUID'].isnull().all():
                for _, row in group.iterrows():
                    # Safely evaluate coordinates
                    try:
                        coords_dict = ast.literal_eval(row['coordinates'])
                        coords_list_for_series.append({
                            'sop_uid': row['SOPInstanceUID'],
                            'coords_xy': coords_dict,
                            'location': row['location']
                        })
                        # --- FIX 2: REMOVE MODALITY ASSIGNMENT FROM HERE ---
                        # modality = row['Modality'] # This is no longer needed
                    except (ValueError, SyntaxError, TypeError):
                        continue # Skip malformed or NaN coordinates
            
            # If the list is empty after checking all rows, pass None.
            # Otherwise, pass the populated list.
            final_coords_arg = coords_list_for_series if coords_list_for_series else None
            
            # This append call will now always work because 'modality' is defined above
            tasks.append((
                uid,
                BASE_PATH,
                OUTPUT_DIR,
                final_coords_arg,
                modality
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
    
    log_df = results_df[['SeriesInstanceUID', 'status', 'shape_z_y_x', 'error']]
    log_df.to_csv(CSV_LOG_PATH, index=False)
    print(f"Log file saved to: {CSV_LOG_PATH}")

    # --- CSV SAVING LOGIC (Identical to CTA script) ---
    
    # Filter for successful results that actually have coordinate data
    loc_df = results_df[
        (results_df['status'] == 'Success') & 
        (results_df['final_coords_zyx'].notna()) &
        (results_df['final_coords_zyx'].apply(lambda x: isinstance(x, list) and len(x) > 0))
    ].copy()

    if not loc_df.empty:
        # Explode the list of dictionaries into separate rows
        loc_df = loc_df.explode('final_coords_zyx')
        
        # Convert the column of dictionaries into separate columns ('final_coords_zyx', 'location')
        extracted_data = loc_df['final_coords_zyx'].apply(pd.Series)
        
        # Use a safe join to combine the SeriesInstanceUID with the new data
        final_loc_df = loc_df[['SeriesInstanceUID']].join(extracted_data)
        
        # Extract z, y, x coordinates from the tuple into separate columns
        coords = pd.DataFrame(final_loc_df['final_coords_zyx'].tolist(), index=final_loc_df.index, columns=['coord_z', 'coord_y', 'coord_x'])
        final_loc_df = pd.concat([final_loc_df, coords], axis=1)
        
        # One-Hot Encoding Logic
        final_loc_df['Aneurysm Present'] = 1
        location_cols = ['Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery', 'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery', 'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery', 'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation']
        
        location_dummies = pd.get_dummies(final_loc_df['location'])
        for col in location_cols:
            if col not in location_dummies.columns:
                location_dummies[col] = 0
        
        # Convert all dummy columns to integers (0 or 1)
        location_dummies = location_dummies[location_cols].astype(int)
                
        final_loc_df = pd.concat([final_loc_df, location_dummies], axis=1)
        
        # Select and reorder the final columns as requested
        final_cols = ['SeriesInstanceUID', 'coord_z', 'coord_y', 'coord_x'] + location_cols + ['Aneurysm Present']
        final_loc_df = final_loc_df[final_cols]
    else:
        # Create an empty dataframe with the correct columns if no aneurysms were processed
        final_cols = ['SeriesInstanceUID', 'coord_z', 'coord_y', 'coord_x'] + location_cols + ['Aneurysm Present']
        final_loc_df = pd.DataFrame(columns=final_cols)
    final_loc_df.drop_duplicates(inplace=True)
    final_loc_df.to_csv(NEW_LOCALIZATION_CSV_PATH, index=False)
    print(f"New localization file saved to: {NEW_LOCALIZATION_CSV_PATH}")
    
    status_counts = results_df['status'].value_counts()
    print("\n--- Summary ---")
    print(status_counts)
    print(f"{final_loc_df.shape[0]} aneurysm locations were successfully transformed.")
    print("-----------------")