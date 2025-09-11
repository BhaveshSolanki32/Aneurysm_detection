# Save this as fix_multiple_localizations.py
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast
import gc

# IMPORTANT: Make sure your original preprocessing script is accessible
# We need its helper functions to replicate the transformation logic.
from preprocess_ct import (
    get_physical_point_from_dicom,
    load_and_reorient_dicom,
    find_neck_cutoff,
    crop_to_body,
    align_to_midsagittal_plane
)

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2) # Keep it low per-process

# --- CONFIGURATION ---
# Path to the original DICOM series
BASE_DICOM_PATH = r'rsna-intracranial-aneurysm-detection\series'
# Path to the folder where you saved the processed .nii.gz files
PROCESSED_NIFTI_DIR = r'processed_data_v2'
# Path to the original, full localization CSV from the competition data
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv'
# Path to the localization file your script *incorrectly* generated (we need it for the single-aneurysm scans)
INCORRECT_LOCALIZATION_CSV = os.path.join(PROCESSED_NIFTI_DIR, 'new_localization.csv')

# --- OUTPUT FILE ---
# The final, corrected localization file will be saved here.
FINAL_CORRECTED_CSV_PATH = os.path.join(PROCESSED_NIFTI_DIR, 'final_corrected_localization.csv')

# Use fewer processes as each might still be memory/IO intensive
NUM_PROCESSES = 3

def get_alignment_transform_for_series(dicom_folder_path: str) -> sitk.Transform:
    """
    This is a "lightweight" version of your main preprocessing function.
    It performs only the steps necessary to calculate the alignment transform.
    It does NOT perform normalization, resampling, or save any images.
    """
    try:
        # Steps 1 & 2: Load and perform initial cropping
        reoriented_itk = load_and_reorient_dicom(dicom_folder_path)
        clipped_itk = sitk.Clamp(reoriented_itk, sitk.sitkFloat32, -1024, 1000)
        del reoriented_itk

        # Step 3: Neck Cropping
        full_scan_np_view = sitk.GetArrayViewFromImage(clipped_itk)
        neck_cutoff_z = find_neck_cutoff(full_scan_np_view, body_threshold_hu=-200)
        
        original_size = clipped_itk.GetSize()
        index = [0, 0, int(neck_cutoff_z)]
        size = [original_size[0], original_size[1], original_size[2] - int(neck_cutoff_z)]

        if size[2] < 5:
            raise RuntimeError(f"Neck cropping resulted in {size[2]} slices.")
            
        head_and_shoulders_itk = sitk.RegionOfInterest(clipped_itk, size=size, index=index)
        del clipped_itk

        # Step 4: Body Cropping
        head_itk = crop_to_body(head_and_shoulders_itk, air_threshold_hu=-500)
        del head_and_shoulders_itk

        # Step 5: Alignment (The crucial step)
        _, alignment_transform = align_to_midsagittal_plane(head_itk)
        del head_itk
        gc.collect()
        
        return alignment_transform

    except Exception as e:
        print(f"Error getting transform for {os.path.basename(dicom_folder_path)}: {e}")
        return None

def process_series_with_multiple_aneurysms(args):
    """
    Worker function for a single Series UID that has multiple aneurysms.
    """
    series_uid, group_df, base_dicom_path, processed_nifti_dir = args
    dicom_folder_path = os.path.join(base_dicom_path, series_uid)
    nifti_file_path = os.path.join(processed_nifti_dir, f"{series_uid}.nii.gz")

    # This is the most time-consuming step of the fix
    alignment_transform = get_alignment_transform_for_series(dicom_folder_path)

    if alignment_transform is None:
        return [] # Return empty list on failure

    if not os.path.exists(nifti_file_path):
        print(f"Warning: Processed NIfTI file not found for {series_uid}, skipping.")
        return []

    # Load the final, saved image. Its metadata (origin, spacing, direction)
    # is our reference for the final coordinate system.
    final_itk_image = sitk.ReadImage(nifti_file_path)

    results = []
    for _, row in group_df.iterrows():
        sop_uid = row['SOPInstanceUID']
        coords_str = row['coordinates']
        
        try:
            coords_xy = ast.literal_eval(coords_str)

            # 1. Get original physical point from the specific DICOM slice
            initial_physical_point = get_physical_point_from_dicom(
                dicom_folder_path, sop_uid, coords_xy
            )

            # 2. Apply the recovered alignment transform
            transformed_physical_point = alignment_transform.TransformPoint(initial_physical_point)

            # 3. Convert this new physical point to a voxel index in the *final* resampled image
            final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(transformed_physical_point)
            
            # 4. Convert from ITK's (x, y, z) to NumPy's (z, y, x)
            final_voxel_coords_zyx = final_voxel_coords_xyz[::-1]
            
            results.append({
                'SeriesInstanceUID': series_uid,
                'final_coords_zyx': final_voxel_coords_zyx
            })

        except Exception as e:
            print(f"Could not process coordinate for {series_uid} / {sop_uid}: {e}")
            continue
            
    return results


if __name__ == '__main__':
    print("Starting localization correction process...")
    
    # 1. Load original, full localizer file
    try:
        df_loc_full = pd.read_csv(ORIGINAL_LOCALIZATION_CSV)
    except FileNotFoundError:
        print(f"ERROR: Original localization file not found at {ORIGINAL_LOCALIZATION_CSV}")
        exit()

    # 2. Identify which Series UIDs have more than one aneurysm
    uid_counts = df_loc_full['SeriesInstanceUID'].value_counts()
    uids_to_fix = uid_counts[uid_counts > 1].index.tolist()
    
    if not uids_to_fix:
        print("No series with multiple aneurysms found. Your localization file might already be correct.")
        exit()
        
    print(f"Found {len(uids_to_fix)} series with multiple aneurysms that need to be fixed.")

    # 3. Create a DataFrame containing only the rows for the series that need fixing
    df_to_fix = df_loc_full[df_loc_full['SeriesInstanceUID'].isin(uids_to_fix)].copy()
    
    # 4. Group by UID to create tasks for the multiprocessing pool
    grouped = df_to_fix.groupby('SeriesInstanceUID')
    tasks = [(uid, group, BASE_DICOM_PATH, PROCESSED_NIFTI_DIR) for uid, group in grouped]

    # 5. Run the processing in parallel
    print(f"Processing {len(tasks)} series with {NUM_PROCESSES} parallel processes...")
    
    all_fixed_results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result_list in tqdm(pool.imap_unordered(process_series_with_multiple_aneurysms, tasks), total=len(tasks)):
            all_fixed_results.extend(result_list)

    df_fixed = pd.DataFrame(all_fixed_results)
    print(f"Successfully re-calculated {len(df_fixed)} aneurysm locations.")

    # 6. Combine the fixed results with the already correct results
    print("Combining fixed results with original single-aneurysm results...")
    try:
        df_incorrect = pd.read_csv(INCORRECT_LOCALIZATION_CSV)
        # Keep only the rows from the original output that we did NOT just fix
        df_singles = df_incorrect[~df_incorrect['SeriesInstanceUID'].isin(uids_to_fix)].copy()
        
        # Combine the two dataframes
        df_final_corrected = pd.concat([df_singles, df_fixed], ignore_index=True)
        df_final_corrected = df_final_corrected.sort_values(by='SeriesInstanceUID').reset_index(drop=True)
        
    except FileNotFoundError:
        print(f"Warning: Your original output file '{INCORRECT_LOCALIZATION_CSV}' was not found.")
        print("The new file will ONLY contain data for the series with multiple aneurysms.")
        df_final_corrected = df_fixed

    # 7. Save the final, fully corrected localization file
    df_final_corrected.to_csv(FINAL_CORRECTED_CSV_PATH, index=False)

    print("\n--- FIX COMPLETE ---")
    print(f"Final corrected localization file saved to: {FINAL_CORRECTED_CSV_PATH}")
    print(f"Total number of aneurysm locations in final file: {len(df_final_corrected)}")
    print("--------------------")
