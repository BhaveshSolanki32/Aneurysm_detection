# Save this as fix_multiple_localizations.py
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import ast
import gc
import traceback

# IMPORTANT: We now need the ORIGINAL align function to replicate the randomness
from preprocess_ct import (
    get_physical_point_from_dicom,
    load_and_reorient_dicom,
    find_neck_cutoff,
    crop_to_body,
    align_to_midsagittal_plane  # <-- The original, non-deterministic function
)

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)

# --- CONFIGURATION (NO CHANGES) ---
BASE_DICOM_PATH = r'rsna-intracranial-aneurysm-detection\series'
PROCESSED_NIFTI_DIR = r'processed_data_v2'
ORIGINAL_LOCALIZATION_CSV = r'rsna-intracranial-aneurysm-detection\train_localizers.csv'
INCORRECT_LOCALIZATION_CSV = os.path.join(PROCESSED_NIFTI_DIR, 'new_localization_1.csv')
FINAL_CORRECTED_CSV_PATH = os.path.join(PROCESSED_NIFTI_DIR, 'final_corrected_localization.csv')
NUM_PROCESSES = max(1, cpu_count() // 4)

def get_alignment_transform_for_series(dicom_folder_path: str) -> sitk.Transform:
    """
    Re-runs the lightweight preprocessing to get the alignment transform.
    Crucially, it now uses the ORIGINAL align_to_midsagittal_plane function
    to best approximate the transform from the initial run.
    """
    try:
        reoriented_itk = load_and_reorient_dicom(dicom_folder_path)
        clipped_itk = sitk.Clamp(reoriented_itk, sitk.sitkFloat32, -1024, 1000)
        del reoriented_itk
        full_scan_np_view = sitk.GetArrayViewFromImage(clipped_itk)
        neck_cutoff_z = find_neck_cutoff(full_scan_np_view, body_threshold_hu=-200)
        original_size = clipped_itk.GetSize()
        index = [0, 0, int(neck_cutoff_z)]
        size = [original_size[0], original_size[1], original_size[2] - int(neck_cutoff_z)]
        if size[2] < 5: raise RuntimeError(f"Neck cropping resulted in {size[2]} slices.")
        head_and_shoulders_itk = sitk.RegionOfInterest(clipped_itk, size=size, index=index)
        del clipped_itk
        head_itk = crop_to_body(head_and_shoulders_itk, air_threshold_hu=-500)
        del head_and_shoulders_itk
        
        # Use the original function to get the transform
        _, alignment_transform = align_to_midsagittal_plane(head_itk)
        del head_itk
        gc.collect()
        return alignment_transform
    except Exception as e:
        error_info = ''.join(traceback.format_exception(None, e, e.__traceback__))
        print(f"Error getting transform for {os.path.basename(dicom_folder_path)}:\n{error_info}")
        return None

def process_series_to_fix(args):
    series_uid, group_df, base_dicom_path, processed_nifti_dir = args
    dicom_folder_path = os.path.join(base_dicom_path, series_uid)
    nifti_file_path = os.path.join(processed_nifti_dir, f"{series_uid}.nii.gz")

    if not os.path.exists(nifti_file_path):
        return []

    alignment_transform = get_alignment_transform_for_series(dicom_folder_path)
    if alignment_transform is None:
        return []

    final_itk_image = sitk.ReadImage(nifti_file_path)
    results = []
    for _, row in group_df.iterrows():
        sop_uid = row['SOPInstanceUID']
        coords_str = row['coordinates']
        try:
            coords_obj = ast.literal_eval(coords_str)
            if isinstance(coords_obj, list):
                coords_xy = {'x': coords_obj[0], 'y': coords_obj[1]}
            elif isinstance(coords_obj, dict):
                coords_xy = coords_obj
            else:
                raise TypeError(f"Unsupported coordinate format: {type(coords_obj)}")

            initial_physical_point = get_physical_point_from_dicom(dicom_folder_path, sop_uid, coords_xy)
            transformed_physical_point = alignment_transform.TransformPoint(initial_physical_point)
            final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(transformed_physical_point)
            final_voxel_coords_zyx = final_voxel_coords_xyz[::-1]
            
            # Check for out-of-bounds coordinates, which indicates a large mismatch
            img_size = final_itk_image.GetSize()
            if not (0 <= final_voxel_coords_xyz[0] < img_size[0] and \
                    0 <= final_voxel_coords_xyz[1] < img_size[1] and \
                    0 <= final_voxel_coords_xyz[2] < img_size[2]):
                print(f"Warning: Out-of-bounds coordinate {final_voxel_coords_zyx} generated for {series_uid}. Skipping this point.")
                continue

            results.append({
                'SeriesInstanceUID': series_uid,
                'final_coords_zyx': str(list(final_voxel_coords_zyx))
            })
        except Exception as e:
            error_info = ''.join(traceback.format_exception(None, e, e.__traceback__))
            print(f"Could not process coordinate for {series_uid} / {sop_uid}:\n{error_info}")
            continue
    return results

if __name__ == '__main__':
    print("Starting localization correction process...")
    try:
        df_loc_full = pd.read_csv(ORIGINAL_LOCALIZATION_CSV)
        df_incorrect = pd.read_csv(INCORRECT_LOCALIZATION_CSV)
    except FileNotFoundError as e:
        print(f"ERROR: A required CSV file was not found: {e}")
        exit()

    # 1. Identify which Series UIDs have more than one aneurysm in the *original* data
    uid_counts = df_loc_full['SeriesInstanceUID'].value_counts()
    uids_to_fix = uid_counts[uid_counts == 1].index.tolist()[0]
    
    if not uids_to_fix:
        print("No series with multiple aneurysms found in the original data. Nothing to fix.")
        exit()
        
    print(f"Identified {len(uids_to_fix)} series with multiple aneurysms to fix.")

    # 2. Separate the good data from the bad data in your generated file
    # The 'good' data are all rows for scans that are NOT in our list to fix.
    df_good_singles = df_incorrect[~df_incorrect['SeriesInstanceUID'].isin(uids_to_fix)].copy()
    print(f"Keeping {len(df_good_singles)} correct locations from single-aneurysm scans.")

    # 3. Create tasks for only the series that need fixing
    df_to_fix = df_loc_full[df_loc_full['SeriesInstanceUID'].isin(uids_to_fix)].copy()
    grouped = df_to_fix.groupby('SeriesInstanceUID')
    tasks = [(uid, group, BASE_DICOM_PATH, PROCESSED_NIFTI_DIR) for uid, group in grouped]

    # 4. Run the processing in parallel
    print(f"Re-processing {len(tasks)} series to get all coordinates...")
    all_fixed_results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result_list in tqdm(pool.imap_unordered(process_series_to_fix, tasks), total=len(tasks)):
            all_fixed_results.extend(result_list)

    df_fixed_multis = pd.DataFrame(all_fixed_results)
    if not df_fixed_multis.empty:
        print(f"Successfully re-calculated {len(df_fixed_multis)} locations for multi-aneurysm scans.")
    else:
        print("Warning: No multi-aneurysm locations were successfully re-calculated.")

    # 5. Combine the trusted single-aneurysm data with the newly fixed multi-aneurysm data
    df_final_corrected = pd.concat([df_good_singles, df_fixed_multis], ignore_index=True)
    df_final_corrected = df_final_corrected.sort_values(by='SeriesInstanceUID').reset_index(drop=True)

    # 6. Save the final file
    if not df_final_corrected.empty:
        df_fixed_multis.to_csv(FINAL_CORRECTED_CSV_PATH, index=False)
        print("\n--- FIX COMPLETE ---")
        print(f"Final corrected localization file saved to: {FINAL_CORRECTED_CSV_PATH}")
        print(f"Total number of aneurysm locations in final file: {len(df_final_corrected)}")
    else:
        print("\n--- FIX FAILED ---")
        print("No data was processed or combined, so no output file was saved.")
    
    print("--------------------")