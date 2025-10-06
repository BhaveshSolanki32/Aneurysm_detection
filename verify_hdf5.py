# Save this as verify_hdf5.py
import os
import numpy as np
import h5py # --- MODIFIED: Use h5py instead of SimpleITK ---
from PIL import Image
from pathlib import Path
import concurrent.futures
import tqdm

# --- Configuration ---
# 1. Set the path to your SINGLE HDF5 file
HDF5_INPUT_PATH = Path(r"aneurysm_dataset_manifests_hdf5_ho\processed_scans.hdf5") 

# 2. Set the path for the output images
OUTPUT_FOLDER = Path("output_slices_mr_from_hdf5")

# 3. Set the number of parallel threads to use
# Reading from a single file can be I/O bound, so using fewer workers can sometimes be faster.
# Start with a moderate number and see what works best for your system.
MAX_WORKERS = os.cpu_count() -5
# --- End of Configuration ---

def normalize_slice(slice_data):
    """
    Normalizes a 2D numpy array to a 0-255 scale for image saving.
    (This function is perfect and requires no changes)
    """
    slice_min = slice_data.min()
    slice_max = slice_data.max()
    
    if slice_max == slice_min:
        return np.zeros_like(slice_data, dtype=np.uint8)
        
    normalized_slice = 255 * (slice_data - slice_min) / (slice_max - slice_min)
    return normalized_slice.astype(np.uint8)

def process_and_save_scan(hdf5_path, series_uid, output_dir):
    """
    MODIFIED: Reads a specific dataset (scan) from an HDF5 file, 
    extracts slices, and saves them as JPG images.
    
    Args:
        hdf5_path (Path): Path to the input .hdf5 file.
        series_uid (str): The SeriesInstanceUID of the scan to process (which is a key in the HDF5 file).
        output_dir (Path): Directory to save the output JPG files.
    """
    try:
        # --- MODIFIED: Data Loading from HDF5 ---
        with h5py.File(hdf5_path, 'r') as f:
            # The [()] syntax loads the entire array for this dataset into memory
            data_array = f[series_uid][()]
        # --- End of Modification ---
        
        num_slices = data_array.shape[0]

        if num_slices < 10:
            return f"Skipped: {series_uid} (only {num_slices} slices)"

        indices = {
            "_start": 4,              # 5th slice
            "_middle": num_slices // 2, # Middle slice
            "_end": num_slices - 5,   # 5th-to-last slice
        }
        
        for suffix, index in indices.items():
            slice_2d = data_array[index, :, :]
            normalized_slice = normalize_slice(slice_2d)
            img = Image.fromarray(normalized_slice)
            
            output_filename = f"{series_uid}{suffix}.jpg"
            output_path = output_dir / output_filename
            
            img.save(output_path)
            
        return f"Success: {series_uid}"

    except Exception as e:
        return f"Failed: {series_uid} with error: {e}"


def main():
    """
    Main function to set up folders and run the multithreaded processing on an HDF5 file.
    """
    print("--- HDF5 Slice Extractor ---")
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if not HDF5_INPUT_PATH.exists():
        print(f"Error: Input HDF5 file not found at '{HDF5_INPUT_PATH.resolve()}'")
        return

    print(f"Input HDF5 file: '{HDF5_INPUT_PATH.resolve()}'")
    print(f"Output folder:   '{OUTPUT_FOLDER.resolve()}'")
    
    # --- MODIFIED: Get list of scans from HDF5 keys, not from file names ---
    with h5py.File(HDF5_INPUT_PATH, 'r') as f:
        scan_uids = list(f.keys())
    # --- End of Modification ---
    
    if not scan_uids:
        print("No scans (datasets) found in the HDF5 file. Exiting.")
        return

    print(f"Found {len(scan_uids)} scans to process using up to {MAX_WORKERS} threads.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # --- MODIFIED: Submit tasks with hdf5_path and series_uid ---
        future_to_uid = {
            executor.submit(process_and_save_scan, HDF5_INPUT_PATH, uid, OUTPUT_FOLDER): uid 
            for uid in scan_uids
        }
        
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_uid), 
            total=len(scan_uids),
            desc="Processing scans"
        ):
            result = future.result()
            if "Failed" in result or "Skipped" in result:
                # To see errors, uncomment the line below
                # print(result)
                pass
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()