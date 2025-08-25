import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path
import concurrent.futures
import tqdm

# --- Configuration ---
# 1. Set the path to your folder containing .nii.gz files
INPUT_FOLDER = Path("processed_data") 

# 2. Set the path for the output images
OUTPUT_FOLDER = Path("output_slices")

# 3. Set the number of parallel threads to use
MAX_WORKERS = os.cpu_count() -2 # Use all available CPU cores, or 4 if detection fails
# --- End of Configuration ---

def normalize_slice(slice_data):
    """
    Normalizes a 2D numpy array to a 0-255 scale for image saving.
    """
    # Handle blank slices to avoid division by zero
    slice_min = slice_data.min()
    slice_max = slice_data.max()
    
    if slice_max == slice_min:
        return np.zeros_like(slice_data, dtype=np.uint8)
        
    # Apply min-max normalization to scale to 0-255
    normalized_slice = 255 * (slice_data - slice_min) / (slice_max - slice_min)
    return normalized_slice.astype(np.uint8)

def process_and_save_file(nii_path, output_dir):
    """
    Loads a .nii.gz file, extracts specific slices, normalizes them, 
    and saves them as JPG images.
    
    Args:
        nii_path (Path): Path to the input .nii.gz file.
        output_dir (Path): Directory to save the output JPG files.
    """
    try:
        # Get a clean base name for the output file
        base_name = nii_path.name.replace(".nii.gz", "")

        # Read the NIfTI image using SimpleITK
        image = sitk.ReadImage(str(nii_path))
        
        # Convert the SimpleITK image to a NumPy array (z, y, x)
        data_array = sitk.GetArrayFromImage(image)
        
        num_slices = data_array.shape[0]

        # --- Slice Index Calculation ---
        # Ensure there are enough slices to extract the required ones
        if num_slices < 10:
            # print(f"Skipping {nii_path.name}: Not enough slices ({num_slices}).")
            return f"Skipped: {nii_path.name} (only {num_slices} slices)"

        # Define the indices for the slices to be extracted
        indices = {
            " (1)": 4,  # 5th slice (0-indexed)
            " (2)": num_slices // 2,  # Middle slice
            " (3)": num_slices - 5, # 5th-to-last slice
        }
        
        # --- Process and Save Each Slice ---
        for suffix, index in indices.items():
            # Extract the 2D slice
            slice_2d = data_array[index, :, :]
            
            # Normalize the slice to be in the 0-255 range
            normalized_slice = normalize_slice(slice_2d)
            
            # Convert the NumPy array to a PIL Image
            img = Image.fromarray(normalized_slice)
            
            # Define the output path
            output_filename = f"{base_name}{suffix}.jpg"
            output_path = output_dir / output_filename
            
            # Save the image
            img.save(output_path)
            
        return f"Success: {nii_path.name}"

    except Exception as e:
        # Catch any errors during processing to prevent the script from crashing
        return f"Failed: {nii_path.name} with error: {e}"


def main():
    """
    Main function to set up folders and run the multithreaded processing.
    """
    print("--- NIfTI Slice Extractor ---")
    
    # Create the output directory if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Input folder:  '{INPUT_FOLDER.resolve()}'")
    print(f"Output folder: '{OUTPUT_FOLDER.resolve()}'")
    
    # Find all .nii.gz files in the input folder
    nii_files = list(INPUT_FOLDER.glob("*.nii.gz"))
    
    if not nii_files:
        print("No .nii.gz files found in the input folder. Exiting.")
        return

    print(f"Found {len(nii_files)} files to process using up to {MAX_WORKERS} threads.")

    # Use ThreadPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary of future tasks
        future_to_file = {
            executor.submit(process_and_save_file, file_path, OUTPUT_FOLDER): file_path 
            for file_path in nii_files
        }
        
        # Use tqdm to create a progress bar as tasks are completed
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_file), 
            total=len(nii_files),
            desc="Processing files"
        ):
            # You can optionally check the result of each task
            result = future.result()
            if "Failed" in result or "Skipped" in result:
                # To reduce console clutter, you can comment this out if you have many files
                # print(result)
                pass
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()