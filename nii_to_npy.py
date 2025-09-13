import SimpleITK as sitk
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

nifti_dir = 'processed_data_v2'
npy_dir = 'processed_data_npy'
os.makedirs(npy_dir, exist_ok=True)

def convert_file(filename):
    """Convert one NIfTI file to NumPy and save it."""
    if not filename.endswith(".nii.gz"):
        return f"Skipped: {filename}"

    try:
        itk_img = sitk.ReadImage(os.path.join(nifti_dir, filename))
        np_array = sitk.GetArrayFromImage(itk_img)
        out_path = os.path.join(npy_dir, filename.replace('.nii.gz', '.npy'))
        np.save(out_path, np_array)
        return f"Done: {filename}"
    except Exception as e:
        return f"Error {filename}: {e}"

# --- Control number of threads (adjust based on CPU cores & disk speed)
max_workers = 5

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(convert_file, f) for f in os.listdir(nifti_dir)]
    for future in as_completed(futures):
        print(future.result())
