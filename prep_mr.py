# Save this as preprocess_mri.py


# --------------------------------------------------------------------------
# 1. ALL NECESSARY IMPORTS
# --------------------------------------------------------------------------
import SimpleITK as sitk
import numpy as np
import os
import pydicom
import collections
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import subprocess
import tempfile
import torch
from scipy.signal import find_peaks


# --------------------------------------------------------------------------
# 2. VISUALIZATION AND HELPER FUNCTIONS (COMPLETE AND VERIFIED)
# --------------------------------------------------------------------------


def visualize_location_in_3d(
    image_np: np.ndarray,
    voxel_coords_zyx: Tuple[int, int, int],
    title: str = "3D Orthogonal Views"
):
    """
    Displays the three orthogonal planes (axial, coronal, sagittal) of a 3D image
    volume, centered on the given ZYX voxel coordinates.
    """
    z, y, x = voxel_coords_zyx
    
    # Input Validation
    if not (0 <= z < image_np.shape[0] and 0 <= y < image_np.shape[1] and 0 <= x < image_np.shape[2]):
        print(f"Error: Voxel coordinates {voxel_coords_zyx} are out of bounds for image shape {image_np.shape}")
        z, y, x = image_np.shape[0] // 2, image_np.shape[1] // 2, image_np.shape[2] // 2
        title += f"\nCOORDS OUT OF BOUNDS - Showing center instead"
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    # Axial View
    axes[0].imshow(image_np[z, :, :], cmap='gray', origin='lower')
    axes[0].axhline(y, color='lime', linewidth=0.8); axes[0].axvline(x, color='lime', linewidth=0.8)
    axes[0].scatter(x, y, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[0].set_title(f"Axial (Z = {z})"); axes[0].set_xlabel("X-axis"); axes[0].set_ylabel("Y-axis")


    # Coronal View
    axes[1].imshow(image_np[:, y, :], cmap='gray', origin='lower', aspect='auto')
    axes[1].axhline(z, color='lime', linewidth=0.8); axes[1].axvline(x, color='lime', linewidth=0.8)
    axes[1].scatter(x, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[1].set_title(f"Coronal (Y = {y})"); axes[1].set_xlabel("X-axis"); axes[1].set_ylabel("Z-axis")


    # Sagittal View
    axes[2].imshow(image_np[:, :, x], cmap='gray', origin='lower', aspect='auto')
    axes[2].axhline(z, color='lime', linewidth=0.8); axes[2].axvline(y, color='lime', linewidth=0.8)
    axes[2].scatter(y, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[2].set_title(f"Sagittal (X = {x})"); axes[2].set_xlabel("Y-axis"); axes[2].set_ylabel("Z-axis")


    plt.show()


def get_physical_point_from_dicom(
    dicom_folder_path: str, sop_uid: str, coords_xy: dict
) -> Tuple[float, float, float]:
    """
    Finds a DICOM slice by its SOPInstanceUID and calculates the 3D physical coordinates.
    This mirrors the logic of the working CTA script.
    """
    for filename in os.listdir(dicom_folder_path):
        filepath = os.path.join(dicom_folder_path, filename)
        if not os.path.isfile(filepath): continue
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
            if hasattr(dcm, 'SOPInstanceUID') and dcm.SOPInstanceUID == sop_uid:
                img_pos_patient = np.array(dcm.ImagePositionPatient, dtype=float)
                img_orient_patient = np.array(dcm.ImageOrientationPatient, dtype=float)
                pixel_spacing = np.array(dcm.PixelSpacing, dtype=float)
                row_vec = img_orient_patient[3:6]; col_vec = img_orient_patient[0:3]
                x_coord = float(coords_xy['x']); y_coord = float(coords_xy['y'])
                physical_coords = img_pos_patient + (x_coord * pixel_spacing[0] * col_vec) + (y_coord * pixel_spacing[1] * row_vec)
                return tuple(physical_coords)
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find DICOM slice with SOPInstanceUID {sop_uid} in {dicom_folder_path}")


# --------------------------------------------------------------------------
# 3. CORE PREPROCESSING FUNCTIONS (WITH QUALITY & PERFORMANCE IMPROVEMENTS)
# --------------------------------------------------------------------------


def load_and_reorient_dicom(dicom_folder_path: str) -> sitk.Image:
    """
    Loads, ensures the image is 3D, converts pixel type to float32, and orients a DICOM series to LPS standard.
    This version handles cases where the DICOM reader incorrectly creates a 4D image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series in {dicom_folder_path}")
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()


    if image_itk.GetDimension() == 4:
        print(f"Warning: DICOM series was read as a 4D image with size {image_itk.GetSize()}.")
        print("Extracting the first 3D volume.")
        
        image_itk = image_itk[:, :, :, 0]
        
        print(f"Image is now 3D with size {image_itk.GetSize()}.")


    final_itk_image = sitk.Cast(image_itk, sitk.sitkFloat32)
    
    # Finally, perform the orientation.
    return sitk.DICOMOrient(final_itk_image, 'LPS')


# --- FINAL, ROBUST n4_bias_field_correction FUNCTION ---
def n4_bias_field_correction(itk_image: sitk.Image) -> sitk.Image:
    """
    Corrects for intensity non-uniformity using the community-standard, robust
    workflow for N4 bias field correction in SimpleITK.
    Now with correct method name and dynamic shrink factor.
    """
    print("Applying N4 Bias Field Correction (Robust & Fast)...")

    # Ensure the input image is Float32, a requirement for the N4 filter.
    original_image_type = itk_image.GetPixelID()
    if original_image_type != sitk.sitkFloat32:
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)

    # 1. Calculate appropriate shrink factor to avoid zero dimensions
    image_size = itk_image.GetSize()
    min_dimension = min(image_size)
    
    # Choose shrink factor that ensures no dimension becomes smaller than 4 pixels
    max_shrink_factor = max(1, min_dimension // 4)  # Ensure at least 4 pixels in smallest dimension
    shrinkFactor = min(4, max_shrink_factor)  # Use original factor of 4 if possible, otherwise reduce
    
    print(f"Using shrink factor: {shrinkFactor} (image size: {image_size})")
    
    shrunk_image = sitk.Shrink(itk_image, [shrinkFactor] * itk_image.GetDimension())
    
    # Verify the shrunk image has valid dimensions
    shrunk_size = shrunk_image.GetSize()
    if min(shrunk_size) == 0:
        print("Warning: Shrink operation resulted in zero dimension, skipping N4 correction")
        return itk_image
    
    print(f"Shrunk image size: {shrunk_size}")

    # 2. Create a mask from the downsampled image.
    mask_image = sitk.OtsuThreshold(shrunk_image, 0, 1, 200)

    # 3. Initialize the N4 corrector.
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([10, 10, 10])
    corrector.SetConvergenceThreshold(0.001)

    # 4. Execute the correction on the downsampled image and mask.
    try:
        corrector.Execute(shrunk_image, mask_image)
    except RuntimeError as e:
        if "Zero-valued spacing" in str(e):
            print(f"Warning: N4 correction failed due to spacing issue, returning original image: {e}")
            return itk_image
        else:
            raise e

    # 5. Get the log bias field automatically resampled to match the original image
    # This method automatically handles the resampling to the reference image space
    log_bias_field_full = corrector.GetLogBiasFieldAsImage(itk_image)

    # 6. Apply the full-resolution bias field to the original image.
    corrected_image_full_resolution = sitk.Divide(itk_image, sitk.Exp(log_bias_field_full))
    
    # Cast back to original type if necessary
    if corrected_image_full_resolution.GetPixelID() != original_image_type:
        corrected_image_full_resolution = sitk.Cast(corrected_image_full_resolution, original_image_type)

    return corrected_image_full_resolution




def skull_strip_hd_bet(itk_image: sitk.Image) -> sitk.Image:
    """
    Removes non-brain tissue using HD-BET in a multiprocess-safe manner
    by creating a unique temporary directory for each run.
    """
    print("Performing skull stripping with HD-BET...")
    
    # Create a unique temporary directory for this specific process run.
    # The 'with' statement ensures this directory and its contents are automatically
    # cleaned up when we're done, even if an error occurs.
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define file paths INSIDE our unique temporary directory
        temp_input_path = os.path.join(temp_dir, "input.nii.gz")
        temp_output_path = os.path.join(temp_dir, "output.nii.gz")


        try:
            # Write the input image
            sitk.WriteImage(itk_image, temp_input_path)


            # Construct and run the command
            command = ["hd-bet", "-i", temp_input_path, "-o", temp_output_path, "-device", "cuda", "--disable_tta"]
            subprocess.run(command, check=True, capture_output=True)


            # If the command was successful, read the result back
            brain_only_itk = sitk.ReadImage(temp_output_path)


        except subprocess.CalledProcessError as e:
            print("--- HD-BET FAILED ---")
            # Decode stderr for better error messages
            print(e.stderr.decode('utf-8', errors='replace'))
            raise e
        except FileNotFoundError:
            print("--- HD-BET FAILED ---")
            print("Error: 'hd-bet' command not found. Make sure it is installed and in your system's PATH.")
            raise
    
    # The temporary directory and its contents (input.nii.gz, output.nii.gz, dataset.json, etc.)
    # are automatically deleted upon exiting the 'with' block.
    return brain_only_itk


def crop_to_brain(itk_image: sitk.Image) -> sitk.Image:
    """Crops to a tight bounding box around the brain."""
    print("Cropping to a tight bounding box around the brain...")
    brain_mask = sitk.Cast(itk_image > 0, sitk.sitkUInt8)
    label_stats_filter = sitk.LabelShapeStatisticsImageFilter()
    label_stats_filter.Execute(brain_mask)
    if not label_stats_filter.GetLabels():
        return itk_image
    bounding_box = label_stats_filter.GetBoundingBox(1)
    return sitk.RegionOfInterest(itk_image, bounding_box[3:], bounding_box[0:3])


def normalize_mri_intensity(itk_image: sitk.Image, modality: str) -> sitk.Image:
    """
    Normalizes intensity range, with a robust, modality-aware workflow that
    preserves contrast and avoids artifacts, especially for T2-weighted images.
    """
    print(f"Normalizing intensity for {modality.upper()}...")
    image_np = sitk.GetArrayFromImage(itk_image)
    
    # Create a mask of the brain/head region from the input. We will use this
    # at the end to ensure we don't re-introduce background voxels.
    # A value slightly above zero is used to be safe.
    original_mask = image_np > 1e-3
    non_zero_voxels = image_np[original_mask]

    if non_zero_voxels.size == 0:
        print("Warning: No non-zero voxels found in image. Returning original.")
        return itk_image
    
    # --- Modality-Specific Processing ---
    if modality.lower() == 'mri t2':
        print("Inverting MRI T2 signal to make vessels bright...")
        
        # 1. Use a robust percentile for the maximum value to avoid outlier influence.
        p99_9 = np.percentile(non_zero_voxels, 99.9)
        
        # 2. Invert the signal using the robust max.
        inverted_np = p99_9 - image_np
        
        # 3. Use the ORIGINAL mask to zero out the background. This is the crucial
        #    fix that prevents creating black holes from inverted bright signals (like CSF).
        inverted_np[~original_mask] = 0
        
        # The image we will proceed with is the inverted one.
        image_to_normalize = inverted_np
        
    else: # For MRA, T1-post, etc.
        image_to_normalize = image_np

    # --- General Normalization Steps ---
    
    # Get the non-zero voxels from the image we've chosen to normalize
    # (either the original or the inverted T2)
    non_zero_voxels_for_norm = image_to_normalize[original_mask]
    
    # 4. Clip intensities using percentiles to remove extreme outliers at both ends.
    #    This improves contrast in the relevant intensity range.
    p_low = np.percentile(non_zero_voxels_for_norm, 0.5)
    p_high = np.percentile(non_zero_voxels_for_norm, 99.8)
    
    clipped_np = np.clip(image_to_normalize, p_low, p_high)
    
    # 5. Scale the intensities to the standard [0, 1] range.
    min_val, max_val = clipped_np.min(), clipped_np.max()
    
    if max_val > min_val:
        normalized_np = (clipped_np - min_val) / (max_val - min_val)
    else:
        # Handle the edge case where all values are the same
        normalized_np = clipped_np * 0
        
    # Final sanity check: ensure the background is still zero.
    normalized_np[~original_mask] = 0
        
    # --- Convert back to SimpleITK Image ---
    normalized_itk = sitk.GetImageFromArray(normalized_np.astype(np.float32))
    normalized_itk.CopyInformation(itk_image)
    
    return normalized_itk



def resample_image(itk_image: sitk.Image, target_spacing: Tuple[float, float, float], pre_smoothing_sigma: Optional[float] = 0.25) -> sitk.Image:
    """Resamples image to target spacing with optional pre-smoothing to reduce pixelation."""
    
    image_to_resample = itk_image
    # --- PIXELATION vs PERFORMANCE FIX ---
    if pre_smoothing_sigma is not None and pre_smoothing_sigma > 0:
        print(f"Applying gentle Gaussian pre-smoothing (sigma={pre_smoothing_sigma})...")
        if all(s >= 4 for s in itk_image.GetSize()):
            image_to_resample = sitk.SmoothingRecursiveGaussian(itk_image, pre_smoothing_sigma)
    else:
        print("Skipping pre-smoothing for maximum performance.")


    print(f"Resampling image to spacing: {target_spacing}...")
    original_spacing = image_to_resample.GetSpacing()
    original_size = image_to_resample.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing); resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_to_resample.GetDirection()); resampler.SetOutputOrigin(image_to_resample.GetOrigin())
    resampler.SetTransform(sitk.Transform()); resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image_to_resample)


def crop_to_head_mri(itk_image: sitk.Image) -> sitk.Image:
    """
    Crops the image to the largest connected component (the head) using an
    automatic threshold and robust blob analysis.
    """
    print("Cropping to a tight bounding box around the head...")
    
    # Use Otsu's method to find an automatic threshold between head and background
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    binary_mask = otsu_filter.Execute(itk_image)
    
    # Use morphological opening to remove small, noisy regions
    radius_mm = 3
    spacing = itk_image.GetSpacing()
    radius_pixels = [int(round(radius_mm / sp)) for sp in spacing if sp > 0]
    if not radius_pixels: radius_pixels = [3, 3, 3] # Fallback
    opened_mask = sitk.BinaryMorphologicalOpening(binary_mask, kernelRadius=radius_pixels, kernelType=sitk.sitkBall)
    
    # Find the largest connected object in the mask (which will be the patient's head)
    relabeled_mask = sitk.RelabelComponent(sitk.ConnectedComponent(opened_mask), sortByObjectSize=True)
    largest_component_mask = relabeled_mask == 1
    
    # Fill any holes inside the largest object
    filled_mask = sitk.BinaryFillhole(largest_component_mask)
    
    # Get the bounding box of this object and crop the original image
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(filled_mask)
    if not stats.GetLabels():
        print("Warning: crop_to_head_mri failed to find a valid object. Returning original image.")
        return itk_image
    bbox = stats.GetBoundingBox(1)
    return sitk.RegionOfInterest(itk_image, size=bbox[3:], index=bbox[:3])



def find_neck_cutoff_mri(image_np: np.ndarray, intensity_threshold: float = 0.1) -> int:
    """
    Finds the Z-slice to cut off the neck based on changes in cross-sectional area.
    Adapted from the CTA 'find_neck_cutoff' function to use a generic intensity threshold.
    """
    print("Finding neck cutoff slice...")
    if image_np.ndim != 3 or image_np.size == 0: return 0
    
    # Create a mask based on the intensity threshold
    mask = image_np > intensity_threshold
    # Calculate the area of the head in each Z-slice
    areas = np.sum(mask, axis=(1, 2))
    
    non_zero_indices = np.where(areas > 0)[0]
    if len(non_zero_indices) < 20: return 0 # Not enough slices to make a determination
    
    # Smooth the area profile to reduce noise
    smoothed_areas = np.convolve(areas[non_zero_indices], np.ones(5)/5, mode='same')
    
    # Find the first major peak (usually the widest part of the head)
    peaks, _ = find_peaks(smoothed_areas, prominence=np.max(smoothed_areas)*0.1, distance=10)
    if not peaks.any(): return 0
    
    first_peak_idx = peaks[0]
    search_area = smoothed_areas[first_peak_idx:]
    
    # Find the first major valley after the peak (this corresponds to the neck)
    valleys, _ = find_peaks(-search_area, prominence=np.max(search_area)*0.05, distance=10)
    if not valleys.any(): return 0
    
    cutoff_local_idx = valleys[0] + first_peak_idx
    cutoff_z = non_zero_indices[cutoff_local_idx]
    
    return int(cutoff_z)


# --------------------------------------------------------------------------
# 4. MAIN PREPROCESSING PIPELINE (WITH PERFORMANCE CONTROL)
# --------------------------------------------------------------------------
def preprocess_mri_scan(
    dicom_folder_path: str,
    modality: str,
    target_spacing: tuple = (0.58, 0.58, 1.2),
    initial_coords_list: Optional[List[dict]] = None,
    pre_smoothing_sigma: Optional[float] = 0.25,
    DEBUG_MODE: bool = False
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[List[dict]]]:
    """
    Main pipeline to create a standardized MRI scan with a robust, re-ordered workflow.
    """
    if modality.lower() not in ['mra', 't1post', 'mri t2']:
        raise ValueError("Modality must be 'mra', 'mri t1post', or 'mri t2'")
    
    print("\n--- STARTING PREPROCESSING ---")
    
    # --- STEP 1: Load Image (No Change) ---
    reoriented_itk = load_and_reorient_dicom(dicom_folder_path)
    
    # Coordinate handling logic (No Change)
    aneurysm_physical_points_info = [] 
    if initial_coords_list:
        for coord_info in initial_coords_list:
            point = get_physical_point_from_dicom(dicom_folder_path, coord_info['sop_uid'], coord_info['coords_xy'])
            aneurysm_physical_points_info.append({'physical_point': point, 'location': coord_info.get('location', 'N/A')})
    
    # --- NEW, OPTIMAL ORDER OF OPERATIONS ---


    # --- STEP 2: Crop First (As you suggested) ---
    # This robustly isolates the head before any other processing.
    head_only_itk = crop_to_head_mri(reoriented_itk)
    del reoriented_itk # Free memory
        
    # --- STEP 3: N4 Bias Correction on the Cropped Head ---
    # Working on the cropped region is faster and more accurate.
    corrected_itk = n4_bias_field_correction(head_only_itk)
    del head_only_itk # Free memory


    # --- STEP 4: Normalize the Corrected Head ---
    # Normalizing now gives much better contrast because it's focused on the ROI.
    normalized_itk = normalize_mri_intensity(corrected_itk, modality)
    del corrected_itk # Free memory


    # --- STEP 5: Resample as the Final Step ---
    final_itk_image = resample_image(normalized_itk, target_spacing=target_spacing, pre_smoothing_sigma=pre_smoothing_sigma)
    del normalized_itk # Free memory
    
    # --- Final coordinate transformation logic (No Change) ---
    final_image_np = sitk.GetArrayFromImage(final_itk_image)
    
    final_output_list = None
    if aneurysm_physical_points_info:
        final_output_list = []
        for info in aneurysm_physical_points_info:
            physical_point = info['physical_point']
            final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(physical_point)
            final_image_size = final_itk_image.GetSize()
            is_inside = all(0 <= c < s for c, s in zip(final_voxel_coords_xyz, final_image_size))
            if is_inside:
                final_output_list.append({'final_coords_zyx': final_voxel_coords_xyz[::-1], 'location': info['location']})
            else:
                print(f"WARNING: Aneurysm at physical point {physical_point} was cropped out.")


    if DEBUG_MODE and final_output_list:
        for i, info in enumerate(final_output_list):
            visualize_location_in_3d(final_image_np, info['final_coords_zyx'], title=f"AFTER (Aneurysm #{i+1} - {info['location']})")
            
    print("--- PREPROCESSING COMPLETE ---")
    return final_image_np, target_spacing, final_output_list