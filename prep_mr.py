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


def load_and_reorient_dicom(dicom_path: str) -> sitk.Image:
    """
    Loads a DICOM series from a folder or a single multi-frame DICOM file,
    ensures it is 3D, converts to float32, and orients to LPS standard.
    """
    if os.path.isdir(dicom_path):
        print(f"Path is a directory, using ImageSeriesReader for: {dicom_path}")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
        if not dicom_names:
            raise FileNotFoundError(f"No DICOM series found in directory: {dicom_path}")
        reader.SetFileNames(dicom_names)
        image_itk = reader.Execute()
    elif os.path.isfile(dicom_path):
        print(f"Path is a file, using ImageFileReader for: {dicom_path}")
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_path)
        image_itk = reader.Execute()
    else:
        raise FileNotFoundError(f"Path does not exist or is not a file/directory: {dicom_path}")

    if image_itk.GetDimension() == 4:
        print(f"Warning: DICOM was read as a 4D image with size {image_itk.GetSize()}. Extracting first 3D volume.")
        image_itk = image_itk[:, :, :, 0]
        print(f"Image is now 3D with size {image_itk.GetSize()}.")

    final_itk_image = sitk.Cast(image_itk, sitk.sitkFloat32)
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
    Normalizes MRI intensity using a unified contrast-stretching approach
    that enhances the visibility of bright structures (like aneurysms and vessels)
    across all modalities (T2, MRA, T1-post) without destructive inversion.
    """
    print(f"Normalizing intensity for {modality.upper()} using unified contrast stretching...")
    image_np = sitk.GetArrayFromImage(itk_image)
    
    # Create a mask of all non-background voxels. A value slightly above zero
    # is used to robustly separate anatomy from the background.
    mask = image_np > 1e-3
    
    # If the mask is empty, there's nothing to process.
    if not mask.any():
        print("Warning: Image appears to be empty. Returning as is.")
        return itk_image
        
    brain_voxels = image_np[mask]
    
    # 1. Determine the robust intensity window of the brain.
    #    We clip at the 0.5 and 99.5 percentiles to discard the darkest noise
    #    and the brightest artifacts, focusing only on the relevant tissue signal.
    p_low = np.percentile(brain_voxels, 0.5)
    p_high = np.percentile(brain_voxels, 99.5)
    
    # 2. Clip the entire image to this robust window.
    #    This makes the normalization much less sensitive to outliers.
    clipped_np = np.clip(image_np, p_low, p_high)
    
    # 3. Normalize (stretch) the clipped intensity range to [0, 1].
    #    This is the core step that enhances contrast.
    min_val = clipped_np.min()
    max_val = clipped_np.max()
    
    if max_val > min_val:
        normalized_np = (clipped_np - min_val) / (max_val - min_val)
    else:
        # Avoid division by zero if all values are the same
        normalized_np = clipped_np * 0
        
    # 4. Re-apply the original mask to ensure the background remains perfectly black.
    normalized_np[~mask] = 0
        
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
    Crops the image to the largest connected component (the head) using a
    robust, multi-step masking process that is more reliable than a single threshold.
    """
    print("Cropping to a tight bounding box around the head...")
    
    # 1. Use Otsu's method to get a starting threshold for the head vs. background.
    #    This uses the correct object-oriented calling style to avoid the TypeError.
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    head_mask = otsu_filter.Execute(itk_image)
    
    # 2. Fill holes within the mask. This is crucial for making the head a single, solid object.
    filled_mask = sitk.BinaryFillhole(head_mask)
    
    # 3. Find all connected objects and keep only the largest one.
    #    This eliminates any other bright objects (e.g., neck muscle, artifacts).
    relabeled_mask = sitk.RelabelComponent(sitk.ConnectedComponent(filled_mask), sortByObjectSize=True)
    largest_component_mask = (relabeled_mask == 1)
    
    # 4. Get the bounding box of this final, clean mask and crop the original image.
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(largest_component_mask)
    
    # Check if any object was found.
    if not stats_filter.GetLabels():
        print("Warning: crop_to_head_mri failed to find a valid object. Returning original image.")
        return itk_image
        
    bbox = stats_filter.GetBoundingBox(1)  # Bounding box of the largest component (label 1)
    
    # The bbox is in the format: [start_x, start_y, start_z, size_x, size_y, size_z]
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

def get_fallback_voxel_points(
    coords_xy: dict,
    final_image_shape: Tuple[int, int, int],
    location_label: str = "FALLBACK"
) -> List[dict]:
    """
    If UID-based localization fails, this function generates a list of potential
    aneurysm locations as a fallback. It uses the original X,Y coordinates and
    places a marker on every 15th Z-slice of the final processed image.

    Args:
        coords_xy: The dictionary with 'x' and 'y' pixel coordinates from the annotation.
        final_image_shape: The (z, y, x) shape of the final numpy array.
        location_label: The original location label to append "_FALLBACK" to.

    Returns:
        A list of dictionaries, each containing ZYX coordinates for a fallback point.
    """
    fallback_points = []
    z_depth, y_max, x_max = final_image_shape

    # It's critical to ensure the original XY coordinates are within the bounds of the FINAL image
    # We round, convert to int, and clip to the maximum valid index (shape - 1)
    try:
        x_coord = min(int(round(float(coords_xy['x']))), x_max - 1)
        y_coord = min(int(round(float(coords_xy['y']))), y_max - 1)
    except (ValueError, TypeError):
        print(f"  -> WARNING: Invalid XY coordinates '{coords_xy}'. Cannot generate fallback.")
        return []

    # Per your request, iterate through the Z-axis, marking every 15th slice.
    # We start at index 14 (the 15th slice) and step by 15.
    for z_index in range(14, z_depth, 15):
        fallback_zyx = (z_index, y_coord, x_coord)
        fallback_points.append({
            'final_coords_zyx': fallback_zyx,
            'location': location_label
        })

    print(f"  -> Generated {len(fallback_points)} fallback locations for annotation at (X:{x_coord}, Y:{y_coord}).")
    return fallback_points
# --------------------------------------------------------------------------
# 4. MAIN PREPROCESSING PIPELINE (WITH PERFORMANCE CONTROL)
# --------------------------------------------------------------------------
def preprocess_mri_scan(
    dicom_folder_path: str,
    modality: str,
    target_spacing: tuple = (0.58, 0.58, 0.58),
    initial_coords_list: Optional[List[dict]] = None,
    pre_smoothing_sigma: Optional[float] = 0.25,
    DEBUG_MODE: bool = False
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[List[dict]]]:
    """
    Main pipeline to create a standardized MRI scan. It now includes a robust
    fallback mechanism for annotations with mismatched UIDs, preventing crashes.
    """
    if modality.lower() not in ['mra', 'mri t1post', 'mri t2']:
        raise ValueError("Modality must be 'mra', 'mri t1post', or 'mri t2'")

    print("\n--- STARTING PREPROCESSING ---")

    # --- STEP 1: Load and Orient Image ---
    reoriented_itk = load_and_reorient_dicom(dicom_folder_path)

    # --- STEP 2: Attempt to Map Coordinates ---
    # We will try the primary method, but save any failures for the fallback.
    aneurysm_physical_points_info = []
    fallback_annotations_to_process = [] # NEW: Store failed annotations here

    if initial_coords_list:
        print(f"Attempting to map {len(initial_coords_list)} annotation(s)...")
        for coord_info in initial_coords_list:
            try:
                # This is the call that might fail for multi-frame DICOMs
                point = get_physical_point_from_dicom(dicom_folder_path, coord_info['sop_uid'], coord_info['coords_xy'])
                aneurysm_physical_points_info.append({'physical_point': point, 'location': coord_info.get('location', 'N/A')})
                print(f"  -> Successfully mapped SOP UID: ...{coord_info['sop_uid'][-15:]}")
            except (FileNotFoundError, AttributeError) as e:
                # THIS IS THE FIX: Instead of crashing, we catch the error and save the annotation.
                print(f"  -> INFO: UID mapping failed ({type(e).__name__}). Flagging for fallback processing.")
                fallback_annotations_to_process.append(coord_info)
                continue

    # --- IMAGE PROCESSING STEPS (UNCHANGED) ---
    corrected_itk = n4_bias_field_correction(reoriented_itk)
    del reoriented_itk
    head_only_itk = crop_to_head_mri(corrected_itk)
    del corrected_itk
    normalized_itk = normalize_mri_intensity(head_only_itk, modality)
    del head_only_itk
    final_itk_image = resample_image(normalized_itk, target_spacing=target_spacing, pre_smoothing_sigma=pre_smoothing_sigma)
    del normalized_itk

    # --- FINAL COORDINATE TRANSFORMATION ---
    final_image_np = sitk.GetArrayFromImage(final_itk_image)
    final_output_list = []

    # First, process the successfully mapped points
    if aneurysm_physical_points_info:
        for info in aneurysm_physical_points_info:
            physical_point = info['physical_point']
            final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(physical_point)
            final_image_size = final_itk_image.GetSize()
            if all(0 <= c < s for c, s in zip(final_voxel_coords_xyz, final_image_size)):
                final_output_list.append({'final_coords_zyx': final_voxel_coords_xyz[::-1], 'location': info['location']})

    # --- NEW: GENERATE FALLBACK POINTS ---
    # Now, process any annotations that failed earlier, using the final image shape.
    if fallback_annotations_to_process:
        print(f"--- Generating fallback locations for {len(fallback_annotations_to_process)} failed annotation(s) ---")
        for failed_coord_info in fallback_annotations_to_process:
            original_xy = failed_coord_info['coords_xy']
            original_location = failed_coord_info.get('location', 'N/A')
            
            # Call the new function to get the list of (z,y,x) points
            generated_points = get_fallback_voxel_points(original_xy, final_image_np.shape, original_location)
            final_output_list.extend(generated_points)

    if DEBUG_MODE and final_output_list:
        for i, info in enumerate(final_output_list):
            visualize_location_in_3d(final_image_np, info['final_coords_zyx'], title=f"AFTER (Aneurysm #{i+1} - {info['location']})")

    print("--- PREPROCESSING COMPLETE ---")
    return final_image_np, target_spacing, final_output_list if final_output_list else None


