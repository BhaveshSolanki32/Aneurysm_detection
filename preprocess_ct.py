# Save this as preprocess_ct.py
import SimpleITK as sitk
import numpy as np
import os
import pydicom
from scipy.signal import find_peaks
from typing import Tuple, List
from view3d_data import display_hu_distribution

# --- Robust HU Conversion (Your provided function) ---

def robust_hu_conversion(dicom_dataset: pydicom.Dataset) -> np.ndarray:
    """
    Converts pixel data to Hounsfield Units (HU) with a heuristic to avoid
    re-converting data that may already be in HU.

    Args:
        dicom_dataset (pydicom.Dataset): A pydicom dataset for a single slice.

    Returns:
        np.ndarray: The pixel array, converted to HU if necessary.
    """
    pixel_array = dicom_dataset.pixel_array.astype(np.float32)

    # Heuristic Check: If the RescaleSlope/Intercept tags are missing, or if the
    # data range already looks like HU (e.g., contains typical air values),
    # return the array as is.
    if 'RescaleSlope' not in dicom_dataset or 'RescaleIntercept' not in dicom_dataset:
        return pixel_array
    
    slope = float(dicom_dataset.RescaleSlope)
    intercept = float(dicom_dataset.RescaleIntercept)

    # Common padding values in DICOM are -2000. Air is ~ -1000 HU.
    # If min value is already very low, assume it's already in HU.
    if pixel_array.min() < -500:
        return pixel_array
    else:
        return (pixel_array * slope) + intercept

# --- NEW: Manual DICOM Loading Function ---

def load_dicom_series_manually(dicom_folder_path: str) -> sitk.Image:
    """
    Loads a DICOM series manually using pydicom to apply robust HU conversion
    on a slice-by-slice basis.

    Args:
        dicom_folder_path (str): Path to the folder containing DICOM files.

    Returns:
        sitk.Image: A SimpleITK image with correctly applied HU conversion and metadata.
    """
    # Use SimpleITK's reader just to get a sorted list of file names
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series found in directory: {dicom_folder_path}")

    # Read all slices using pydicom
    slices = [pydicom.dcmread(dcm) for dcm in dicom_names]

    # Sort slices by instance number or slice location to be safe
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Apply robust HU conversion to each slice and stack them

    hu_slices = [robust_hu_conversion(s) for s in slices]
    image_3d_np = np.stack(hu_slices, axis=0)
    display_hu_distribution(image_3d_np, 'raw_array')#temp------------------------------------------->

    # --- Create a SimpleITK image with correct metadata ---
    image_itk = sitk.GetImageFromArray(image_3d_np)

    # 1. Spacing
    # XY spacing is from PixelSpacing
    # Z spacing is the distance between slice centers
    pixel_spacing = slices[0].PixelSpacing
    slice_positions = [s.ImagePositionPatient[2] for s in slices]
    
    # Calculate z-spacing robustly from slice positions
    if len(slice_positions) > 1:
        z_spacing = np.abs(slice_positions[1] - slice_positions[0])
    else:
        # Fallback to SliceThickness if only one slice
        z_spacing = slices[0].SliceThickness if 'SliceThickness' in slices[0] else 1.0

    image_itk.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing)))

    # 2. Origin (position of the first pixel in the first slice)
    image_itk.SetOrigin(slices[0].ImagePositionPatient)

    # 3. Direction Cosines (orientation of the axes)
    orientation = slices[0].ImageOrientationPatient
    direction_cosines = [float(o) for o in orientation]
    # SimpleITK expects a 9-element flat list/tuple
    # [d_x_col, d_y_col, d_z_col] = [[Xx, Xy, Xz], [Yx, Yy, Yz], [Zx, Zy, Zz]]
    # DICOM provides [Xx, Xy, Xz, Yx, Yy, Yz]
    row_cosines = direction_cosines[0:3]
    col_cosines = direction_cosines[3:6]
    z_dir = np.cross(row_cosines, col_cosines)
    full_direction = (*row_cosines, *col_cosines, *z_dir)
    image_itk.SetDirection(full_direction)

    return image_itk


# --- Modular Preprocessing Steps (Updated Loader) ---

def load_and_reorient_dicom(dicom_folder_path: str) -> Tuple[sitk.Image, Tuple[float, float, float]]:
    """
    Loads a DICOM series using a robust manual method and reorients it to a
    standard 'upright' axial orientation (LPS) without changing its spacing.

    Args:
        dicom_folder_path (str): Path to the folder containing DICOM files.

    Returns:
        Tuple[sitk.Image, Tuple[float, float, float]]:
            - The reoriented SimpleITK image.
            - The original spacing of the reoriented image.
    """
    # Step 1: Load the series with the robust, manual pydicom method
    itk_image = load_dicom_series_manually(dicom_folder_path)

    # Step 2: Reorient the manually loaded image
    # Create a reference grid with the *original* spacing but a standard orientation
    reorient_grid = create_reference_grid(itk_image, target_spacing=itk_image.GetSpacing())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reorient_grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0) # Typical CT background
    reoriented_itk_image = resampler.Execute(itk_image)

    return reoriented_itk_image, reoriented_itk_image.GetSpacing()


# --- All other functions from the previous refactoring remain UNCHANGED ---
# They are included here for completeness of the script.

def create_reference_grid(itk_image: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
    """Creates a reference SimpleITK Image for resampling."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    reference_image = sitk.Image(new_size, itk_image.GetPixelIDValue())
    reference_image.SetSpacing(target_spacing)
    reference_image.SetOrigin(itk_image.GetOrigin())
    reference_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # Standard axial orientation
    return reference_image

def find_neck_cutoff(mask: np.ndarray) -> int:
    """Analyzes the area of a binary mask to find the neck cutoff slice."""
    if mask.ndim != 3 or mask.size == 0: return 0
    areas = np.sum(mask, axis=(1, 2))
    display_hu_distribution(areas, 'neck_crop_mask')
    non_zero_indices = np.where(areas != 0)[0]
    if len(non_zero_indices) < 3: return 0
    areas_filtered = areas[non_zero_indices]
    min_area, max_area = areas_filtered.min(), areas_filtered.max()
    if max_area == min_area: return 0
    normalized_areas = (areas_filtered - min_area) / (max_area - min_area)
    peaks, _ = find_peaks(normalized_areas, prominence=0.005)
    valleys, _ = find_peaks(-normalized_areas, prominence=0.005)
    if len(peaks) == 0: return 0
    if len(valleys) > 0:
        min_valley_local_idx = np.argmin(normalized_areas[valleys])
        min_valley_filtered_idx = valleys[min_valley_local_idx]
        cutoff_z = non_zero_indices[min_valley_filtered_idx]
        return int(cutoff_z)
    return 0

def apply_hu_window(itk_image: sitk.Image, hu_window: Tuple[int, int] = (-100, 400)) -> np.ndarray:
    """Applies HU windowing and scales the result to a [0, 1] float range."""
    image_hu = sitk.GetArrayFromImage(itk_image)
    hu_min, hu_max = hu_window
    clipped_image = np.clip(image_hu, hu_min, hu_max)
    scaled_image = (clipped_image - hu_min) / (hu_max - hu_min)
    return scaled_image.astype(np.float32)

def crop_neck(scaled_image_np: np.ndarray, crop_threshold: float = (-200,600)) -> np.ndarray:
    """Crops the z-axis to remove the neck region."""
    binary_mask = ((scaled_image_np > crop_threshold[0]) & (scaled_image_np < crop_threshold[1]))
    neck_cutoff_z = find_neck_cutoff(binary_mask)
    return scaled_image_np[neck_cutoff_z:, :, :]

def crop_to_largest_component(image_np: np.ndarray, crop_threshold: float = (-200,600)) -> np.ndarray:
    """Crops X and Y axes to the bounding box of the largest connected component."""
    binary_mask = ((image_np > crop_threshold[0]) & (image_np < crop_threshold[1])).astype(np.uint8)
    mask_itk = sitk.GetImageFromArray(binary_mask)
    connected_components = sitk.ConnectedComponent(mask_itk)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected_components)
    if not stats.GetLabels(): return image_np
    largest_label = max(stats.GetLabels(), key=stats.GetNumberOfPixels)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(largest_label)
    return image_np[z:(z + sz), y:(y + sy), x:(x + sx)]

def resample_image(image_np: np.ndarray, current_spacing: Tuple[float, float, float], target_spacing: Tuple[float, float, float] = (0.58, 0.58, 1.2)) -> np.ndarray:
    """Resamples a NumPy array to a new target spacing."""
    image_itk = sitk.GetImageFromArray(image_np)
    image_itk.SetSpacing(current_spacing)
    final_grid = create_reference_grid(image_itk, target_spacing=target_spacing)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(final_grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    final_itk_image = resampler.Execute(image_itk)
    return sitk.GetArrayFromImage(final_itk_image)

# --- Main Pipeline Function (unchanged logic) ---
def mask_tissue(img_np: np.ndarray,sof_tissue_range_hu: tuple) -> np.ndarray:
    soft_tissue = np.where((img_np > sof_tissue_range_hu[0]) & (img_np < sof_tissue_range_hu[1]),img_np,img_np.min())
    # soft_tissue_np = img_np[soft_tissue_idxs]
    return soft_tissue

def preprocess_cta_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (0.58, 0.58, 1.2),
    hu_window: tuple = (0.4, 0.6),
    crop_threshold: float = (-200,600),
    sof_tissue_range_hu: tuple = (-600,1000)
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Full preprocessing pipeline for a CTA scan."""
    # Step 1: Load and Reorient (Now uses the robust manual loader)
    reoriented_itk_image, original_spacing = load_and_reorient_dicom(dicom_folder_path)

    # normalize data
    reoriented_itk_image_np = sitk.GetArrayFromImage(reoriented_itk_image)
    # min_reoriented_itk_image_np, max_reoriented_itk_image_np = np.min(reoriented_itk_image_np) , np.max(reoriented_itk_image_np)
    # normalized_tissue = (reoriented_itk_image_np - min_reoriented_itk_image_np) / (max_reoriented_itk_image_np - min_reoriented_itk_image_np)
    # display_hu_distribution(normalized_tissue,'normalized_tissue')
   
    # Step 2: get soft tissue
    # sof_tissue_range_hu_normalized = (sof_tissue_range_hu-min_reoriented_itk_image_np) / (max_reoriented_itk_image_np - min_reoriented_itk_image_np)
    soft_tissue_img_np = mask_tissue(reoriented_itk_image_np, sof_tissue_range_hu)
    display_hu_distribution(soft_tissue_img_np,'soft_tissue_img_np')
    # Step 3: normalize the data

    scaled_image_np = soft_tissue_img_np#apply_hu_window(reoriented_itk_image, hu_window)
    # Step 3: Crop Neck (Z-axis)
    head_image_np = crop_neck(scaled_image_np, crop_threshold)
    display_hu_distribution(head_image_np,'head_image_np')
    # Step 4: Crop to Largest Component (X, Y axes)
    cropped_head_image_np = crop_to_largest_component(head_image_np, crop_threshold)
    display_hu_distribution(cropped_head_image_np,'cropped_head_image_np')
    # Step 5: Final Resample
    final_image_np = resample_image(
        cropped_head_image_np,
        current_spacing=original_spacing,
        target_spacing=target_spacing
    )
    return final_image_np, target_spacing