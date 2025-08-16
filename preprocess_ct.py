# Save this as preprocess_v4.py
import SimpleITK as sitk
import numpy as np
import os
from scipy.signal import find_peaks
# (Keep the create_reference_grid function from V2/V3)
def create_reference_grid(itk_image, target_spacing=(1.0, 1.0, 1.0)):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    reference_image = sitk.Image(new_size, itk_image.GetPixelIDValue())
    reference_image.SetSpacing(target_spacing)
    reference_image.SetOrigin(itk_image.GetOrigin())
    reference_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return reference_image

def find_neck_cutoff(mask: np.ndarray) -> int:
    areas = np.sum(mask, axis=(1, 2))
    peaks, _ = find_peaks(areas, prominence=0.005)
    valleys, _ = find_peaks(-areas, prominence=0.005)

    valley_idx = None

    if len(peaks) == 0:
        return None, None

    if len(peaks) > 1:
        valley_idx = valleys[np.argmin(areas[valleys])]


    return valley_idx

def preprocess_cta_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    hu_window: tuple = (-100, 400),
    crop_threshold: float = 0.05,
) -> np.ndarray:
    """
    V4: Adds robust neck cropping based on cross-sectional area analysis.
    """
    # Steps 1, 2, 3 are the same as V3
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    reader.SetFileNames(dicom_names)
    itk_image = reader.Execute()

    reference_grid = create_reference_grid(itk_image, target_spacing)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(int(np.min(hu_window)))
    resampled_itk_image = resampler.Execute(itk_image)

    image_hu = sitk.GetArrayFromImage(resampled_itk_image)
    hu_min, hu_max = hu_window
    clipped_image = np.clip(image_hu, hu_min, hu_max)
    scaled_image = (clipped_image - hu_min) / (hu_max - hu_min)
    scaled_image = scaled_image.astype(np.float32)

    # 4. Create binary mask of the largest component (head + neck + shoulders)
    binary_mask_np = scaled_image > crop_threshold
    
    # Use SimpleITK for robust connected component analysis
    binary_mask_itk = sitk.GetImageFromArray(binary_mask_np.astype(np.uint8))
    connected_components = sitk.ConnectedComponent(binary_mask_itk)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected_components)
    
    largest_label = max(stats.GetLabels(), key=stats.GetNumberOfPixels)
    largest_component_mask_np = (sitk.GetArrayFromImage(connected_components) == largest_label)

    # 5. Apply your neck cropping logic
    neck_cutoff_z = find_neck_cutoff(largest_component_mask_np)
    
    # Apply the crop to both the mask and the scaled image
    final_mask = largest_component_mask_np[neck_cutoff_z:, :, :]
    final_image = scaled_image[neck_cutoff_z:, :, :]
    
    # 6. Final bounding box crop on the head-only volume
    if np.any(final_mask):
        coords = np.argwhere(final_mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Crop the image, not the mask
        cropped_image = final_image[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    else:
        # Fallback if something went wrong
        cropped_image = final_image

    return cropped_image
    # return neck_cutoff_z