# Save this as preprocess_ct.py
import SimpleITK as sitk
import numpy as np
import os
from scipy.signal import find_peaks

def create_reference_grid(itk_image, target_spacing):
    """Helper function to create a reference grid for resampling."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    
    reference_image = sitk.Image(new_size, itk_image.GetPixelIDValue())
    reference_image.SetSpacing(target_spacing)
    reference_image.SetOrigin(itk_image.GetOrigin())
    reference_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1]) # Standard axial orientation
    
    return reference_image

def find_neck_cutoff(mask: np.ndarray) -> int:
    areas = np.sum(mask, axis=(1, 2))
    areas = areas[areas != 0]
    areas = (areas - areas.min()) / (areas.max() - areas.min())
    peaks, _ = find_peaks(areas, prominence=0.005)
    valleys, _ = find_peaks(-areas, prominence=0.005)

    valley_idx = 0

    if len(peaks) == 0:
        return 0

    if len(peaks) > 1:
        valley_idx = valleys[np.argmin(areas[valleys])]


    return valley_idx

def preprocess_cta_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (0.58, 0.58, 1.2),
    hu_window: tuple = (-100, 400),
    crop_threshold: float = 0.8,
) -> np.ndarray:
    """
    V4: Implements your proposed pipeline:
    1. Reorient (don't resample yet)
    2. Crop Z-axis (neck removal)
    3. Crop X,Y-axis (slice-wise)
    4. Final Resample
    """
    # 1. Load & Reorient to Upright Axial (without changing spacing yet)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    reader.SetFileNames(dicom_names)
    itk_image = reader.Execute()

    # Create a reference grid with the *original* spacing but a standard orientation
    reorient_grid = create_reference_grid(itk_image, target_spacing=itk_image.GetSpacing())
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reorient_grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(int(np.min(hu_window)))
    reoriented_itk_image = resampler.Execute(itk_image)

    # Convert to NumPy for processing
    image_hu = sitk.GetArrayFromImage(reoriented_itk_image)
    
    # Perform HU windowing
    hu_min, hu_max = hu_window
    clipped_image = np.clip(image_hu, hu_min, hu_max)
    scaled_image = (clipped_image - hu_min) / (hu_max - hu_min)
    scaled_image = scaled_image.astype(np.float32)

    # 2. Crop Z-Axis using your neck cutoff logic
    binary_mask = scaled_image > crop_threshold
    neck_cutoff_z = find_neck_cutoff(binary_mask)
    
    # Keep everything from the neck up
    head_image = scaled_image[neck_cutoff_z:, :, :]
    head_mask = binary_mask[neck_cutoff_z:, :, :]

    # 3. Crop X,Y-Axis using slice-wise analysis
    head_mask_itk = sitk.GetImageFromArray(head_mask.astype(np.uint8))
    
    connected_components = sitk.ConnectedComponent(head_mask_itk)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected_components)
    
    if not stats.GetLabels():
        cropped_head_image = head_image
    else:
        largest_label = max(stats.GetLabels(), key=stats.GetNumberOfPixels)
        bounding_box = stats.GetBoundingBox(largest_label)
        x_start, y_start, z_start, size_x, size_y, size_z = bounding_box
        cropped_head_image = head_image[z_start:z_start+size_z, y_start:y_start+size_y, x_start:x_start+size_x]

    # 4. Final Resampling to target isotropic spacing
    # Convert the cropped NumPy array back to a SimpleITK image
    cropped_itk_image = sitk.GetImageFromArray(cropped_head_image)
    
    # It's crucial to set the correct spacing before resampling
    original_spacing_xyz = reoriented_itk_image.GetSpacing()
    cropped_itk_image.SetSpacing(original_spacing_xyz)

    # Now create the final reference grid with the target spacing
    final_grid = create_reference_grid(cropped_itk_image, target_spacing=target_spacing)
    
    final_resampler = sitk.ResampleImageFilter()
    final_resampler.SetReferenceImage(final_grid)
    final_resampler.SetInterpolator(sitk.sitkLinear)
    final_resampler.SetDefaultPixelValue(0.0) # Background is 0 after scaling
    
    final_itk_image = final_resampler.Execute(cropped_itk_image)
    
    final_image_np = sitk.GetArrayFromImage(final_itk_image)

    return final_image_np, target_spacing