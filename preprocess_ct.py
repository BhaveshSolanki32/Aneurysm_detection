# Save this as preprocess_v3.py
import SimpleITK as sitk
import numpy as np
import os

# (Keep the create_reference_grid function from V2)
def create_reference_grid(itk_image, target_spacing=(1.0, 1.0, 1.0)):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    reference_image = sitk.Image(new_size, itk_image.GetPixelIDValue())
    reference_image.SetSpacing(target_spacing)
    reference_image.SetOrigin(itk_image.GetOrigin())
    reference_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    
    return reference_image

def preprocess_cta_scan_v3(
    dicom_folder_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    hu_window: tuple = (-100, 400),
    crop_background_threshold: float = 0.05
) -> np.ndarray:
    """
    V3: Includes robust cropping using connected components analysis.
    """
    if not os.path.isdir(dicom_folder_path):
        raise ValueError(f"Provided path is not a directory: {dicom_folder_path}")

    # 1. Load DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series found in: {dicom_folder_path}")
    reader.SetFileNames(dicom_names)
    itk_image = reader.Execute()

    # 2. Reorient and Resample
    reference_grid = create_reference_grid(itk_image, target_spacing)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(int(np.min(hu_window)))
    
    resampled_itk_image = resampler.Execute(itk_image)
    
    # 3. Intensity Windowing
    hu_min, hu_max = hu_window
    
    # Convert to NumPy array for windowing
    image_hu = sitk.GetArrayFromImage(resampled_itk_image)
    clipped_image = np.clip(image_hu, hu_min, hu_max)
    scaled_image = (clipped_image - hu_min) / (hu_max - hu_min)
    scaled_image = scaled_image.astype(np.float32)

    # 4. Robust Cropping (Skull Stripping)
    
    # Convert back to SimpleITK image for segmentation operations
    scaled_itk_image = sitk.GetImageFromArray(scaled_image)
    scaled_itk_image.CopyInformation(resampled_itk_image) # Keep spatial info

    # Threshold the image to create a binary mask of the foreground
    binary_mask = sitk.BinaryThreshold(scaled_itk_image, 
                                       lowerThreshold=crop_background_threshold, 
                                       upperThreshold=1.0, 
                                       insideValue=1, 
                                       outsideValue=0)

    # Find connected components
    connected_components = sitk.ConnectedComponent(binary_mask)
    
    # Get the size of each component
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected_components)
    
    # Find the largest component (the head/brain)
    largest_label = 0
    largest_size = 0
    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        if size > largest_size:
            largest_size = size
            largest_label = label
            
    # Create a mask of only the largest component
    largest_component_mask = sitk.BinaryThreshold(connected_components, 
                                                  lowerThreshold=largest_label, 
                                                  upperThreshold=largest_label, 
                                                  insideValue=1, 
                                                  outsideValue=0)

    # Get the bounding box of the largest component
    bounding_box = stats.GetBoundingBox(largest_label)
    
    # Bounding box is (x_start, y_start, z_start, size_x, size_y, size_z)
    x_start, y_start, z_start, size_x, size_y, size_z = bounding_box
    
    # Crop the original scaled image using the bounding box
    cropped_itk_image = sitk.RegionOfInterest(scaled_itk_image, 
                                              size=(size_x, size_y, size_z), 
                                              index=(x_start, y_start, z_start))

    # Convert back to NumPy array (z, y, x)
    cropped_image = sitk.GetArrayFromImage(cropped_itk_image)

    return cropped_image