# Save this as preprocess_ct.py
import SimpleITK as sitk
import numpy as np
import os
import pydicom
from scipy.signal import find_peaks
from typing import Tuple, List
import collections
from view3d_data import display_hu_distribution 

# --- Helper Functions ---
# (robust_hu_conversion, load_dicom_series_manually, load_and_reorient_dicom, crop_to_body,
# find_neck_cutoff, normalize_hu_window, resample_image are all unchanged and correct)

# ... [Paste all the other helper functions here exactly as they were in the last version] ...

def robust_hu_conversion(dicom_dataset):
    pixel_array = dicom_dataset.pixel_array
    
    if 'RescaleSlope' in dicom_dataset and 'RescaleIntercept' in dicom_dataset:
        slope = float(dicom_dataset.RescaleSlope)
        intercept = float(dicom_dataset.RescaleIntercept)
        if pixel_array.min() < -500 or pixel_array.min() == -2000:
            return pixel_array
        else:
            return (pixel_array * slope) + intercept
            
    else:
        return pixel_array

# The only function that needs to be updated is load_dicom_series_manually

def load_dicom_series_manually(dicom_folder_path: str) -> sitk.Image:
    """
    Loads a DICOM series manually, handling series with inconsistent slice dimensions
    and spatially duplicate slices.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series found in directory: {dicom_folder_path}")

    slices = [pydicom.dcmread(dcm, stop_before_pixels=False) for dcm in dicom_names]

    # --- Filter 1: Keep only slices with pixel data and consistent shapes ---
    slices_with_pixels = [s for s in slices if hasattr(s, 'pixel_array') and hasattr(s, 'ImagePositionPatient')]
    if not slices_with_pixels:
        raise ValueError(f"No readable slices with pixel data found in {dicom_folder_path}")
    slice_shapes = [s.pixel_array.shape for s in slices_with_pixels]
    most_common_shape = collections.Counter(slice_shapes).most_common(1)[0][0]
    uniform_slices = [s for s in slices_with_pixels if s.pixel_array.shape == most_common_shape]
    if len(uniform_slices) < len(slices):
        print(f"Warning: Discarded {len(slices) - len(uniform_slices)} slices with inconsistent shapes from series in {dicom_folder_path}.")

    # --- NEW Filter 2: Keep only spatially unique slices ---
    spatially_unique_slices_dict = {}
    for s in uniform_slices:
        # The ImagePositionPatient tag is a pydicom MultiValue, convert it to a
        # hashable tuple to use as a dictionary key.
        position = tuple(s.ImagePositionPatient)
        if position not in spatially_unique_slices_dict:
            spatially_unique_slices_dict[position] = s
            
    unique_slices = list(spatially_unique_slices_dict.values())
    
    if len(unique_slices) < len(uniform_slices):
        print(f"Warning: Discarded {len(uniform_slices) - len(unique_slices)} spatially duplicate slices.")
    # --- END NEW Filter ---
    
    # We must have at least 2 slices to form a volume with non-zero spacing
    if len(unique_slices) < 2:
        raise RuntimeError(f"Series in {dicom_folder_path} has fewer than 2 unique slices, cannot form a volume.")

    # Sort the final, clean list of slices
    try:
        unique_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, KeyError):
        unique_slices.sort(key=lambda x: int(x.InstanceNumber))

    # Proceed with the rest of the function using 'unique_slices'
    hu_slices = [robust_hu_conversion(s) for s in unique_slices]
    image_3d_np = np.stack(hu_slices, axis=0)
    image_itk = sitk.GetImageFromArray(image_3d_np)
    
    first_slice = unique_slices[0]
    pixel_spacing = first_slice.PixelSpacing
    slice_positions = [s.ImagePositionPatient[2] for s in unique_slices]
    
    # This calculation is now safe because all slice_positions are unique
    z_spacing = np.median(np.diff(sorted(slice_positions)))
    if z_spacing == 0:
        # Fallback for safety, though it should not be triggered now
        print(f"Warning: Calculated z-spacing is still zero for {dicom_folder_path}. Falling back to SliceThickness.")
        z_spacing = first_slice.SliceThickness if 'SliceThickness' in first_slice else 1.0

    image_itk.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing)))
    image_itk.SetOrigin(first_slice.ImagePositionPatient)
    orientation = first_slice.ImageOrientationPatient
    row_cosines = [float(o) for o in orientation[0:3]]
    col_cosines = [float(o) for o in orientation[3:6]]
    z_dir = np.cross(row_cosines, col_cosines)
    image_itk.SetDirection((*row_cosines, *col_cosines, *z_dir))
    
    return image_itk


def load_and_reorient_dicom(dicom_folder_path: str) -> sitk.Image:
    itk_image = load_dicom_series_manually(dicom_folder_path)
    return sitk.DICOMOrient(itk_image, 'LPS')

def crop_to_body(itk_image: sitk.Image, air_threshold_hu: int = -500) -> sitk.Image:
    binary_mask = itk_image > air_threshold_hu; radius_mm = 3; spacing = itk_image.GetSpacing()
    radius_pixels = [int(round(radius_mm / sp)) for sp in spacing]
    opened_mask = sitk.BinaryMorphologicalOpening(binary_mask, kernelRadius=radius_pixels, kernelType=sitk.sitkBall)
    relabeled_mask = sitk.RelabelComponent(sitk.ConnectedComponent(opened_mask), sortByObjectSize=True)
    largest_component_mask = relabeled_mask == 1
    filled_mask = sitk.BinaryFillhole(largest_component_mask); stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(filled_mask)
    if not stats.GetLabels(): print(f"Warning: crop_to_body failed. Returning original image."); return itk_image
    bbox = stats.GetBoundingBox(1)
    return sitk.RegionOfInterest(itk_image, size=bbox[3:], index=bbox[:3])

def find_neck_cutoff(image_np: np.ndarray, body_threshold_hu: int = -200) -> int:
    if image_np.ndim != 3 or image_np.size == 0: return 0
    mask = image_np > body_threshold_hu; areas = np.sum(mask, axis=(1, 2))
    non_zero_indices = np.where(areas > 0)[0]
    if len(non_zero_indices) < 20: return 0
    smoothed_areas = np.convolve(areas[non_zero_indices], np.ones(5)/5, mode='same')
    peaks, _ = find_peaks(smoothed_areas, prominence=np.max(smoothed_areas)*0.1, distance=10)
    if not peaks.any(): return 0
    first_peak_idx = peaks[0]; search_area = smoothed_areas[first_peak_idx:]
    valleys, _ = find_peaks(-search_area, prominence=np.max(search_area)*0.05, distance=10)
    if not valleys.any(): return 0
    cutoff_local_idx = valleys[0] + first_peak_idx
    cutoff_z = non_zero_indices[cutoff_local_idx]
    return int(cutoff_z)


# --- NEW AND IMPROVED ALIGNMENT FUNCTION ---
def align_to_midsagittal_plane(itk_image: sitk.Image) -> sitk.Image:
    """
    Aligns a 3D head CT image to its mid-sagittal plane using a robust,
    multi-resolution registration strategy.
    """
    flipper = sitk.FlipImageFilter(); flipper.SetFlipAxes((True, False, False))
    flipped_image = flipper.Execute(itk_image)
    registration_method = sitk.ImageRegistrationMethod()
    transform = sitk.Euler3DTransform()
    image_center = itk_image.TransformContinuousIndexToPhysicalPoint([(sz-1)/2.0 for sz in itk_image.GetSize()])
    transform.SetCenter(image_center)
    registration_method.SetInitialTransform(transform, inPlace=False)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.20)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=0.001, numberOfIterations=200, relaxationFactor=0.5)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInterpolator(sitk.sitkLinear)
    # registration_method.AddCommand(sitk.sitkIterationEvent)
    print("Aligning head with multi-resolution strategy...")
    final_transform = registration_method.Execute(itk_image, flipped_image)
    print(f"Final transform parameters: {final_transform.GetParameters()}")
    print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(itk_image.GetPixel(0, 0, 0)))
    resampler.SetTransform(final_transform)
    aligned_image = resampler.Execute(itk_image)
    return aligned_image


def normalize_hu_window(itk_image: sitk.Image, hu_window: Tuple[int, int]) -> sitk.Image:
    hu_min, hu_max = hu_window; window_filter = sitk.IntensityWindowingImageFilter()
    window_filter.SetWindowMinimum(float(hu_min)); window_filter.SetWindowMaximum(float(hu_max))
    window_filter.SetOutputMinimum(0.0); window_filter.SetOutputMaximum(1.0)
    return window_filter.Execute(itk_image)

def resample_image(itk_image: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
    original_spacing = itk_image.GetSpacing(); original_size = itk_image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing); resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection()); resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform()); resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(itk_image)

# --- Main Pipeline Function (Order is correct) ---

def preprocess_cta_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (0.58, 0.58, 1.2),
    cta_hu_window: tuple = (-100, 400),
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    reoriented_itk = load_and_reorient_dicom(dicom_folder_path)
    clipped_itk = sitk.Clamp(reoriented_itk, sitk.sitkFloat32, -1024, 3000)
    full_scan_np = sitk.GetArrayFromImage(clipped_itk)
    neck_cutoff_z = find_neck_cutoff(full_scan_np, body_threshold_hu=-200)
    head_and_shoulders_np = full_scan_np[neck_cutoff_z:, :, :]
    if head_and_shoulders_np.shape[0] < 5:
        raise RuntimeError(f"Neck cropping failed for {dicom_folder_path}, resulted in {head_and_shoulders_np.shape[0]} slices.")
    head_and_shoulders_itk = sitk.GetImageFromArray(head_and_shoulders_np)
    head_and_shoulders_itk.SetSpacing(clipped_itk.GetSpacing())
    head_and_shoulders_itk.SetDirection(clipped_itk.GetDirection())
    original_origin = np.array(clipped_itk.GetOrigin()); original_spacing = np.array(clipped_itk.GetSpacing())
    z_direction_vector = np.array(clipped_itk.GetDirection())[6:]
    new_origin = original_origin + neck_cutoff_z * original_spacing[2] * z_direction_vector
    head_and_shoulders_itk.SetOrigin(new_origin.tolist())
    head_itk = crop_to_body(head_and_shoulders_itk, air_threshold_hu=-500)
    aligned_head_itk = align_to_midsagittal_plane(head_itk)
    normalized_itk = normalize_hu_window(aligned_head_itk, hu_window=cta_hu_window)
    final_itk_image = resample_image(normalized_itk, target_spacing=target_spacing)
    final_image_np = sitk.GetArrayFromImage(final_itk_image)
    return final_image_np, target_spacing