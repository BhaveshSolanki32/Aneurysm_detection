# Save this as preprocess_ct.py
import SimpleITK as sitk
import numpy as np
import os
import pydicom
from scipy.signal import find_peaks
from typing import Tuple, List, Optional
import collections
import gc
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4) 
print('robust')
# --- New Helper Function for Coordinates (This part is needed for the new functionality) ---
def get_physical_point_from_dicom(
    dicom_folder_path: str, sop_uid: str, coords_xy: dict
) -> Tuple[float, float, float]:
    for filename in os.listdir(dicom_folder_path):
        filepath = os.path.join(dicom_folder_path, filename)
        if not os.path.isfile(filepath): continue
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
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


# --- Helper Functions (RESTORED TO YOUR ORIGINAL, WORKING VERSION) ---

# --- THIS FUNCTION IS NOW CORRECTED AND RESTORED TO YOUR ORIGINAL ---
def robust_hu_conversion(dicom_dataset):
    """
    Converts raw pixel data to Hounsfield Units (HU) and corrects for
    MONOCHROME1 photometric interpretation (inverted pixels).
    """
    pixel_array = dicom_dataset.pixel_array.astype(np.int32)
    
    # --- Part 1: Convert to HU (Your existing logic) ---
    hu_pixels = pixel_array
    # return pixel_array
    if 'RescaleSlope' in dicom_dataset and 'RescaleIntercept' in dicom_dataset:
        slope = float(dicom_dataset.RescaleSlope)
        intercept = float(dicom_dataset.RescaleIntercept)
        # Prevent double-conversion if already in HU
        # hu_pixels = (pixel_array * slope) + intercept

        if not (pixel_array.min() < -2500):
            print('converting to hu')
            hu_pixels = (pixel_array * slope) + intercept

    # --- Part 2: NEW - Check and Correct for MONOCHROME1 ---
    if 'PhotometricInterpretation' in dicom_dataset and dicom_dataset.PhotometricInterpretation == "MONOCHROME1":

        max_hu = np.max(hu_pixels)
        min_hu = np.min(hu_pixels)
        hu_pixels = (max_hu + min_hu) - hu_pixels
        
    return hu_pixels

def filter_bad_slices(image_3d_itk_list):
    final_itk_stack =[]
    for x in image_3d_itk_list:
        x_np = x.pixel_array.astype(np.int16)
        if x_np.max() < 5000 & x_np.min() > -5000:
            final_itk_stack.append(x)
    return final_itk_stack           



def load_dicom_series_manually(dicom_folder_path: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    if not dicom_names: raise FileNotFoundError(f"No DICOM series found in directory: {dicom_folder_path}")
    slices = [pydicom.dcmread(dcm, stop_before_pixels=False) for dcm in dicom_names]
    slices_with_pixels = [s for s in slices if hasattr(s, 'pixel_array') and hasattr(s, 'ImagePositionPatient')]
    if not slices_with_pixels: raise ValueError(f"No readable slices with pixel data found in {dicom_folder_path}")
    slice_shapes = [s.pixel_array.shape for s in slices_with_pixels]
    most_common_shape = collections.Counter(slice_shapes).most_common(1)[0][0]
    uniform_slices = [s for s in slices_with_pixels if s.pixel_array.shape == most_common_shape]
    if len(uniform_slices) < len(slices): print(f"Warning: Discarded {len(slices) - len(uniform_slices)} slices with inconsistent shapes from series in {dicom_folder_path}.")
    spatially_unique_slices_dict = {}
    for s in uniform_slices:
        position = tuple(s.ImagePositionPatient)
        if position not in spatially_unique_slices_dict: spatially_unique_slices_dict[position] = s
    unique_slices = list(spatially_unique_slices_dict.values())
    if len(unique_slices) < len(uniform_slices): print(f"Warning: Discarded {len(uniform_slices) - len(unique_slices)} spatially duplicate slices.")
    if len(unique_slices) < 2: raise RuntimeError(f"Series in {dicom_folder_path} has fewer than 2 unique slices, cannot form a volume.")
    try: unique_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, KeyError): unique_slices.sort(key=lambda x: int(x.InstanceNumber))
    unique_slices = filter_bad_slices(unique_slices)
    hu_slices = [robust_hu_conversion(s) for s in unique_slices]
    image_3d_np = np.stack(hu_slices, axis=0)
    image_itk = sitk.GetImageFromArray(image_3d_np)
    first_slice = unique_slices[0]
    pixel_spacing = first_slice.PixelSpacing
    slice_positions = [s.ImagePositionPatient[2] for s in unique_slices]
    z_spacing = np.median(np.diff(sorted(slice_positions)))
    if z_spacing == 0:
        print(f"Warning: Calculated z-spacing is still zero for {dicom_folder_path}. Falling back to SliceThickness.")
        z_spacing = first_slice.SliceThickness if 'SliceThickness' in first_slice else 1.0
    image_itk.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing)))
    image_itk.SetOrigin(first_slice.ImagePositionPatient)
    orientation = first_slice.ImageOrientationPatient
    row_cosines = [float(o) for o in orientation[0:3]]; col_cosines = [float(o) for o in orientation[3:6]]; z_dir = np.cross(row_cosines, col_cosines)
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

# --- This function is MODIFIED but only to return the transform, which is necessary ---
def align_to_midsagittal_plane(itk_image: sitk.Image) -> Tuple[sitk.Image, sitk.Transform]:
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
    return aligned_image, final_transform

# --- Main Pipeline Function ---
def preprocess_cta_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (0.58, 0.58, 1.2),
    cta_hu_window: tuple = (-200, 500),
    initial_coords_xy: Optional[dict] = None,
    sop_instance_uid: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[Tuple[int, int, int]]]:
    
    aneurysm_physical_point = None
    if initial_coords_xy and sop_instance_uid:
        print(f"Finding initial physical coordinates for SOP UID {sop_instance_uid}...")
        aneurysm_physical_point = get_physical_point_from_dicom(
            dicom_folder_path, sop_instance_uid, initial_coords_xy
        )

    reoriented_itk = load_and_reorient_dicom(dicom_folder_path)
    clipped_itk = sitk.Clamp(reoriented_itk, sitk.sitkFloat32, -1024, 3000)
    del reoriented_itk

    full_scan_np = sitk.GetArrayFromImage(clipped_itk)
    neck_cutoff_z = find_neck_cutoff(full_scan_np, body_threshold_hu=-200)
    head_and_shoulders_np = full_scan_np[neck_cutoff_z:, :, :]
    del full_scan_np 


    if head_and_shoulders_np.shape[0] < 5:
        raise RuntimeError(f"Neck cropping failed for {dicom_folder_path}, resulted in {head_and_shoulders_np.shape[0]} slices.")
    head_and_shoulders_itk = sitk.GetImageFromArray(head_and_shoulders_np)
    del head_and_shoulders_np 

    head_and_shoulders_itk.SetSpacing(clipped_itk.GetSpacing())
    head_and_shoulders_itk.SetDirection(clipped_itk.GetDirection())
    original_origin = np.array(clipped_itk.GetOrigin()); original_spacing = np.array(clipped_itk.GetSpacing())
    z_direction_vector = np.array(clipped_itk.GetDirection())[6:]
    new_origin = original_origin + neck_cutoff_z * original_spacing[2] * z_direction_vector
    head_and_shoulders_itk.SetOrigin(new_origin.tolist())
    head_itk = crop_to_body(head_and_shoulders_itk, air_threshold_hu=-500)
    del head_and_shoulders_itk 


    aligned_head_itk, alignment_transform = align_to_midsagittal_plane(head_itk)
    del head_itk 
    gc.collect()

    if aneurysm_physical_point:
        aneurysm_physical_point = alignment_transform.TransformPoint(aneurysm_physical_point)
        print(f"Transformed physical point: {aneurysm_physical_point}")
    normalized_itk = normalize_hu_window(aligned_head_itk, hu_window=cta_hu_window)
    del aligned_head_itk 

    final_itk_image = resample_image(normalized_itk, target_spacing=target_spacing)
    del normalized_itk 

    final_image_np = sitk.GetArrayFromImage(final_itk_image)
    final_voxel_coords = None
    if aneurysm_physical_point:
        final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(aneurysm_physical_point)
        final_voxel_coords = final_voxel_coords_xyz[::-1]
        print(f"Final voxel coordinates (z, y, x): {final_voxel_coords}")

    return final_image_np, target_spacing, final_voxel_coords