# Save this as preprocess_ct.py
import SimpleITK as sitk
import numpy as np
import os
import pydicom
from scipy.signal import find_peaks
from typing import Tuple, List, Optional
import collections
import gc
from scipy.ndimage import gaussian_filter1d
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4) 

# --- Add this SINGLE, UNIFIED visualization function to your script ---
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

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
    
    # --- Input Validation ---
    if not (0 <= z < image_np.shape[0] and 0 <= y < image_np.shape[1] and 0 <= x < image_np.shape[2]):
        print(f"Error: Voxel coordinates {voxel_coords_zyx} are out of bounds for image shape {image_np.shape}")
        # Show the plot anyway, but centered on the middle of the image
        z = image_np.shape[0] // 2
        y = image_np.shape[1] // 2
        x = image_np.shape[2] // 2
        title += f"\nCOORDS OUT OF BOUNDS - Showing center instead"
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Axial View (Z-plane, top-down) ---
    axes[0].imshow(image_np[z, :, :], cmap='gray', origin='lower')
    axes[0].axhline(y, color='lime', linewidth=0.8)
    axes[0].axvline(x, color='lime', linewidth=0.8)
    axes[0].scatter(x, y, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[0].set_title(f"Axial (Z = {z})")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")

    # --- Coronal View (Y-plane, front-back) ---
    axes[1].imshow(image_np[:, y, :], cmap='gray', origin='lower', aspect='auto')
    axes[1].axhline(z, color='lime', linewidth=0.8)
    axes[1].axvline(x, color='lime', linewidth=0.8)
    axes[1].scatter(x, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[1].set_title(f"Coronal (Y = {y})")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Z-axis")

    # --- Sagittal View (X-plane, side-to-side) ---
    axes[2].imshow(image_np[:, :, x], cmap='gray', origin='lower', aspect='auto')
    axes[2].axhline(z, color='lime', linewidth=0.8)
    axes[2].axvline(y, color='lime', linewidth=0.8)
    axes[2].scatter(y, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[2].set_title(f"Sagittal (X = {x})")
    axes[2].set_xlabel("Y-axis")
    axes[2].set_ylabel("Z-axis")

    plt.show()

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


def correct_hu_value_if_peak_not_at_zero(data, sigma=7, prominence=1, num_peaks=3, tolerance=100):
    if not isinstance(data, np.ndarray) or data.size == 0:
        print("Warning: Input data is empty or not a numpy array.")
        return False

    flat_data = data.flatten()
    if flat_data.size > 100_000:
        flat_data = np.random.choice(flat_data, size=100_000, replace=False)

    flat_data = np.round(flat_data).astype(int)

    min_val = flat_data.min()
    shifted_data = flat_data - min_val
    
    freq_dist = np.bincount(shifted_data)
    original_values = np.arange(min_val, min_val + len(freq_dist))

    smoothed_freq = gaussian_filter1d(freq_dist.astype(float), sigma=sigma)

    indices, _ = find_peaks(smoothed_freq, prominence=prominence)
    
    if len(indices) == 0:
        print("No peaks found.")
        return False

    heights = smoothed_freq[indices]
    sorted_indices_by_height = indices[np.argsort(heights)[::-1]]
    top_peak_indices = sorted_indices_by_height[:num_peaks]
    top_peak_values = original_values[top_peak_indices]

    # --- 5. Check if a peak is NEAR zero ---
    result = any(abs(val) <= tolerance for val in top_peak_values)

    return result



# --- THIS FUNCTION IS NOW CORRECTED AND RESTORED TO YOUR ORIGINAL ---
def robust_hu_conversion(dicom_dataset):
    pixel_array = dicom_dataset.pixel_array
    
    # --- Part 1: Convert to HU (Your existing logic) ---
    hu_pixels = pixel_array
    if 'PhotometricInterpretation' in dicom_dataset and dicom_dataset.PhotometricInterpretation == "MONOCHROME1":
        max_hu = np.max(hu_pixels)
        min_hu = np.min(hu_pixels)
        hu_pixels = (max_hu + min_hu) - hu_pixels

    # return pixel_array
    if 'RescaleSlope' in dicom_dataset and 'RescaleIntercept' in dicom_dataset:
        slope = float(dicom_dataset.RescaleSlope)
        intercept = float(dicom_dataset.RescaleIntercept)

        hu_conv_pixels = (pixel_array * slope) + intercept

        if not correct_hu_value_if_peak_not_at_zero(hu_conv_pixels):
            hu_conv_pixels = hu_pixels
    else:
        hu_conv_pixels = hu_pixels

    return hu_conv_pixels

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
    if len(unique_slices) < len(uniform_slices): print(f"Warning: Discarded {len(uniform_slices) - len(uniform_slices)} spatially duplicate slices.")
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

# --- REVISED preprocess_cta_scan (without alignment) ---
# This is the ONLY function with changes.
# It now accepts and returns the 'location' string along with the coordinates.

def preprocess_cta_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (0.58, 0.58, 1.2),
    cta_hu_window: tuple = (-200, 500),
    initial_coords_list: Optional[List[dict]] = None,
    sof_tissue_peak_tolerance = 100,
    DEBUG_MODE: bool = False
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[List[dict]]]: # MODIFIED: Return type

    # --- Step 1: Load Initial Full Volume (No change) ---
    reoriented_itk = load_and_reorient_dicom(dicom_folder_path)

    # MODIFIED: This list will now store dictionaries to hold extra info
    aneurysm_physical_points_info = [] 
    if initial_coords_list:
        for coord_info in initial_coords_list:
            point = get_physical_point_from_dicom(
                dicom_folder_path, coord_info['sop_uid'], coord_info['coords_xy']
            )
            # Store the physical point AND its original location string
            aneurysm_physical_points_info.append({
                'physical_point': point,
                'location': coord_info.get('location', None) # Safely get location
            })

    # --- VISUALIZATION POINT 1: "BEFORE" (No change in logic) ---
    if DEBUG_MODE and aneurysm_physical_points_info:
        print("DEBUG: Displaying initial locations on original, reoriented volume...")
        initial_image_np = sitk.GetArrayFromImage(reoriented_itk)
        for i, info in enumerate(aneurysm_physical_points_info):
            initial_physical_point = info['physical_point']
            initial_voxel_coords_xyz = reoriented_itk.TransformPhysicalPointToIndex(initial_physical_point)
            initial_voxel_coords_zyx = initial_voxel_coords_xyz[::-1]
            visualize_location_in_3d(
                image_np=initial_image_np,
                voxel_coords_zyx=initial_voxel_coords_zyx,
                title=f"BEFORE Processing (Aneurysm #{i+1})\nLocation in original scan: {initial_voxel_coords_zyx}"
            )
        del initial_image_np

    # --- Step 2: Pre-processing & Cropping (No change) ---
    clipped_itk = sitk.Clamp(reoriented_itk, sitk.sitkFloat32, -1024, 1000)
    del reoriented_itk
    full_scan_np_view = sitk.GetArrayViewFromImage(clipped_itk)
    neck_cutoff_z = find_neck_cutoff(full_scan_np_view, body_threshold_hu=cta_hu_window[0])
    original_size = clipped_itk.GetSize()
    index = [0, 0, int(neck_cutoff_z)]
    size = [original_size[0], original_size[1], original_size[2] - int(neck_cutoff_z)]
    if size[2] < 5:
        raise RuntimeError(f"Neck cropping failed for {dicom_folder_path}, resulted in {size[2]} slices.")
    head_and_shoulders_itk = sitk.RegionOfInterest(clipped_itk, size=size, index=index)
    del clipped_itk
    head_itk = crop_to_body(head_and_shoulders_itk, air_threshold_hu=-500)
    del head_and_shoulders_itk
    
    # --- STEP 3: ALIGNMENT REMOVED (No change) ---
    aligned_head_itk = head_itk
    identity_transform = sitk.Transform() 
    
    # --- Step 4: Transform Coordinates (Now uses the identity transform) ---
    # MODIFIED: This list will also store dictionaries
    transformed_physical_points_info = []
    if aneurysm_physical_points_info:
        for info in aneurysm_physical_points_info:
            # Applying an identity transform returns the same point
            transformed_point = identity_transform.TransformPoint(info['physical_point']) 
            transformed_physical_points_info.append({
                'physical_point': transformed_point,
                'location': info['location'] # Carry the location string forward
            })

    # --- Step 5: Normalization & Resampling (No change) ---
    normalized_itk = normalize_hu_window(aligned_head_itk, hu_window=cta_hu_window)
    del aligned_head_itk
    final_itk_image = resample_image(normalized_itk, target_spacing=target_spacing)
    del normalized_itk

    # --- Step 6: Final Conversion and Coordinate Calculation ---
    final_image_np = sitk.GetArrayFromImage(final_itk_image)
    
    # MODIFIED: The final list will contain dictionaries
    final_output_list = None
    if transformed_physical_points_info:
        final_output_list = []
        for info in transformed_physical_points_info:
            final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(info['physical_point'])
            final_voxel_coords = final_voxel_coords_xyz[::-1]
            # Create the final output dictionary for this aneurysm
            final_output_list.append({
                'final_coords_zyx': final_voxel_coords,
                'location': info['location']
            })

    # --- VISUALIZATION POINT 2: "AFTER" (No change in logic) ---
    if DEBUG_MODE and final_output_list:
        print("DEBUG: Displaying final transformed locations on processed volume.")
        for i, info in enumerate(final_output_list):
            final_coords_zyx = info['final_coords_zyx']
            visualize_location_in_3d(
                image_np=final_image_np,
                voxel_coords_zyx=final_coords_zyx,
                title=f"AFTER Processing (Aneurysm #{i+1})\nFinal location in processed scan: {final_coords_zyx}"
            )

    # MODIFIED: Return the final list of dictionaries
    return final_image_np, target_spacing, final_output_list