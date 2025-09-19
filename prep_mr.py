# Save this as preprocess_mri.py
import SimpleITK as sitk
import numpy as np
import os
import pydicom
import collections
from typing import Tuple, List, Optional

# For Skull Stripping
from deepbrain import Extractor

# --- Visualization Function (Identical to your CT script for consistency) ---
import matplotlib.pyplot as plt

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
        z, y, x = [s // 2 for s in image_np.shape]
        title += f"\nCOORDS OUT OF BOUNDS - Showing center instead"
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Axial View
    axes[0].imshow(image_np[z, :, :], cmap='gray', origin='lower')
    axes[0].axhline(y, color='lime', linewidth=0.8)
    axes[0].axvline(x, color='lime', linewidth=0.8)
    axes[0].scatter(x, y, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[0].set_title(f"Axial (Z = {z})")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")

    # Coronal View
    axes[1].imshow(image_np[:, y, :], cmap='gray', origin='lower', aspect='auto')
    axes[1].axhline(z, color='lime', linewidth=0.8)
    axes[1].axvline(x, color='lime', linewidth=0.8)
    axes[1].scatter(x, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[1].set_title(f"Coronal (Y = {y})")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Z-axis")

    # Sagittal View
    axes[2].imshow(image_np[:, :, x], cmap='gray', origin='lower', aspect='auto')
    axes[2].axhline(z, color='lime', linewidth=0.8)
    axes[2].axvline(y, color='lime', linewidth=0.8)
    axes[2].scatter(y, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
    axes[2].set_title(f"Sagittal (X = {x})")
    axes[2].set_xlabel("Y-axis")
    axes[2].set_ylabel("Z-axis")
    
    plt.show()

# --- Coordinate Helper Function (Identical to your CT script) ---
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

# --- Core MRI Preprocessing Functions ---

def load_dicom_series_manually(dicom_folder_path: str) -> sitk.Image:
    """Robustly loads a DICOM series, handling inconsistencies."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series found in directory: {dicom_folder_path}")

    slices = [pydicom.dcmread(dcm, stop_before_pixels=False) for dcm in dicom_names]
    
    # Filter for readable slices with necessary attributes
    slices_with_pixels = [s for s in slices if hasattr(s, 'pixel_array') and hasattr(s, 'ImagePositionPatient')]
    if not slices_with_pixels:
        raise ValueError(f"No readable slices with pixel data found in {dicom_folder_path}")

    # Ensure uniform slice dimensions
    slice_shapes = [s.pixel_array.shape for s in slices_with_pixels]
    most_common_shape = collections.Counter(slice_shapes).most_common(1)[0][0]
    uniform_slices = [s for s in slices_with_pixels if s.pixel_array.shape == most_common_shape]
    if len(uniform_slices) < len(slices_with_pixels):
        print(f"Warning: Discarded {len(slices_with_pixels) - len(uniform_slices)} slices with inconsistent shapes.")
    
    # Sort slices by spatial location (z-coordinate)
    try:
        uniform_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, KeyError):
        print("Warning: Could not sort by ImagePositionPatient. Falling back to InstanceNumber.")
        uniform_slices.sort(key=lambda x: int(x.InstanceNumber))

    # Stack slices into a 3D numpy array
    image_3d_np = np.stack([s.pixel_array for s in uniform_slices], axis=0).astype(np.float32)
    image_itk = sitk.GetImageFromArray(image_3d_np)
    
    # Set metadata
    first_slice = uniform_slices[0]
    pixel_spacing = first_slice.PixelSpacing
    slice_positions = [s.ImagePositionPatient[2] for s in uniform_slices]
    z_spacing = np.median(np.diff(sorted(slice_positions)))
    
    if 'SliceThickness' in first_slice and z_spacing < 1e-3:
        z_spacing = float(first_slice.SliceThickness)

    image_itk.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(z_spacing)))
    image_itk.SetOrigin(first_slice.ImagePositionPatient)
    
    orientation = first_slice.ImageOrientationPatient
    row_cosines = [float(o) for o in orientation[0:3]]
    col_cosines = [float(o) for o in orientation[3:6]]
    z_dir = np.cross(row_cosines, col_cosines)
    image_itk.SetDirection((*row_cosines, *col_cosines, *z_dir))
    
    return image_itk

def n4_bias_field_correction(itk_image: sitk.Image) -> sitk.Image:
    """Applies N4 bias field correction to an ITK image."""
    print("Applying N4 Bias Field Correction...")
    mask_image = sitk.OtsuThreshold(itk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(itk_image, mask_image)
    return corrected_image

def skull_strip(itk_image: sitk.Image) -> sitk.Image:
    """
    Performs brain extraction (skull stripping) on the image.
    Note: This function converts to NumPy and back, which is required by the library.
    """
    print("Performing skull stripping...")
    image_np = sitk.GetArrayFromImage(itk_image)
    
    # deepbrain's Extractor expects (Height, Width, Depth)
    image_np_transposed = np.transpose(image_np, (1, 2, 0))
    
    extractor = Extractor()
    # The extractor returns a probability map of the brain
    brain_prob_map = extractor.run(image_np_transposed)
    
    # Create a binary mask by thresholding the probability map
    brain_mask_np = (brain_prob_map > 0.5).astype(np.uint8)
    
    # Transpose mask back to (Depth, Height, Width)
    brain_mask_np_transposed = np.transpose(brain_mask_np, (2, 0, 1))
    
    # Convert mask to ITK image and copy metadata from original
    mask_itk = sitk.GetImageFromArray(brain_mask_np_transposed)
    mask_itk.CopyInformation(itk_image)
    
    # Apply the mask to the original image
    mask_filter = sitk.MaskImageFilter()
    brain_only_itk = mask_filter.Execute(itk_image, mask_itk)
    
    return brain_only_itk

def normalize_mri_intensity(
    itk_image: sitk.Image, 
    modality: str
) -> sitk.Image:
    """
    Normalizes MRI intensity based on the modality to make vessels hyperintense.
    - For MRA/T1c, clips outliers and scales to [0, 1].
    - For T2, inverts the signal, clips, and scales to [0, 1].
    """
    print(f"Normalizing intensity for {modality.upper()}...")
    
    stats = sitk.StatisticsImageFilter()
    stats.Execute(itk_image)
    
    if stats.GetMean() == 0 and stats.GetStandardDeviation() == 0:
        print("Warning: Image is empty. Skipping normalization.")
        return itk_image

    image_np = sitk.GetArrayFromImage(itk_image)
    
    # Use only non-zero voxels (i.e., inside the brain mask) for statistics
    non_zero_voxels = image_np[image_np > 0]
    
    if modality.lower() == 't2w':
        # Invert the signal so flow voids (vessels) become bright
        # We find the max value within the brain to invert against
        max_val = np.percentile(non_zero_voxels, 99.9)
        inverted_np = max_val - image_np
        # Ensure the background (zeros) remains zero after inversion
        inverted_np[image_np == 0] = 0
        image_np = inverted_np

    # Clip intensity outliers to make normalization more robust
    # Using 0.5 and 99.5 percentiles of the non-zero voxels
    p_low = np.percentile(non_zero_voxels, 0.5)
    p_high = np.percentile(non_zero_voxels, 99.5)
    
    clipped_np = np.clip(image_np, p_low, p_high)
    
    # Rescale to [0, 1]
    min_val = clipped_np.min()
    max_val = clipped_np.max()
    if max_val > min_val:
        normalized_np = (clipped_np - min_val) / (max_val - min_val)
    else:
        normalized_np = clipped_np * 0 # Avoid division by zero
        
    normalized_itk = sitk.GetImageFromArray(normalized_np.astype(np.float32))
    normalized_itk.CopyInformation(itk_image)
    
    return normalized_itk

def resample_image(
    itk_image: sitk.Image, 
    target_spacing: Tuple[float, float, float]
) -> sitk.Image:
    """Resamples an ITK image to a target voxel spacing."""
    print(f"Resampling image to spacing: {target_spacing}...")
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    new_size = [
        int(round(osz * ospc / tspc)) 
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0) # Black background
    resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(itk_image)


# --- Main Preprocessing Pipeline ---

def preprocess_mri_scan(
    dicom_folder_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    initial_coords_list: Optional[List[dict]] = None,
    DEBUG_MODE: bool = False,
    modality: str = 'MRA'
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[List[dict]]]:
    """
    Main pipeline to preprocess an MRI scan for deep learning.
    
    Args:
        dicom_folder_path: Path to the folder containing DICOM files for one series.
        modality: The MRI modality, one of ['mra', 't1c', 't2w']. This determines
                  how intensity normalization is handled.
        target_spacing: The isotropic voxel spacing for the final output image.
        initial_coords_list: Optional list of aneurysm coordinates to transform.
        DEBUG_MODE: If True, displays visualizations at key steps.

    Returns:
        A tuple containing:
        - The preprocessed 3D image as a NumPy array.
        - The target spacing tuple.
        - The list of transformed aneurysm coordinates in the new image space.
    """
    
    if modality.lower() not in ['mra', 't1c', 't2w']:
        raise ValueError("Modality must be one of 'mra', 't1c', or 't2w'")

    # --- Step 1: Load and Reorient DICOM series ---
    print("Step 1: Loading DICOM series...")
    initial_itk = load_dicom_series_manually(dicom_folder_path)
    # Reorient to a standard orientation (LPS) for consistency
    reoriented_itk = sitk.DICOMOrient(initial_itk, 'LPS')
    
    # Transform initial coordinates to physical space before any modifications
    aneurysm_physical_points_info = [] 
    if initial_coords_list:
        for coord_info in initial_coords_list:
            point = get_physical_point_from_dicom(
                dicom_folder_path, coord_info['sop_uid'], coord_info['coords_xy']
            )
            aneurysm_physical_points_info.append({
                'physical_point': point,
                'location': coord_info.get('location', None)
            })
            
    # --- VISUALIZATION POINT 1: "BEFORE" ---
    if DEBUG_MODE and aneurysm_physical_points_info:
        print("DEBUG: Displaying initial location on original, reoriented volume...")
        initial_image_np = sitk.GetArrayFromImage(reoriented_itk)
        for i, info in enumerate(aneurysm_physical_points_info):
            voxel_coords_xyz = reoriented_itk.TransformPhysicalPointToIndex(info['physical_point'])
            visualize_location_in_3d(
                image_np=initial_image_np,
                voxel_coords_zyx=voxel_coords_xyz[::-1],
                title=f"BEFORE Processing (Aneurysm #{i+1})\nOriginal, Reoriented Volume"
            )
        del initial_image_np

    # --- Step 2: Bias Field Correction ---
    corrected_itk = n4_bias_field_correction(reoriented_itk)
    del reoriented_itk
    
    # --- Step 3: Skull Stripping ---
    brain_itk = skull_strip(corrected_itk)
    del corrected_itk
    
    # --- Step 4: Modality-Specific Intensity Normalization ---
    normalized_itk = normalize_mri_intensity(brain_itk, modality)
    del brain_itk
    
    # --- Step 5: Resampling to Isotropic Resolution ---
    final_itk_image = resample_image(normalized_itk, target_spacing=target_spacing)
    del normalized_itk
    
    # --- Step 6: Final Conversion and Coordinate Calculation ---
    final_image_np = sitk.GetArrayFromImage(final_itk_image)
    
    final_output_list = None
    if aneurysm_physical_points_info:
        final_output_list = []
        for info in aneurysm_physical_points_info:
            # Physical points are unchanged, we just find their new index in the final image
            final_voxel_coords_xyz = final_itk_image.TransformPhysicalPointToIndex(info['physical_point'])
            final_output_list.append({
                'final_coords_zyx': final_voxel_coords_xyz[::-1],
                'location': info['location']
            })

    # --- VISUALIZATION POINT 2: "AFTER" ---
    if DEBUG_MODE and final_output_list:
        print("DEBUG: Displaying final transformed locations on processed volume.")
        for i, info in enumerate(final_output_list):
            visualize_location_in_3d(
                image_np=final_image_np,
                voxel_coords_zyx=info['final_coords_zyx'],
                title=f"AFTER Processing (Aneurysm #{i+1} on {modality.upper()})\nFinal Processed Volume"
            )

    return final_image_np, target_spacing, final_output_list
