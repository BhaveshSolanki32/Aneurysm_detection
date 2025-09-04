import os
import numpy as np
import pandas as pd
import pydicom

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# New imports for multiprocessing and progress bar
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# ==============================================================================
# PART 1: YOUR DICOM PROCESSING FUNCTIONS (No changes needed here)
# ==============================================================================
# These functions are self-contained and work on a single slice or scan,
# so they are perfectly fine to be called by multiple processes.

def basic_hu_conversion(dicom_dataset):
    """Converts pixel data in a DICOM dataset to Hounsfield Units (HU)."""
    pixel_array = dicom_dataset.pixel_array
    hu_pixels = pixel_array

    if 'RescaleSlope' in dicom_dataset and 'RescaleIntercept' in dicom_dataset:
        slope = float(dicom_dataset.RescaleSlope)
        intercept = float(dicom_dataset.RescaleIntercept)
        hu_pixels = (pixel_array * slope) + intercept

    if 'PhotometricInterpretation' in dicom_dataset and dicom_dataset.PhotometricInterpretation == "MONOCHROME1":
        max_hu = np.max(hu_pixels)
        min_hu = np.min(hu_pixels)
        hu_pixels = (max_hu + min_hu) - hu_pixels
        
    return hu_pixels

def filter_bad_slices_raw(image_3d_hu_list):
    """Filters out slices with HU values outside a plausible range."""
    final_hu_stack =[]
    for x in image_3d_hu_list[1:-1]:
        if x.max() < 5000 and x.min() > -5000:
            final_hu_stack.append(x)
    return final_hu_stack

def save_scan_array(folder_path):
    """Reads all DICOM files in a folder, converts to HU, filters, and stacks them."""
    files = []
    for filename in os.listdir(folder_path):
        try:
            filepath = os.path.join(folder_path, filename)
            dcm = pydicom.dcmread(filepath)
            hu_slice = basic_hu_conversion(dcm)
            files.append(hu_slice)
        except Exception as e:
            pass

    if not files: return np.array([])
    files = filter_bad_slices_raw(files)
    if not files: return np.array([])
    volume = np.stack(files)
    return volume

def get_hu_sample(uid, root_folder):
    """Processes a scan UID to get a flattened, sampled array of HU values."""
    img_path = os.path.join(root_folder, uid)
    if not os.path.isdir(img_path): return None
    array = save_scan_array(img_path).flatten()
    if array.size == 0: return None
    if array.size > 100_000:
        dist = np.random.choice(array, size=100_000, replace=False)
    else:
        dist = np.random.choice(array, size=100_000, replace=True)
    return dist

# ==============================================================================
# PART 1.5: NEW WORKER FUNCTION FOR MULTIPROCESSING
# ==============================================================================

def process_uid(uid, root_folder, hist_bins, hist_range):
    """
    Worker function to process a single UID.
    This function will be executed in a separate process.
    It takes a UID, processes it, and returns the UID and its feature histogram.
    Returning the UID is crucial to link the feature back to its source.
    """
    try:
        hu_sample = get_hu_sample(uid, root_folder)
        if hu_sample is not None:
            hist, _ = np.histogram(hu_sample, bins=hist_bins, range=hist_range)
            return uid, hist # Return a tuple of (uid, features)
    except Exception as e:
        # It's good practice to log errors, but for now, we'll just let it fail silently
        # print(f"\nError processing UID {uid} in a worker process: {e}")
        pass
    return uid, None # Return None for features if processing fails

# ==============================================================================
# PART 2: MAIN WORKFLOW (Refactored for Parallel Processing)
# ==============================================================================

def main():
    # --- 1. SETUP ---
    train_df = pd.read_csv(r'D:\projects\aneurysm\rsna-intracranial-aneurysm-detection\train.csv')
    df_ct = train_df[train_df['Modality'] == 'CTA']
    root_folder = r'D:\projects\aneurysm\rsna-intracranial-aneurysm-detection\series'
    all_uids = df_ct['SeriesInstanceUID'].__array__()[:600]
    
    # --- 2. DATA PREPARATION: FROM DICOM TO FEATURE MATRIX (PARALLELIZED) ---
    print("\nProcessing DICOM files and building feature matrix using multiprocessing...")
    
    # Define constants for the worker function
    HIST_BINS = 1000
    HIST_RANGE = (-2000, 3000)
    
    # Use functools.partial to create a new function with some arguments pre-filled.
    # This is necessary because pool.map only accepts a function with a single iterable argument.
    worker_func = partial(process_uid, root_folder=root_folder, hist_bins=HIST_BINS, hist_range=HIST_RANGE)
    
    # The number of processes to use. cpu_count() is a good default.
    num_processes = cpu_count()-6
    print(f"Using {num_processes} processes for parallel execution.")

    # A Pool of processes will run our worker function on the list of UIDs.
    # The `with` statement ensures the pool is properly closed.
    with Pool(processes=num_processes) as pool:
        # We use pool.imap() with tqdm for a live progress bar.
        # pool.map() would block until all results are done, so the progress bar wouldn't update.
        # list() is used to force the lazy imap iterator to compute everything.
        results = list(tqdm(pool.imap(worker_func, all_uids), total=len(all_uids)))

    # --- Post-processing the parallel results ---
    all_scan_features = []
    processed_uids = []
    for uid, hist in results:
        if hist is not None:
            all_scan_features.append(hist)
            processed_uids.append(uid)
            
    df = pd.DataFrame(all_scan_features, columns=[f'hu_bin_{i}' for i in range(HIST_BINS)])
    df.insert(0, 'scan_id', processed_uids)
    print("\n\nFeature matrix created successfully.")
    print(f"Initial DataFrame shape: {df.shape}")
    
    if df.empty:
        print("DataFrame is empty. Cannot proceed. Exiting.")
        return

    # --- 3. STANDARDIZE ---
    print("\nStandardizing features...")
    features = df.drop('scan_id', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # --- 4. NEW STEP: OUTLIER REMOVAL WITH ISOLATION FOREST ---
    print("\nApplying Isolation Forest to detect outliers...")
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    outlier_predictions = iso_forest.fit_predict(features_scaled)
    
    is_inlier = outlier_predictions == 1
    df_inliers = df[is_inlier].copy()
    df_outliers = df[~is_inlier].copy()
    features_scaled_inliers = features_scaled[is_inlier]
    
    print(f"Found {len(df_outliers)} outliers and {len(df_inliers)} inliers.")

    # --- 5. PCA on CLEANED DATA ---
    print("\nApplying PCA for dimensionality reduction on inlier data...")
    if len(df_inliers) < 2:
        print("Not enough inliers to perform PCA and clustering. Exiting.")
        return
        
    N_COMPONENTS = 30
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    features_pca_inliers = pca.fit_transform(features_scaled_inliers)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"{N_COMPONENTS} components explain {explained_variance:.2%} of the variance.")

    # --- 6. K-MEANS CLUSTERING on CLEANED DATA ---
    print("\nApplying K-Means clustering on inlier data...")
    K = 5
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(features_pca_inliers)
    df_inliers['cluster'] = cluster_labels

    # --- 7. SAVE RESULTS ---
    print("\nSaving each cluster and outliers into separate CSV files...")
    output_dir = 'scan_clusters_with_outlier_removal'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(K):
        df_cluster = df_inliers[df_inliers['cluster'] == i]
        output_filename = os.path.join(output_dir, f'cluster_{i}_data.csv')
        df_cluster.drop('cluster', axis=1).to_csv(output_filename, index=False)
        print(f"Cluster {i}: Found {len(df_cluster)} samples. Saved to '{output_filename}'")
        
    if not df_outliers.empty:
        outlier_filename = os.path.join(output_dir, 'outliers_data.csv')
        df_outliers.to_csv(outlier_filename, index=False)
        print(f"Outliers: Found {len(df_outliers)} samples. Saved to '{outlier_filename}'")
        
    print("\nProcess complete!")

# This is CRITICAL for multiprocessing on Windows and macOS.
# It prevents child processes from re-importing and re-executing the script's main code.
if __name__ == '__main__':
    # You may need to install tqdm: pip install tqdm
    main()