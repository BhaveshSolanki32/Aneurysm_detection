import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

# --- New Imports for this Workflow ---
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- Multiprocessing & Progress Bar ---
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# ==============================================================================
# PART 1: DICOM PROCESSING FUNCTIONS (Re-used from your previous code)
# ==============================================================================
# Your functions basic_hu_conversion, filter_bad_slices_raw, save_scan_array,
# and get_hu_sample are assumed to be here. I've included them for completeness.

def basic_hu_conversion(dicom_dataset):
    pixel_array = dicom_dataset.pixel_array.astype(np.int16) # Use int16 for memory
    hu_pixels = pixel_array
    if 'RescaleSlope' in dicom_dataset and 'RescaleIntercept' in dicom_dataset:
        slope = float(dicom_dataset.RescaleSlope)
        intercept = float(dicom_dataset.RescaleIntercept)
        hu_pixels = (pixel_array * slope) + intercept
    if 'PhotometricInterpretation' in dicom_dataset and dicom_dataset.PhotometricInterpretation == "MONOCHROME1":
        hu_pixels = np.max(hu_pixels) - hu_pixels
    return hu_pixels

def filter_bad_slices_raw(image_3d_hu_list):
    final_hu_stack =[]
    for x in image_3d_hu_list[1:-1]:
        if x.max() < 5000 and x.min() > -5000:
            final_hu_stack.append(x)
    return final_hu_stack

def save_scan_array(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        try:
            filepath = os.path.join(folder_path, filename)
            dcm = pydicom.dcmread(filepath, stop_before_pixels=False)
            hu_slice = basic_hu_conversion(dcm)
            files.append(hu_slice)
        except Exception:
            pass
    if not files: return np.array([])
    files = filter_bad_slices_raw(files)
    if not files: return np.array([])
    return np.stack(files)

def get_hu_sample(uid, root_folder):
    img_path = os.path.join(root_folder, uid)
    if not os.path.isdir(img_path): return None
    array = save_scan_array(img_path).flatten()
    if array.size == 0: return None
    # No longer need to sample to a fixed size, we'll use the whole array
    return array

# ==============================================================================
# PART 1.5: NEW WORKER FUNCTION FOR FEATURE ENGINEERING
# ==============================================================================

def process_uid_for_features(uid, root_folder, hist_bins, hist_range):
    """
    Worker function to process a single UID, extract peak features,
    and return them along with the original histogram.
    """
    try:
        # 1. Get raw HU values and create original histogram
        hu_values = get_hu_sample(uid, root_folder)
        if hu_values is None:
            return uid, None, None
            
        original_hist, bin_edges = np.histogram(hu_values, bins=hist_bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. Smooth the histogram to find major peaks
        # Sigma is a key parameter; a value of 2-5 is good for a 1000-bin histogram
        smoothed_hist = gaussian_filter1d(original_hist.astype(float), sigma=3)

        # 3. Find all prominent peaks
        # Prominence is set relative to the max peak height to be adaptive
        min_prominence = 0.05 * np.max(smoothed_hist)
        peak_indices, properties = find_peaks(smoothed_hist, prominence=min_prominence, width=5)

        # 4. Sort peaks by prominence and take the top 4
        if len(peak_indices) > 0:
            sorted_peak_indices = sorted(zip(peak_indices, properties['prominences']), key=lambda x: x[1], reverse=True)
            top_peaks = sorted_peak_indices[:4]
        else:
            top_peaks = []
            
        # 5. Extract features for each of the top 4 peaks
        feature_list = []
        num_features_per_peak = 6 # Location, Height, Prominence, Width, Mean, Std
        
        for i in range(4): # Loop to ensure we always have 4 "slots"
            if i < len(top_peaks):
                peak_idx, prominence = top_peaks[i]
                
                # Find the index in the original properties dict
                prop_idx = np.where(peak_indices == peak_idx)[0][0]

                # Feature: Location (the HU value)
                loc = bin_centers[peak_idx]
                # Feature: Height (from smoothed histogram)
                height = smoothed_hist[peak_idx]
                # Feature: Prominence
                # prom = properties['prominences'][prop_idx] # Redundant, we already have it
                # Feature: Width
                width = properties['widths'][prop_idx]
                
                # Features: Mean & Std in a window around the peak from ORIGINAL data
                window_radius = 25 # bins
                start_idx = max(0, peak_idx - window_radius)
                end_idx = min(len(bin_centers) - 1, peak_idx + window_radius)
                
                # Get HU values that fall in this bin window
                hu_min = bin_centers[start_idx]
                hu_max = bin_centers[end_idx]
                values_in_window = hu_values[(hu_values >= hu_min) & (hu_values <= hu_max)]
                
                mean_in_window = np.mean(values_in_window) if values_in_window.size > 0 else loc
                std_in_window = np.std(values_in_window) if values_in_window.size > 1 else 0

                feature_list.extend([loc, height, prominence, width, mean_in_window, std_in_window])
            else:
                # Pad with zeros if fewer than 4 peaks are found
                feature_list.extend([0.0] * num_features_per_peak)
                
        return uid, feature_list, original_hist

    except Exception as e:
        # print(f"Error in worker for {uid}: {e}") # Uncomment for debugging
        return uid, None, None

# ==============================================================================
# PART 2: MAIN WORKFLOW
# ==============================================================================

def main():
    # --- 1. SETUP ---
    train_df = pd.read_csv(r'D:\projects\aneurysm\rsna-intracranial-aneurysm-detection\train.csv')
    df_ct = train_df[train_df['Modality'] == 'CTA']
    root_folder = r'D:\projects\aneurysm\rsna-intracranial-aneurysm-detection\series'
    all_uids = df_ct['SeriesInstanceUID'].unique()[:600]
    
    # --- 2. FEATURE ENGINEERING (PARALLELIZED) ---
    print("\nStep 1: Processing DICOMs and engineering peak features in parallel...")
    HIST_BINS = 1000
    HIST_RANGE = (-2000, 3000)
    
    worker_func = partial(process_uid_for_features, root_folder=root_folder, hist_bins=HIST_BINS, hist_range=HIST_RANGE)
    num_processes = max(1, cpu_count() -6 ) # Leave some cores for system responsiveness
    print(f"Using {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_func, all_uids), total=len(all_uids)))

    # --- 3. ASSEMBLE DATAFRAMES AND SAVE FEATURES ---
    print("\nStep 2: Assembling DataFrames and saving engineered features...")
    
    # Create column names for the feature DataFrame
    feature_names = []
    for i in range(4):
        p = f'p{i+1}'
        feature_names.extend([f'{p}_loc', f'{p}_height', f'{p}_prom', f'{p}_width', f'{p}_mean', f'{p}_std'])

    processed_features = []
    original_hists = []
    processed_uids = []
    for uid, features, hist in results:
        if features is not None and hist is not None:
            processed_features.append(features)
            original_hists.append(hist)
            processed_uids.append(uid)

    # DataFrame with engineered features (for clustering)
    features_df = pd.DataFrame(processed_features, columns=feature_names)
    features_df.insert(0, 'scan_id', processed_uids)
    
    # DataFrame with original histograms (for final output)
    original_hist_df = pd.DataFrame(original_hists, columns=[f'hu_bin_{i}' for i in range(HIST_BINS)])
    original_hist_df.insert(0, 'scan_id', processed_uids)
    
    # Save the engineered features
    features_df.to_csv('scan_peak_features.csv', index=False)
    print(f"Successfully created feature matrix with shape: {features_df.shape}")
    print("Engineered features saved to 'scan_peak_features.csv'")
    
    if features_df.empty:
        print("No features could be extracted. Exiting.")
        return

    # --- 4. PREPARE FOR DBSCAN ---
    print("\nStep 3: Preparing for DBSCAN clustering...")
    features_for_clustering = features_df.drop('scan_id', axis=1)
    
    # Standardization is ESSENTIAL for DBSCAN
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)

    # Find the optimal 'eps' using the k-distance plot method
    # MinPts is often set to 2 * num_dimensions. Here, 2*24 is too high.
    # A value between 5 and 15 is a robust choice for this dataset size.
    MIN_PTS = 8
    print(f"Calculating k-distance plot to find optimal 'eps' (for MinPts = {MIN_PTS})...")
    neighbors = NearestNeighbors(n_neighbors=MIN_PTS)
    neighbors_fit = neighbors.fit(features_scaled)
    distances, indices = neighbors_fit.kneighbors(features_scaled)
    
    sorted_distances = np.sort(distances[:, MIN_PTS-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.title('k-Distance Graph')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'{MIN_PTS}-th Nearest Neighbor Distance (eps)')
    plt.grid(True)
    print("--> Please inspect the plot to find the 'elbow'. This is your optimal 'eps' value.")
    print("--> Close the plot window to continue the script.")
    plt.show()
    
    # Prompt the user for the chosen eps value
    try:
        chosen_eps = float(input("Enter the chosen 'eps' value from the plot's elbow: "))
    except ValueError:
        print("Invalid input. Using a default value of 2.5")
        chosen_eps = 2.5 # A fallback default

    # --- 5. RUN DBSCAN CLUSTERING ---
    print(f"\nStep 4: Running DBSCAN with eps={chosen_eps} and MinPts={MIN_PTS}...")
    dbscan = DBSCAN(eps=chosen_eps, min_samples=MIN_PTS)
    cluster_labels = dbscan.fit_predict(features_scaled)

    original_hist_df['cluster'] = cluster_labels
    
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers_found = np.sum(cluster_labels == -1)
    print(f"DBSCAN finished. Found {n_clusters_found} clusters and {n_outliers_found} outliers.")

    # --- 6. SAVE CLUSTERED RESULTS ---
    print("\nStep 5: Saving clustered original histograms...")
    output_dir = 'scan_clusters_dbscan'
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster_id in sorted(set(cluster_labels)):
        df_cluster = original_hist_df[original_hist_df['cluster'] == cluster_id]
        
        if cluster_id == -1:
            filename = 'outliers_data.csv'
            print(f"Outliers: Found {len(df_cluster)} samples. Saved to '{os.path.join(output_dir, filename)}'")
        else:
            filename = f'cluster_{cluster_id}_data.csv'
            print(f"Cluster {cluster_id}: Found {len(df_cluster)} samples. Saved to '{os.path.join(output_dir, filename)}'")
            
        df_cluster.drop('cluster', axis=1).to_csv(os.path.join(output_dir, filename), index=False)
        
    print("\nProcess complete!")


if __name__ == '__main__':
    # You may need to install/update:
    # pip install numpy pandas pydicom scikit-learn scipy matplotlib tqdm
    main()