import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider
import numpy as np


def view_3d_volume(volume_array, crosshair_coords=None, cmap='gray', title='3D Volume Viewer'):
    """
    Displays aneurysm locations in separate plots and then provides an
    interactive slice-by-slice viewer for the full volume.
    
    Args:
        volume_array (np.ndarray): The 3D numpy array to display.
        crosshair_coords (list of tuples, optional): A list of (z, y, x) coordinates.
            If provided, a separate 3-panel plot will be generated for each coordinate.
        cmap (str, optional): The colormap for the display.
        title (str, optional): The base title for the plots.
    """
    if crosshair_coords is None:
        crosshair_coords = []
        
    # --- PART 1: STATIC VISUALIZATION FOR EACH ANEURYSM ---
    # This block runs first if there are any coordinates to show.
    if crosshair_coords:
        print(f"--- Displaying {len(crosshair_coords)} Aneurysm Location(s) ---")
        for i, (z, y, x) in enumerate(crosshair_coords):
            aneurysm_title = f"Aneurysm #{i+1} Location: (Z={z}, Y={y}, X={x})"
            
            # Input Validation to prevent crashing
            if not (0 <= z < volume_array.shape[0] and 0 <= y < volume_array.shape[1] and 0 <= x < volume_array.shape[2]):
                print(f"Error: Aneurysm coordinates {(z,y,x)} are out of bounds for image shape {volume_array.shape}. Skipping this plot.")
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(aneurysm_title, fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            linewidth =0.6
            # Axial View
            axes[0].imshow(volume_array[z, :, :], cmap=cmap, origin='lower')
            axes[0].axhline(y, color='lime', linewidth=linewidth)
            axes[0].axvline(x, color='lime', linewidth=linewidth)
            # axes[0].scatter(x, y, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
            axes[0].set_title(f"Axial (Z = {z})"); axes[0].set_xlabel("X-axis"); axes[0].set_ylabel("Y-axis")

            # Coronal View
            axes[1].imshow(volume_array[:, y, :], cmap=cmap, origin='lower', aspect='auto')
            axes[1].axhline(z, color='lime', linewidth=linewidth)
            axes[1].axvline(x, color='lime', linewidth=linewidth)
            # axes[1].scatter(x, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
            axes[1].set_title(f"Coronal (Y = {y})"); axes[1].set_xlabel("X-axis"); axes[1].set_ylabel("Z-axis")

            # Sagittal View
            axes[2].imshow(volume_array[:, :, x], cmap=cmap, origin='lower', aspect='auto')
            axes[2].axhline(z, color='lime', linewidth=linewidth)
            axes[2].axvline(y, color='lime', linewidth=linewidth)
            # axes[2].scatter(y, z, s=100, facecolors='none', edgecolors='lime', linewidth=1.5)
            axes[2].set_title(f"Sagittal (X = {x})"); axes[2].set_xlabel("Y-axis"); axes[2].set_ylabel("Z-axis")

            plt.show()

    # --- PART 2: INTERACTIVE SLICE BROWSER (NO CROSSHAIRS) ---
    print(f"\n--- Initializing Interactive Volume Browser ---")
    
    data_min = float(np.min(volume_array))
    data_max = float(np.max(volume_array))
    step_val = (data_max - data_min) / 256.0

    def show_slice(slice_index, mask_min, mask_max):
        display_slice = volume_array[slice_index].copy()
        mask = (display_slice < mask_min) | (display_slice > mask_max)
        display_slice[mask] = data_min
        
        plt.figure(figsize=(8, 8))
        plt.imshow(display_slice, cmap=cmap, aspect='equal', vmin=data_min, vmax=data_max)
        plt.title(f'{title} - Slice {slice_index}/{volume_array.shape[0] - 1}\nMask Range: [{mask_min:.2f}, {mask_max:.2f}]')
        plt.axis('off')
        plt.show()

    interact(
        show_slice,
        slice_index=IntSlider(
            min=0, max=volume_array.shape[0] - 1, step=1, value=volume_array.shape[0] // 2, description='Slice Index'
        ),
        mask_min=FloatSlider(
            min=data_min, max=data_max, step=step_val, value=data_min, description='Mask Min'
        ),
        mask_max=FloatSlider(
            min=data_min, max=data_max, step=step_val, value=data_max, description='Mask Max'
        )
    )



def display_hu_distribution(final_image_np, title="HU Distribution", window_vals=None):
    """
    Displays the Hounsfield Unit distribution of the image volume.
    """
    flat_pixels = final_image_np.flatten()
    if flat_pixels.size > 1_000_000:
        flat_pixels = np.random.choice(flat_pixels, size=1_000_000, replace=False)
    plt.figure(figsize=(12, 6))
    plt.hist(flat_pixels, bins=256, color='c', label='HU Distribution')
    
    # If window values are provided, plot them as vertical lines
    if window_vals:
        min_val, max_val = window_vals
        plt.axvline(min_val, color='r', linestyle='--', label=f'Window Min: {min_val:.1f}')
        plt.axvline(max_val, color='g', linestyle='--', label=f'Window Max: {max_val:.1f}')
        print(f"Visualizing window: [{min_val:.1f}, {max_val:.1f}]")
        
    plt.title(title)
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()