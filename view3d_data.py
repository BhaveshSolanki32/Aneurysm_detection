import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider
import numpy as np

def view_3d_volume(volume_array, cmap='gray', title='3D Volume Viewer'):

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
            min=0, max=volume_array.shape[0] - 1, step=1, value=volume_array.shape[0] // 2
        ),
        mask_min=FloatSlider(
            min=data_min, max=data_max, step=step_val, value=data_min, description='Mask Min'
        ),
        mask_max=FloatSlider(
            min=data_min, max=data_max, step=step_val, value=data_max, description='Mask Max'
        )
    )

def display_hu_distribution(final_image_np, title = "HU Distribution"):
    flat_pixels = final_image_np.flatten()
    if flat_pixels.size > 1_000_000:
        flat_pixels = np.random.choice(flat_pixels, size=1_000_000, replace=False)
    plt.figure(figsize=(12, 6))
    plt.hist(flat_pixels, bins=256, color='c', label='HU Distribution')
    window_vals = None
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
