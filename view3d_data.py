import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def view_3d_volume(volume_array, cmap='gray', title='3D Volume Viewer'):

    # Interactive display function
    def show_slice(slice_index):
        plt.figure(figsize=(8, 8))  # New figure per slice to avoid overlap
        plt.imshow(volume_array[slice_index], cmap=cmap, aspect='equal')  # Preserves aspect ratio
        plt.title(f'{title} - Slice {slice_index}/{volume_array.shape[0] - 1}')
        plt.axis('off')
        plt.show()

    # Create interactive slider
    interact(show_slice, slice_index=IntSlider(min=0, max=volume_array.shape[0] - 1, step=1, value=volume_array.shape[0] // 2))
