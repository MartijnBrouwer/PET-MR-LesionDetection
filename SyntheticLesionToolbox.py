'''
This code contains a synthetic lesion generation code. The variable "lesion" can be easily saved as NiFTI-file.
The code has been written in such a way that the generated lesions are connected and the intensity is spread according to Gaussian distributions.
Author: @MartijnBrouwer
'''

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import random
import matplotlib.pyplot as plt
import imageio

def translate_to_center(data):
    """
    Translate the lesion to the center of the grid based on its center of mass.

    Parameters:
    - data: 3D array representing the lesion.

    Returns:
    - Translated 3D array.
    """
    # Compute the center of mass weighted by voxel intensities
    center_of_mass = ndimage.center_of_mass(data)
    
    # Compute translation to center the lesion in the grid
    grid_center = np.array(data.shape) // 2
    shift = grid_center - np.round(center_of_mass).astype(int)

    # Translate the lesion
    translated_data = ndimage.shift(data, shift, order=1, mode='constant', cval=0)
    return translated_data

def GenerateSyntheticLesion(num_bulges=3, mean_voxels=50, 
                            x_dim=10, y_dim=10, z_dim=10, 
                            radius_range=(5, 15), smoothness=2, spread=5, 
                            decay_range=(2, 8), noise_scale=0.5, std_voxels=0.3,
                            salt_and_pepper_amount=0.01, salt_vs_pepper=0.5):
    """
    Generate lesions in 3D space with noise and translation to center.
    """
    # Create a 3D grid of coordinates
    x, y, z = np.indices((x_dim, y_dim, z_dim))
    lesion_field = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    center = np.array([x_dim // 2, y_dim // 2, z_dim // 2])
    ball_centers = []

    for i in range(num_bulges):
        if i == 0:
            ball_center = center
        else:
            ball_center = random.choice(ball_centers) + np.random.randint(-spread, spread, size=3)
            ball_center = np.clip(ball_center, 0, [x_dim - 1, y_dim - 1, z_dim - 1])

        radius = random.uniform(*radius_range)
        offset = np.random.uniform(-radius / 2, radius / 2, size=3)
        ball_center += offset.astype(int)
        ball_center = np.clip(ball_center, 0, [x_dim - 1, y_dim - 1, z_dim - 1])

        distance_squared = ((x - ball_center[0]) ** 2 + (y - ball_center[1]) ** 2 + (z - ball_center[2]) ** 2)
        decay = random.uniform(*decay_range)
        direction_vector = np.random.randn(3)
        direction_vector /= np.linalg.norm(direction_vector)

        bulge_distance = np.abs((x - ball_center[0]) * direction_vector[0] + 
                                (y - ball_center[1]) * direction_vector[1] + 
                                (z - ball_center[2]) * direction_vector[2])
        lesion = np.exp(-bulge_distance ** 2 / (2 * (radius ** decay))) * (distance_squared < (radius**2)).astype(float)

        noise = np.random.normal(loc=0.0, scale=noise_scale, size=lesion.shape)
        lesion += noise
        lesion = np.clip(lesion, 0, None)

        current_voxel_count = np.sum(lesion > 0)
        if current_voxel_count > 0:
            scaling_factor = mean_voxels / current_voxel_count
            lesion *= scaling_factor

        lesion_field = merge_lesions(lesion_field, lesion)
        ball_centers.append(ball_center)

    smoothed_field = ndimage.gaussian_filter(lesion_field, sigma=smoothness)
    smoothed_field = (smoothed_field - np.min(smoothed_field)) / (np.max(smoothed_field) - np.min(smoothed_field))
    smoothed_field = np.power(smoothed_field, 2)

    target_voxel_count = int(np.clip(np.random.normal(mean_voxels, mean_voxels * std_voxels), 100, None))
    threshold = np.percentile(smoothed_field, 100 * (1 - target_voxel_count / np.prod(smoothed_field.shape)))
    smoothed_field[smoothed_field < threshold] = 0

    if np.sum(smoothed_field > 0) < 25:
        threshold = np.percentile(smoothed_field, 100 * (1 - 100 / np.prod(smoothed_field.shape)))
        smoothed_field[smoothed_field < threshold] = 0

    # Translate to the center of the grid
    translated_field = translate_to_center(smoothed_field)

    voxel_count = np.sum(translated_field > 0)
    return translated_field, voxel_count

def merge_lesions(field, new_lesion):
    """
    Merge two scalar fields (the current field and the new lesion's field).
    
    Parameters:
    - field: The existing 3D field of lesions.
    - new_lesion: The new lesion's intensity field to be merged.
    
    Returns:
    - The updated scalar field after merging the new lesion.
    """
    # Combine the new lesion with the existing field using maximum intensity at each voxel
    merged_field = np.maximum(field, new_lesion)
    return merged_field

def visualize_lesion_3d(smudge, voxel_count):
    """
    Visualize the generated 3D lesions using a scatter plot, mapping intensity to colors with a colorbar.
    
    Parameters:
    - smudge: The 3D array of lesion intensities.
    - voxel_count: The number of non-zero voxels.
    """
    # Get the coordinates of non-zero values
    coords = np.argwhere(smudge > 0)
    intensities = smudge[smudge > 0]

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot with intensity values mapped to color
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=intensities, cmap='rainbow', alpha=0.8, vmin=0, vmax=1)

    # Fix colorbar range to [0, 1] for consistent coloring
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Intensity')

    # Set the title to include the voxel count
    ax.set_title(f'Number of Voxels: {voxel_count}')

    # Adjust the scale and view
    ax.set_xlim(0, smudge.shape[0])
    ax.set_ylim(0, smudge.shape[1])
    ax.set_zlim(0, smudge.shape[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()

# Create the lesion
lesion, voxel_count = GenerateSyntheticLesion(
            num_bulges=3,
            mean_voxels=25,
            x_dim=10,
            y_dim=10,
            z_dim=10,
            radius_range=(4, 8),
            smoothness=random.uniform(1, 2),
            spread=random.randint(3, 6),
            decay_range=(0.01, 0.5),
            noise_scale=random.uniform(0, 1),
            std_voxels=random.uniform(0, 4)
        )

# Visualize the lesion
visualize_lesion_3d(lesion, voxel_count)