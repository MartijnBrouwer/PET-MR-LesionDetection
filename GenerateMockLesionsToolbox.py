# Generate mock data
# =============================================================================================
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
import joblib
import random
import matplotlib.pyplot as plt
import os
import nibabel as nib
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndimage


def pca_gmm_sampling_with_plot(pca, pca_features, real_lesion_path, num_samples):
    # Process real lesion data
    real_lesion_samples = process_real_lesion(real_lesion_path)

    # Apply PCA transformation to the real lesion data
    real_pca_features = pca.transform(real_lesion_samples)

    # Fit GMM on the PCA-transformed features
    gmm = GaussianMixture(n_components=5, covariance_type='full')
    gmm.fit(real_pca_features)

    # Sample new points from the GMM
    gmm_samples, _ = gmm.sample(num_samples)

    # Plot the GMM distribution in 2D (assuming 2 PCA components)
    if real_pca_features.shape[1] == 2:  # Only plot if it's 2D PCA for visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot of the original real data
        ax.scatter(real_pca_features[:, 0], real_pca_features[:, 1], alpha=0.7, c='blue', edgecolors='k')

        # Plot the sampled GMM points
        ax.scatter(gmm_samples[:, 0], gmm_samples[:, 1], alpha=0.7, c='red', edgecolors='k', marker='x')

        # Plot the GMM components as ellipses
        for i in range(gmm.n_components):
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]
            
            # Create an ellipse for the covariance matrix of each Gaussian
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            w = w.T
            
            angle = np.arctan(w[0, 1] / w[0, 0]) * 180.0 / np.pi
            angle = 180.0 if angle < 0 else angle
            ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color='red', alpha=0.5)
            ax.add_patch(ell)

        ax.set_title("PCA Components with GMM Sampling")
        ax.set_xlabel("First PCA Component")
        ax.set_ylabel("Second PCA Component")
        plt.grid(True)
        plt.show()

    # Plot in 3D with ellipsoids
    elif real_pca_features.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the original real data
        ax.scatter(real_pca_features[:, 0], real_pca_features[:, 1], real_pca_features[:, 2], 
                   alpha=0.7, c='blue', edgecolors='k', label="Real Lesions")

        # Plot the sampled GMM points
        ax.scatter(gmm_samples[:, 0], gmm_samples[:, 1], gmm_samples[:, 2], 
                   alpha=0.7, c='red', edgecolors='k', marker='x', label="GMM Samples")

        # Plot the GMM components as ellipsoids
        for i in range(gmm.n_components):
            plot_ellipsoid(ax, gmm.means_[i], gmm.covariances_[i])

        ax.set_title("PCA Components with GMM Sampling (3D)")
        ax.set_xlabel("First PCA Component")
        ax.set_ylabel("Second PCA Component")
        ax.set_zlabel("Third PCA Component")
        ax.legend()
        plt.show()
    
    return gmm_samples

def process_real_lesion(real_lesion_path):
    lesion_samples = []
    lesion_files = [f for f in os.listdir(real_lesion_path) if f.endswith('.nii.gz')]
    if not lesion_files:
        raise FileNotFoundError(f"No .nii.gz files found in {real_lesion_path}")

    # print(f"Found {len(lesion_files)} real lesion files. Processing...")
    for i, lesion_file in enumerate(lesion_files):
        lesion_file_path = os.path.join(real_lesion_path, lesion_file)
        # print(f"Processing real lesion {i+1}/{len(lesion_files)}: {lesion_file}")
        
        # Load lesion data and resize
        lesion_data = nib.load(lesion_file_path).get_fdata()
        lesion_samples.append(lesion_data.flatten())

    # Min-Max normalization for each lesion (row) based on its own min and max
    min_vals = np.min(lesion_samples, axis=1, keepdims=True)  # Min value for each lesion (row)
    max_vals = np.max(lesion_samples, axis=1, keepdims=True)  # Max value for each lesion (row)

    # Normalize each lesion (row) independently
    lesion_samples = (lesion_samples - min_vals) / (max_vals - min_vals)

    return lesion_samples

def preprocess_lesions_with_pca(real_lesion_path, pca, pca_features, lesion_shape=(24, 24, 24), pca_components=100):
    # Calculate and display explained variance
    # cumulative_variance = np.sum(pca.explained_variance_ratio_) * 100
    # print(f"Selected 100 components explain {cumulative_variance:.2f}% of the variance in the data.")

    # Display individual component variance
    # for i, variance in enumerate(pca.explained_variance_ratio_):
        # print(f"Component {i+1} explains {variance * 100:.2f}% of the variance")

    kde_samples = pca_gmm_sampling_with_plot(pca, pca_features, real_lesion_path, num_samples=1)
    return kde_samples

def pca_kde_sampling_with_plot(pca, pca_features, real_lesion_path, num_samples):

    real_lesion_samples = process_real_lesion(real_lesion_path)

    real_pca_features = pca.transform(real_lesion_samples)

    # Fit KDE for each PCA component
    # print("Fitting KDE models for PCA components...")
    kde_samples = []
    for i in range(pca_features.shape[1]):
        # Extract values of the i-th PCA component
        component_values = real_pca_features[:, i]

        # Fit KDE to the component
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(component_values[:, np.newaxis])

        # Sample new values for this component
        samples = kde.sample(num_samples).flatten()
        kde_samples.append(samples)

        # # Plot the KDE distribution for the first 5 components
        # x_d = np.linspace(min(component_values) - 0.5, max(component_values) + 0.5, 1000)
        # log_dens = kde.score_samples(x_d[:, np.newaxis])
        # plt.figure(figsize=(6, 4))
        # plt.plot(x_d, np.exp(log_dens), label=f"Component {i+1} KDE")
        # plt.hist(component_values, bins=20, density=True, alpha=0.5, label="Histogram")
        # plt.title(f"KDE of PCA Component {i+1}")
        # plt.xlabel("Value")
        # plt.ylabel("Density")
        # plt.legend()
        # plt.grid()
        # plt.show()

    kde_samples = np.stack(kde_samples, axis=1)

    return kde_samples


def plot_lesion_top_values_histogram_with_kde(lesion_values, bandwidth):

    # Compute KDE
    kde = gaussian_kde(lesion_values, bw_method='scott')
    kde.set_bandwidth(kde.factor * bandwidth)
    kde_x = np.linspace(min(lesion_values), max(lesion_values), 500)
    kde_y = kde(kde_x)

    return kde

def sample_from_kde(kde, num_samples=1):
    """
    Sample values from a KDE distribution.

    Parameters:
        kde (gaussian_kde): The KDE object fitted to lesion values.
        num_samples (int): Number of samples to draw.

    Returns:
        numpy.ndarray: Random samples from the KDE.
    """
    return kde.resample(num_samples).flatten()

def sample_z_level_from_kde(kde, num_samples=1):
    """Sample a z-level from the KDE distribution."""
    return kde.resample(num_samples).flatten()

def plot_pca_variance_explained(pca):
    """
    Plot the explained variance ratio for each PCA component.

    Parameters:
        pca (sklearn.decomposition.PCA): The PCA object containing the explained variance data.
    """
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
            alpha=0.7, label='Individual Component')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 
             where='mid', linestyle='--', color='red', label='Cumulative Variance')

    plt.title('PCA Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Generate a lesion
pca_model_path = '//cifs.research.erasmusmc.nl/nuge0001/MartijnB/pca_model_real_lesions.joblib'
pca_features_path = '//cifs.research.erasmusmc.nl/nuge0001/MartijnB/pca_features_real_lesions.joblib'
pca = joblib.load(pca_model_path)
pca_features = joblib.load(pca_features_path)

# Call the function with your PCA object
plot_pca_variance_explained(pca)

def plot_lesion_3d(lesion_array):
    """ Plot the extracted lesion in 3D with a colorbar. """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    nonzero_voxels = np.argwhere(lesion_array > 0)
    values = lesion_array[lesion_array > 0]
    
    sc = ax.scatter(nonzero_voxels[:, 0], nonzero_voxels[:, 1], nonzero_voxels[:, 2], 
                    c=values, cmap='rainbow', marker='o')
    ax.set_title('3D Visualization of Extracted Lesion')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_zlim(0, 12)
    fig.colorbar(sc, ax=ax, label='Voxel Intensity')
    plt.show()

image_dir = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\InsertionImages\images"
mask_dir = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\InsertionImages\masks"
real_lesion_path = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\ExtractedLesions"

# Load KDE distribution
kde_distribution_path = r'\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\kde_distribution.npy'
kde_data = np.load(kde_distribution_path, allow_pickle=True).item()
z_range = kde_data['z_range']
kde_values = kde_data['kde_values']

# Define function to sample z-level from the KDE distribution
def sample_z_from_kde(kde_values, z_range):
    # Use the KDE values to sample a z-level. We can use np.random.choice for sampling.
    z_level = np.random.choice(z_range, p=kde_values / kde_values.sum())  # Normalize the KDE
    return z_level

for i in range(10):
    # Generate lesion using PCA and KDE
    lesion_features = preprocess_lesions_with_pca(real_lesion_path, pca, pca_features)
    inverse_pca_flattened = pca.inverse_transform(lesion_features[0])
    inverse_pca_lesion = inverse_pca_flattened.reshape((12,12,12))
    inverse_pca_lesion[inverse_pca_lesion < 0.65] = 0  # Threshold for reconstruction artifacts
    lesion_10 = inverse_pca_lesion[1:11,1:11,1:11]
    plot_lesion_3d(lesion_10) # Show what inserted lesion looks like
    lesion_10_sitk = sitk.GetImageFromArray(lesion_10)
    # sitk.WriteImage(lesion_10_sitk, fr'C:\Users\r106186\Desktop\MockLesions\Mock_lesion_{i}.nii.gz') # Save lesion