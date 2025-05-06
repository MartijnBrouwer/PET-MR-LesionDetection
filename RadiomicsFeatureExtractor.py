import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, marching_cubes
from scipy.ndimage import binary_erosion, center_of_mass, gaussian_gradient_magnitude, gaussian_filter
from scipy.spatial import ConvexHull, distance
from skimage.measure import find_contours
from skimage.feature import local_binary_pattern

def create_spherical_lesion_with_intensity_fluctuations(voxel_count=1000, shape=(100, 100, 100)):
    lesion = np.zeros(shape, dtype=np.float32)
    
    radius = int(((3 * voxel_count) / (4 * np.pi)) ** (1/3))
    center = np.array(shape) // 2
    z, y, x = np.indices(shape)
    sphere = (x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2 <= radius**2
    
    voxel_indices = np.nonzero(sphere)
    if len(voxel_indices[0]) > voxel_count:
        indices = np.random.choice(len(voxel_indices[0]), size=voxel_count, replace=False)
        lesion[voxel_indices[0][indices], voxel_indices[1][indices], voxel_indices[2][indices]] = np.random.uniform(0, 255, size=voxel_count)
    else:
        lesion[sphere] = np.random.uniform(0, 255, size=np.sum(sphere))  # Assign random intensities

    return lesion

def extract_features_from_lesion(lesion):
    first_order_features = extract_first_order_features(lesion)
    shape_features = extract_shape_features(lesion)
    orientation_features = extract_orientation_features(lesion)
    intensity_features = extract_intensity_features(lesion)
    
    # New features
    mesh_verts, mesh_faces = create_mesh(lesion > 0)
    local_curvature_mean, local_curvature_std = compute_principal_curvature(lesion)
    spatial_correlation = calculate_spatial_correlation(lesion)
    neighborhood_relations_mean, neighborhood_relations_std = calculate_neighborhood_intensity_relations(lesion)
    glcm_contrast, glcm_dissimilarity = extract_glcm_features(lesion)
    lbp_hist = extract_lbp_features(lesion)
    aspect_ratio = calculate_aspect_ratio(shape_features)

    # Combine all features into one flattened array
    features = np.array([
        # First order features
        first_order_features['mean_intensity'],
        first_order_features['std_intensity'],
        first_order_features['min_intensity'],
        first_order_features['max_intensity'],
        first_order_features['energy'],
        first_order_features['entropy'],
        first_order_features['skewness'],
        first_order_features['kurtosis'],

        # Shape features
        shape_features['lesion_volume'],
        shape_features['surface_area'],
        shape_features['bounding_box_volume'],
        shape_features['solidity'],
        shape_features['convexity'],
        shape_features['compactness'],
        shape_features['major_axis_length'],
        shape_features['minor_axis_length'],
        shape_features['least_axis_length'],
        shape_features['max_3D_diameter'],
        shape_features['surface_volume_ratio'],
        shape_features['sphericity'],
        shape_features['concave_points_count'],
        shape_features['fractal_dimension'],
        
        # Orientation features
        orientation_features['COM_x'],
        orientation_features['COM_y'],
        orientation_features['COM_z'],
        orientation_features['COM_index_x'],
        orientation_features['COM_index_y'],
        orientation_features['COM_index_z'],

        # Intensity features
        intensity_features['hist_mean'],
        intensity_features['hist_std'],
        intensity_features['hist_peak_intensity'],
        intensity_features['num_nonzero_bins'],
        intensity_features['percentile_10'],
        intensity_features['percentile_25'],
        intensity_features['percentile_50'],
        intensity_features['percentile_75'],
        intensity_features['percentile_90'],
        intensity_features['intensity_range'],
        intensity_features['intensity_variance'],
        intensity_features['coefficient_of_variation'],
        intensity_features['IQR'],
        intensity_features['intensity_hotspots_count'],

        # New features
        aspect_ratio,
        local_curvature_mean,
        glcm_contrast,
        glcm_dissimilarity,
        neighborhood_relations_mean,
        neighborhood_relations_std
    ])
    
    return features

def extract_first_order_features(lesion):
    features = {
        'mean_intensity': np.mean(lesion),
        'std_intensity': np.std(lesion),
        'min_intensity': np.min(lesion),
        'max_intensity': np.max(lesion),
        'energy': np.sum(lesion ** 2),
        'entropy': -np.sum(lesion * np.log2(lesion + 1e-10)),
        'skewness': (np.mean((lesion - np.mean(lesion)) ** 3) / (np.std(lesion) ** 3)) if np.std(lesion) > 0 else 0,
        'kurtosis': (np.mean((lesion - np.mean(lesion)) ** 4) / (np.std(lesion) ** 4) - 3) if np.std(lesion) > 0 else 0,
    }
    return features

def extract_shape_features(lesion):
    features = {}
    binary_lesion = lesion > 0
    labeled_lesion = label(binary_lesion)
    props = regionprops(labeled_lesion)

    if props:
        lesion_prop = props[0]  
        features['lesion_volume'] = lesion_prop.area
        features['surface_area'] = calculate_surface_area(binary_lesion)
        features['bounding_box_volume'] = np.prod(np.array(lesion_prop.bbox[3:]) - np.array(lesion_prop.bbox[:3]))
        features['solidity'] = lesion_prop.solidity
        features['convexity'] = calculate_convexity(binary_lesion)
        features['compactness'] = calculate_compactness(binary_lesion)

        features['major_axis_length'] = lesion_prop.major_axis_length
        features['minor_axis_length'] = lesion_prop.minor_axis_length
        features['least_axis_length'] = lesion_prop.minor_axis_length  

        features['max_3D_diameter'] = calculate_max_3D_diameter(binary_lesion)
        features['surface_volume_ratio'] = features['surface_area'] / features['lesion_volume'] if features['lesion_volume'] > 0 else 0
        features['sphericity'] = calculate_sphericity(binary_lesion)
        features['concave_points_count'] = detect_concave_points(binary_lesion)
        features['fractal_dimension'] = calculate_fractal_dimension(binary_lesion)
    else:
        # Safe assignments to avoid NaN values
        features = {key: 0 for key in [
            'lesion_volume', 'surface_area', 'solidity', 'convexity', 'compactness',
            'major_axis_length', 'minor_axis_length', 'least_axis_length',
            'max_3D_diameter', 'surface_volume_ratio', 'sphericity', 'concave_points_count',
            'fractal_dimension'
        ]}

    return features

def calculate_fractal_dimension(binary_lesion):
    sizes = np.arange(1, min(binary_lesion.shape) // 2, 2)  # Box sizes
    counts = []
    
    for size in sizes:
        count = np.sum(binary_lesion[::size, ::size, ::size])  # Count filled boxes
        counts.append(count)

    counts = np.array(counts)
    
    # Filter out sizes that resulted in zero counts
    valid_sizes = sizes[counts > 0]
    valid_counts = counts[counts > 0]

    # Avoid computing if there are no valid counts
    if len(valid_counts) < 2:  # Need at least two points to fit
        return np.nan  # or some other fallback value like 0 or -1

    log_sizes = np.log(valid_sizes)
    log_counts = np.log(valid_counts)

    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return -coeffs[0]

def extract_orientation_features(lesion):
    features = {}
    com = center_of_mass(lesion)
    com_index = np.unravel_index(np.argmax(lesion), lesion.shape)

    features['COM_x'] = com[0]
    features['COM_y'] = com[1]
    features['COM_z'] = com[2]
    
    features['COM_index_x'] = com_index[0]
    features['COM_index_y'] = com_index[1]
    features['COM_index_z'] = com_index[2]
    
    return features

def extract_intensity_features(lesion):
    features = {}
    intensity_values = lesion.flatten()
    intensity_values = intensity_values[intensity_values > 0]
    
    hist, bin_edges = np.histogram(intensity_values, bins=256, range=(0, 255))
    normalized_hist = hist / np.sum(hist)  
    
    features['hist_mean'] = np.mean(intensity_values)
    features['hist_std'] = np.std(intensity_values)
    features['hist_peak_intensity'] = np.argmax(hist)
    features['num_nonzero_bins'] = np.count_nonzero(hist)

    features['percentile_10'] = np.percentile(intensity_values, 10)
    features['percentile_25'] = np.percentile(intensity_values, 25)
    features['percentile_50'] = np.percentile(intensity_values, 50)  
    features['percentile_75'] = np.percentile(intensity_values, 75)
    features['percentile_90'] = np.percentile(intensity_values, 90)

    features['intensity_range'] = np.max(intensity_values) - np.min(intensity_values)
    features['intensity_variance'] = np.var(intensity_values)
    features['coefficient_of_variation'] = features['hist_std'] / features['hist_mean'] if features['hist_mean'] != 0 else 0
    features['IQR'] = features['percentile_75'] - features['percentile_25']
    features['intensity_hotspots_count'], features['intensity_hotspots'] = find_intensity_hotspots(lesion)

    weighted_com = weighted_center_of_mass(lesion)
    features['weighted_COM_x'] = weighted_com[0]
    features['weighted_COM_y'] = weighted_com[1]
    features['weighted_COM_z'] = weighted_com[2]

    return features

def calculate_surface_area(binary_lesion):
    verts, faces, _, _ = marching_cubes(binary_lesion)
    surface_area = len(faces)  # Each face contributes to surface area
    return surface_area

def calculate_convexity(binary_lesion):
    convex_hull = ConvexHull(np.argwhere(binary_lesion))
    return convex_hull.volume / np.sum(binary_lesion) if np.sum(binary_lesion) > 0 else 0

def calculate_compactness(binary_lesion):
    volume = np.sum(binary_lesion)
    surface_area = calculate_surface_area(binary_lesion)
    return (surface_area ** 2) / volume if volume > 0 else 0

def calculate_max_3D_diameter(binary_lesion):
    coords = np.argwhere(binary_lesion)
    max_distance = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = distance.euclidean(coords[i], coords[j])
            if dist > max_distance:
                max_distance = dist
    return max_distance

def calculate_sphericity(binary_lesion):
    volume = np.sum(binary_lesion)
    surface_area = calculate_surface_area(binary_lesion)
    return (4 * np.pi * volume) / (surface_area ** 2) if surface_area > 0 else 0

def detect_concave_points(binary_lesion):
    # Calculate concavity in 3D; use the local surface normals or curvature.
    # For simplicity, we will estimate the count by the number of contours found in each slice.
    concave_points_count = 0
    for i in range(binary_lesion.shape[0]):  # Iterate through each slice
        contours = find_contours(binary_lesion[i], 0.5)
        if contours:  # Check if any contours were found
            for contour in contours:
                concave_points_count += len(contour)  # Count the number of contour points in each slice
    return concave_points_count if concave_points_count > 0 else 0  # Return 0 if no concave points found

def find_intensity_hotspots(lesion, threshold=200):
    hotspots = np.where(lesion >= threshold)
    return len(hotspots[0]), hotspots

def weighted_center_of_mass(lesion):
    total_intensity = np.sum(lesion)
    if total_intensity == 0:
        return (0, 0, 0)
    
    weighted_sum_x = np.sum(np.arange(lesion.shape[0])[:, None, None] * lesion) 
    weighted_sum_y = np.sum(np.arange(lesion.shape[1])[None, :, None] * lesion) 
    weighted_sum_z = np.sum(np.arange(lesion.shape[2])[None, None, :] * lesion) 
    
    return (weighted_sum_x / total_intensity, weighted_sum_y / total_intensity, weighted_sum_z / total_intensity)

def create_mesh(binary_lesion):
    verts, faces, _, _ = marching_cubes(binary_lesion, level=0.5)
    return verts, faces

def compute_principal_curvature(lesion):
    # Smooth the lesion to reduce noise
    smooth_lesion = gaussian_filter(lesion, sigma=1)

    # Compute the gradients
    dz, dy, dx = np.gradient(smooth_lesion)

    # Compute the second derivatives (Hessian)
    dxx = np.gradient(dx)[0]
    dyy = np.gradient(dy)[1]
    dzz = np.gradient(dz)[2]
    dxy = np.gradient(dx)[1]
    dxz = np.gradient(dx)[2]
    dyz = np.gradient(dy)[2]

    # Construct Hessian matrix
    H = np.array([[dxx, dxy, dxz],
                  [dxy, dyy, dyz],
                  [dxz, dyz, dzz]])

    # Compute eigenvalues of Hessian
    curvature_values = np.linalg.eigvalsh(H)
    
    # The principal curvatures are the eigenvalues
    mean_curvature = np.mean(curvature_values)
    std_curvature = np.std(curvature_values)
    return mean_curvature, std_curvature

def calculate_spatial_correlation(lesion):
    flattened = lesion.flatten()
    return np.corrcoef(flattened)

def calculate_neighborhood_intensity_relations(lesion):
    gradients = gaussian_gradient_magnitude(lesion, sigma=1)
    return np.mean(gradients), np.std(gradients)

def extract_glcm_features(lesion):
    contrast_list = []
    dissimilarity_list = []
    
    for i in range(lesion.shape[0]):  # Iterate through each slice in the 3D lesion
        slice_2d = lesion[i].astype(np.uint8)
        glcm = graycomatrix(slice_2d, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast_list.append(graycoprops(glcm, 'contrast')[0, 0])
        dissimilarity_list.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    
    # Average the features across all slices
    return np.mean(contrast_list), np.mean(dissimilarity_list)

def extract_lbp_features(lesion):
    lbp_hist_list = []
    
    for i in range(lesion.shape[0]):  # Iterate through each slice in the 3D lesion
        slice_2d = lesion[i].astype(np.uint8)
        lbp = local_binary_pattern(slice_2d, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
        lbp_hist_list.append(hist)
    
    # Average the LBP histograms across all slices
    avg_lbp_hist = np.mean(lbp_hist_list, axis=0)
    return avg_lbp_hist

def calculate_aspect_ratio(shape_features):
    return shape_features['major_axis_length'] / shape_features['minor_axis_length'] if shape_features['minor_axis_length'] > 0 else 0

def calculate_local_curvature(binary_lesion):
    # Placeholder function; the implementation would depend on the specific algorithm chosen
    local_curvature = np.random.rand(np.sum(binary_lesion))  # Replace with actual curvature calculations
    return np.mean(local_curvature), np.std(local_curvature)

# Example usage
lesion = create_spherical_lesion_with_intensity_fluctuations()
features = extract_features_from_lesion(lesion)
print(features)