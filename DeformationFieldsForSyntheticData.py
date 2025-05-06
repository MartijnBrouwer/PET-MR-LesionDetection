import itk
import os
import numpy as np

def register_masks_and_apply_to_mri(fixed_mask_path, moving_mask_path, fixed_mri_path, moving_mri_path, output_directory):
    # Load masks
    fixed_mask = itk.imread(fixed_mask_path, itk.UC)
    moving_mask = itk.imread(moving_mask_path, itk.UC)
    
    # Convert masks to numpy arrays
    fixed_array = itk.GetArrayFromImage(fixed_mask)
    moving_array = itk.GetArrayFromImage(moving_mask)
    
    # Remove unwanted artifacts (labels > 80)
    fixed_array = np.where(fixed_array > 80, 0, fixed_array)
    moving_array = np.where(moving_array > 80, 0, moving_array)
    
    # Find common labels
    fixed_labels = set(np.unique(fixed_array))
    moving_labels = set(np.unique(moving_array))
    common_labels = fixed_labels.intersection(moving_labels)
    
    # Create new masks with only common labels
    fixed_array_filtered = np.where(np.isin(fixed_array, list(common_labels)), fixed_array, 0)
    moving_array_filtered = np.where(np.isin(moving_array, list(common_labels)), moving_array, 0)
    
    # Convert back to ITK images
    fixed_mask_filtered = itk.GetImageFromArray(fixed_array_filtered)
    moving_mask_filtered = itk.GetImageFromArray(moving_array_filtered)
    fixed_mask_filtered.CopyInformation(fixed_mask)
    moving_mask_filtered.CopyInformation(moving_mask)
    
    # Resample moving mask to match the fixed mask
    moving_mask_filtered = resample_image(moving_mask_filtered, fixed_mask_filtered)
    
    # Setup Elastix for mask registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_mask_filtered, moving_mask_filtered)
    
    # Create parameter object with rigid, affine, and b-spline transformations
    parameter_object = itk.ParameterObject.New()
    
    # Add parameters for rigid, affine, and bspline (with higher iterations and multi-resolution)
    rigid_map = parameter_object.GetDefaultParameterMap("rigid")
    affine_map = parameter_object.GetDefaultParameterMap("affine")
    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    
    # Increase iterations for each transformation
    rigid_map["MaximumNumberOfIterations"] = ["5000"]
    affine_map["MaximumNumberOfIterations"] = ["5000"]
    bspline_map["MaximumNumberOfIterations"] = ["5000"]
    
    # Multi-resolution strategy
    rigid_map["ResampleInterval"] = ["4"]
    affine_map["ResampleInterval"] = ["4"]
    bspline_map["ResampleInterval"] = ["2"]
    
    # B-spline: use more conservative parameters (lower regularization to reduce over-deformation)
    bspline_map["MaximumStepLength"] = ["0.01"]  # Reduce the deformation strength
    bspline_map["FinalBSplineInterpolationOrder"] = ["0"]
    
    # Add the maps to the parameter object
    parameter_object.AddParameterMap(rigid_map)
    parameter_object.AddParameterMap(affine_map)
    parameter_object.AddParameterMap(bspline_map)
    
    # Refine rigid registration settings
    parameter_object.SetParameter("DefaultPixelValue", "0")
    
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetOutputDirectory(output_directory)
    elastix_object.SetLogToConsole(True)
    elastix_object.SetLogToFile(True)
    
    # Run rigid registration and remove labels above 80
    elastix_object.Update()
    fixed_mask_filtered = remove_labels_above_80(elastix_object.GetOutput())
    
    # Get the transformation parameters after rigid registration
    transform_parameter_map = elastix_object.GetTransformParameterObject()
    
    # Apply affine registration and remove labels above 80
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.Update()
    fixed_mask_filtered = remove_labels_above_80(elastix_object.GetOutput())
    
    # Apply B-spline registration and remove labels above 80
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.Update()
    fixed_mask_filtered = remove_labels_above_80(elastix_object.GetOutput())
    
    # Save registered mask after all steps
    itk.imwrite(fixed_mask_filtered, os.path.join(output_directory, "registered_mask.nii.gz"))
    
    # Apply transformation to the moving MRI
    moving_mri = itk.imread(moving_mri_path, itk.F)
    
    transformix_object = itk.TransformixFilter.New(moving_mri)
    transformix_object.SetTransformParameterObject(transform_parameter_map)
    transformix_object.SetOutputDirectory(output_directory)
    transformix_object.Update()
    
    # Save transformed MRI
    transformed_mri = transformix_object.GetOutput()
    itk.imwrite(transformed_mri, os.path.join(output_directory, "registered_mri.nii.gz"))
    
    print("Registration complete. Transformed PET saved.")

    moving_mri = itk.imread(r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\PETSUV\EMC-0001r.nii", itk.F)
    
    transformix_object = itk.TransformixFilter.New(moving_mri)
    transformix_object.SetTransformParameterObject(transform_parameter_map)
    transformix_object.SetOutputDirectory(output_directory)
    transformix_object.Update()
    
    # Save transformed MRI
    transformed_mri = transformix_object.GetOutput()
    itk.imwrite(transformed_mri, os.path.join(output_directory, "registered_pet.nii.gz"))


def resample_image(moving_image, reference_image):
    resample = itk.ResampleImageFilter.New(Input=moving_image, ReferenceImage=reference_image)
    resample.SetUseReferenceImage(True)
    resample.SetInterpolator(itk.NearestNeighborInterpolateImageFunction.New(moving_image))
    resample.Update()
    return resample.GetOutput()

def remove_labels_above_80(mask):
    """Remove labels greater than 80 from the mask image"""
    mask_array = itk.GetArrayFromImage(mask)
    mask_array = np.where(mask_array > 80, 0, mask_array)  # Remove labels above 80
    mask = itk.GetImageFromArray(mask_array)
    mask.CopyInformation(mask)  # Preserve metadata
    return mask

# Paths
output_directory = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\Other\SyntheticPatients"
moving_mri_path = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\AlignedMRs\EMC-0001r.nii.gz"
fixed_mri_path = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\AlignedMRs\EMC-0003r.nii.gz"
moving_mask_path = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\SegmentedMRs\EMC-0001r\combined_segmentation.nii.gz"
fixed_mask_path = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\SegmentedMRs\EMC-0003r\combined_segmentation.nii.gz"

os.makedirs(output_directory, exist_ok=True)

# Perform registration
register_masks_and_apply_to_mri(fixed_mask_path, moving_mask_path, fixed_mri_path, moving_mri_path, output_directory)