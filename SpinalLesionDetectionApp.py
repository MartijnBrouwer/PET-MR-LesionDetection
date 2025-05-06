import sys
import os
import SimpleITK as sitk
import numpy as np
import subprocess
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit)
from PyQt6.QtCore import Qt

class SpinalLesionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.mr_label = QLabel("MR Image: Not Selected")
        self.pet_label = QLabel("PET Image: Not Selected")
        
        self.mr_button = QPushButton("Select MR Image (NiFTI)")
        self.pet_button = QPushButton("Select PET Image (NiFTI)")
        self.run_button = QPushButton("Run Processing and Save")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        
        self.mr_button.clicked.connect(self.select_mr_image)
        self.pet_button.clicked.connect(self.select_pet_image)
        self.run_button.clicked.connect(self.run_processing)
        
        layout.addWidget(self.mr_label)
        layout.addWidget(self.mr_button)
        layout.addWidget(self.pet_label)
        layout.addWidget(self.pet_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_text)

        self.setLayout(layout)
        self.setWindowTitle("Spinal Lesion Detection Tool")
        self.setGeometry(100, 100, 600, 400)
        
        self.mr_path = ""
        self.pet_path = ""

    def select_mr_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select MR Image", "", "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.mr_path = file_path
            self.mr_label.setText(f"MR Image: {file_path}")

    def select_pet_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PET Image", "", "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.pet_path = file_path
            self.pet_label.setText(f"PET Image: {file_path}")
    
    def load_image(self, image_path):
        if os.path.isdir(image_path):  # DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_series = reader.GetGDCMSeriesFileNames(image_path)
            reader.SetFileNames(dicom_series)
            image = reader.Execute()
        else:  # NIfTI file
            image = sitk.ReadImage(image_path)
        return image
    
    def align_images(self, fixed_image_path, moving_image_path, output_image_path):
        fixed_image = self.load_image(fixed_image_path)
        moving_image = self.load_image(moving_image_path)
        
        moving_image = sitk.Cast(moving_image, fixed_image.GetPixelID())
        
        # Initial alignment using a rigid transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(initial_transform)
        initial_resampled = resampler.Execute(moving_image)
        
        # Refine alignment using Demons registration
        demons_filter = sitk.DemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(10)
        demons_filter.SetStandardDeviations(2.0)
        displacement_field = demons_filter.Execute(fixed_image, initial_resampled)
        refined_transform = sitk.DisplacementFieldTransform(displacement_field)
        
        # Apply refined transform
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            refined_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID()
        )

        # Save and return output
        sitk.WriteImage(moving_resampled, output_image_path)
        return output_image_path

    def crop_image(self, image_path):
        image = self.load_image(image_path)
        array = sitk.GetArrayFromImage(image)
        
        z_size, y_size, x_size = array.shape
        x_third = x_size // 3
        x_start = x_third
        x_end = 2 * x_third

        if 'r' in image_path[-18:]: # Check if it is a reference patient (those have an "r" in its name and are full body)
            z_mid = z_size // 2
            processed_array = array[:z_mid, :, x_start:x_end]
            new_origin = list(image.GetOrigin())
            new_origin[0] += x_start * image.GetSpacing()[0]
            new_origin[2] += z_mid * image.GetSpacing()[2]
        else:
            z_third = z_size // 3
            processed_array = array[z_third:2*z_third, :, x_start:x_end]
            new_origin = list(image.GetOrigin())
            new_origin[0] += x_start * image.GetSpacing()[0]
            new_origin[2] += z_third * image.GetSpacing()[2]
        
        processed_image = sitk.GetImageFromArray(processed_array)
        processed_image.SetSpacing(image.GetSpacing())
        processed_image.SetDirection(image.GetDirection())
        processed_image.SetOrigin(new_origin)
        output_path = image_path.replace(".nii", "_cropped.nii")
        sitk.WriteImage(processed_image, output_path)
        return output_path

    def segment_spinal_cord(self, mr_path, output_dir):
        mask_folder = os.path.join(output_dir, "segmentation_masks")
        os.makedirs(mask_folder, exist_ok=True)

        command = ['TotalSegmentator', '-i', mr_path, '-o', mask_folder, '--task', 'total_mr']
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            self.output_text.append(f"Error in spinal cord segmentation: {result.stderr}")
            return None

        spinal_cord_path = os.path.join(mask_folder, 'spinal_cord.nii.gz')
        if os.path.exists(spinal_cord_path):
            new_path = os.path.join(output_dir, os.path.basename(mr_path).replace(".nii", "_spinal_cord.nii"))
            os.rename(spinal_cord_path, new_path)
            return new_path
        return None
    
    def extract_pet_spinal_canal(self, pet_path, spinal_canal_path, output_path):
        pet_image = sitk.ReadImage(pet_path)
        spinal_canal_image = sitk.ReadImage(spinal_canal_path)

        # Convert to NumPy arrays
        pet_array = sitk.GetArrayFromImage(pet_image)
        spine_mask_array = sitk.GetArrayFromImage(spinal_canal_image)

        # Apply mask
        pet_array *= spine_mask_array

        # Skip if there is no nonzero intensity left
        if np.count_nonzero(pet_array) == 0:
            self.output_text.append(f"Skipping PET extraction - No overlap with spinal canal mask.")
            return
        
        # Find bounding box
        nonzero_coords = np.argwhere(pet_array > 0)
        min_z, min_y, min_x = nonzero_coords.min(axis=0)
        max_z, max_y, max_x = nonzero_coords.max(axis=0)

        # Crop the PET image
        pet_cropped = pet_array[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

        # Convert back to SimpleITK image
        pet_cropped_image = sitk.GetImageFromArray(pet_cropped)
        
        # Preserve metadata
        pet_cropped_image.SetSpacing(pet_image.GetSpacing())
        pet_cropped_image.SetDirection(pet_image.GetDirection())

        # Compute new origin
        original_origin = np.array(pet_image.GetOrigin())
        spacing = np.array(pet_image.GetSpacing())
        new_origin = original_origin + spacing * np.array([min_x, min_y, min_z])
        pet_cropped_image.SetOrigin(tuple(new_origin))

        # Save output
        sitk.WriteImage(pet_cropped_image, output_path)
        self.output_text.append(f"Processed and saved: {output_path}")

    def run_nnUnet_inference(self, input_image, output_folder):
        dataset_id = "006"
        model_type = "3d_fullres"
        trainer = "nnUNetTrainer"
        fold = "0"
        checkpoint_file = "checkpoint_best.pth"

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        command = [
            "nnUNetv2_predict",
            "-i", input_image,
            "-o", output_folder,
            "-d", dataset_id,
            "-c", model_type,
            "-tr", trainer,
            "-f", fold,
            "-chk", checkpoint_file,
            "-device", "cpu",
            "--save_probabilities"
        ]

        print(f"Running nnUNet with command: {' '.join(command)}")

        result = subprocess.run(command, capture_output=True, text=True)

    def run_processing(self):
        if not self.mr_path or not self.pet_path:
            self.output_text.append("\nPlease select both MR and PET images.")
            return
        
        self.output_text.append("\nAligning MR image to PET...")
        aligned_mr_path = self.mr_path.replace(".nii", "_aligned.nii")
        aligned_mr_path = self.align_images(self.pet_path, self.mr_path, aligned_mr_path)
        
        self.output_text.append("\nCropping PET and MR images...")
        cropped_pet_path = self.crop_image(self.pet_path)
        cropped_mr_path = self.crop_image(aligned_mr_path)
        
        self.output_text.append("\nExtracting spinal cord from MR image...")
        spinal_cord_path = self.segment_spinal_cord(cropped_mr_path, os.path.dirname(cropped_mr_path))
        
        self.output_text.append(f"\nSpinal canal extracted. Files saved:\nAligned MR: {aligned_mr_path}\nCropped PET: {cropped_pet_path}\nCropped MR: {cropped_mr_path}\nSpinal Cord: {spinal_cord_path}")

        self.output_text.append("\nExtracting spinal canal from PET image...")
        
        # Create a dedicated folder for the PET spinal canal output & save the extracted PET spinal canal image inside the new folder
        pet_spinalcanal_folder = os.path.join(os.path.dirname(self.pet_path), "PET_SpinalCanal_Output")
        os.makedirs(pet_spinalcanal_folder, exist_ok=True)
        output_pet_spinal_canal_path = os.path.join(pet_spinalcanal_folder, f"PET_spinalcanal_{os.path.basename(self.pet_path)[:-4]}_0000.nii.gz")

        if spinal_cord_path:
            self.extract_pet_spinal_canal(cropped_pet_path, spinal_cord_path, output_pet_spinal_canal_path)
            self.output_text.append(f"\nExtracted PET spinal canal saved at: {output_pet_spinal_canal_path}")
        else:
            self.output_text.append("\nSpinal canal segmentation failed, skipping PET extraction.")
                
        os.environ["nnUNet_raw"] = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\nnUnet\nnUNet_raw"
        os.environ["nnUNet_preprocessed"] = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\nnUnet\nnUNet_preprocessed"
        os.environ["nnUNet_results"] = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\nnUnet\nnUNet_results"

        self.output_text.append("\nRunning nnUNet inference on extracted PET spinal canal...")
        nnunet_output_folder = os.path.join(os.path.dirname(self.pet_path), "nnUNet_output")
        os.makedirs(nnunet_output_folder, exist_ok=True)
        self.run_nnUnet_inference(pet_spinalcanal_folder, nnunet_output_folder)
        self.output_text.append(f"\nLesion prediction saved in: {nnunet_output_folder}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpinalLesionApp()
    window.show()
    sys.exit(app.exec())