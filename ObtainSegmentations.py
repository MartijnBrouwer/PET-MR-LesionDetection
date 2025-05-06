# TotalSegmentator for MR images
# This codes segments all available body parts in the tasks 'total_mr' and 'vertebrae_mr' and saves a combined output mask.
# It loops over all NiFTI-images available in the input directory and saves the saves all body parts segmented separately,
# combined and also only the relevant spinal parts combined. 
# Author: @MartijnBrouwer
# ===========================================================

import subprocess
import os
import nibabel as nib
import numpy as np
import gc
import time

# Define input and output directories
input_dir = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\INPHASE"
output_base_dir = r"\\cifs.research.erasmusmc.nl\nuge0001\MartijnB\NIFTI-images\SegmentedMRs"

# Segmentation tasks
tasks = ['total_mr', 'vertebrae_mr']

# Define body part mappings
body_part_mapping = {
    'adrenal_gland_left': 1, 'adrenal_gland_right': 2, 'aorta': 3, 'autochthon_left': 4,
    'autochthon_right': 5, 'brain': 6, 'clavicula_left': 7, 'clavicula_right': 8, 'colon': 9,
    'duodenum': 10, 'esophagus': 11, 'femur_left': 12, 'femur_right': 13, 'gallbladder': 14,
    'gluteus_maximus_left': 15, 'gluteus_maximus_right': 16, 'gluteus_medius_left': 17,
    'gluteus_medius_right': 18, 'gluteus_minimus_left': 19, 'gluteus_minimus_right': 20,
    'heart': 21, 'hip_left': 22, 'hip_right': 23, 'humerus_left': 24, 'humerus_right': 25,
    'iliac_artery_left': 26, 'iliac_artery_right': 27, 'iliac_vena_left': 28, 'iliac_vena_right': 29,
    'iliopsoas_left': 30, 'iliopsoas_right': 31, 'inferior_vena_cava': 32, 'intervertebral_discs': 33,
    'kidney_left': 34, 'kidney_right': 35, 'liver': 36, 'lung_left': 37, 'lung_right': 38,
    'pancreas': 39, 'portal_vein_and_splenic_vein': 40, 'prostate': 41, 'sacrum': 42,
    'scapula_left': 43, 'scapula_right': 44, 'small_bowel': 45, 'spinal_cord': 46, 'spleen': 47,
    'stomach': 48, 'urinary_bladder': 49, 'vertebrae_C1': 50, 'vertebrae_C2': 51, 'vertebrae_C3': 52,
    'vertebrae_C4': 53, 'vertebrae_C5': 54, 'vertebrae_C6': 55, 'vertebrae_C7': 56,
    'vertebrae_L1': 57, 'vertebrae_L2': 58, 'vertebrae_L3': 59, 'vertebrae_L4': 60,
    'vertebrae_L5': 61, 'vertebrae_T1': 62, 'vertebrae_T10': 63, 'vertebrae_T11': 64,
    'vertebrae_T12': 65, 'vertebrae_T2': 66, 'vertebrae_T3': 67, 'vertebrae_T4': 68,
    'vertebrae_T5': 69, 'vertebrae_T6': 70,
}

# Define spine-specific structures
spine_labels = {   
    'intervertebral_discs': 33, 'sacrum': 42, 'spinal_cord': 46, 'vertebrae_C1': 50,
    'vertebrae_C2': 51, 'vertebrae_C3': 52, 'vertebrae_C4': 53, 'vertebrae_C5': 54,
    'vertebrae_C6': 55, 'vertebrae_C7': 56, 'vertebrae_L1': 57, 'vertebrae_L2': 58,
    'vertebrae_L3': 59, 'vertebrae_L4': 60, 'vertebrae_L5': 61, 'vertebrae_T1': 62,
    'vertebrae_T10': 63, 'vertebrae_T11': 64, 'vertebrae_T12': 65, 'vertebrae_T2': 66,
    'vertebrae_T3': 67, 'vertebrae_T4': 68, 'vertebrae_T5': 69, 'vertebrae_T6': 70,
}

# Define timeout limit (in seconds)
TIMEOUT_LIMIT = 3600*0.75  # 1 hour per task

# Process all NIfTI images
for file_name in os.listdir(input_dir):
    if file_name.endswith(".nii.gz"):
        file_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(os.path.splitext(file_name)[0])[0]  # Remove .nii.gz extension
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Run segmentation tasks with timeout
        for task in tasks:
            command = ['TotalSegmentator', '-i', file_path, '-o', output_dir, '--task', task]
            print(f"Running: {' '.join(command)}")

            try:
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                        errors="replace", timeout=TIMEOUT_LIMIT)
                if result.returncode != 0:
                    print(f"Task {task} failed for {file_name}. Skipping...")
                    continue
            except subprocess.TimeoutExpired:
                print(f"Task {task} took too long for {file_name}. Skipping...")
                continue
            except Exception as e:
                print(f"Error running TotalSegmentator for {file_name}: {e}")
                continue

        # Combine segmentations
        segmented_masks = []
        try:
            for mask_file in os.listdir(output_dir):
                if mask_file.endswith('.nii.gz') and mask_file != "vertebrae.nii.gz":
                    mask_path = os.path.join(output_dir, mask_file)
                    mask_img = nib.load(mask_path)
                    mask_data = mask_img.get_fdata(dtype=np.float32)
                    segmented_masks.append((mask_data, mask_file))
        except Exception as e:
            print(f"Error loading segmentation masks for {file_name}: {e}")
            continue

        # Create combined masks
        if segmented_masks:
            combined_mask = np.zeros_like(segmented_masks[0][0], dtype=np.uint16)
            spine_mask = np.zeros_like(segmented_masks[0][0], dtype=np.uint16)

            for mask_data, mask_name in segmented_masks:
                for part_name, label in body_part_mapping.items():
                    if part_name.lower() in mask_name.lower():
                        combined_mask[mask_data != 0] = label
                        if part_name in spine_labels:
                            spine_mask[mask_data != 0] = label
                        break

            # Save combined masks
            try:
                combined_nifti = nib.Nifti1Image(combined_mask, affine=mask_img.affine, header=mask_img.header)
                combined_mask_path = os.path.join(output_dir, "combined_segmentation.nii.gz")
                nib.save(combined_nifti, combined_mask_path)
                print(f"Saved combined mask: {combined_mask_path}")

                spine_nifti = nib.Nifti1Image(spine_mask, affine=mask_img.affine, header=mask_img.header)
                spine_mask_path = os.path.join(output_dir, "spine_segmentation.nii.gz")
                nib.save(spine_nifti, spine_mask_path)
                print(f"Saved spine mask: {spine_mask_path}")

            except Exception as e:
                print(f"Error saving masks for {file_name}: {e}")

        # Clear memory after processing each patient
        del segmented_masks, combined_mask, spine_mask
        gc.collect()
