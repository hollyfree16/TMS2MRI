#!/usr/bin/python3

import os
import pydicom
import shutil

def is_dicom_file(file_path):
    try:
        dicom_file = pydicom.dcmread(file_path)
        # If reading succeeds without error, it's a valid DICOM file
        return True
    except Exception as e:
        # If reading fails, it's not a DICOM file
        return False

def is_nifti_file(file_path):
    # Assuming NIfTI files end with .nii or .nii.gz
    return file_path.lower().endswith('.nii') or file_path.lower().endswith('.nii.gz')

def check_directory_for_dicom(directory):
    all_dicom = False
    nifti_file_path = None
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            if is_dicom_file(file_path):
                all_dicom = True
            elif is_nifti_file(file_path):
                print(f"Found NIfTI file: '{filename}'")
                nifti_file_path = file_path

    if all_dicom:
        print("All files in the directory are DICOM files.")
    elif nifti_file_path:
        print("Found at least one NIfTI file in the directory.")
    else:
        print("No DICOM or NIfTI files found in the directory.")
        return False, None
    
    return True, nifti_file_path

