#!/usr/bin/python3


import os
import subprocess

def run_command(command):
    print(f"Running command: {command}")
    proc = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    proc.communicate() 


def convert_dicom(subject, input_directory, output_directory, skip=False):

    dcm2niix = os.path.join(os.getcwd(), "share/dcm2niix/build/bin/dcm2niix")
    nifti_filename = f"{subject}_T1w"
    output_nifti = os.path.join(output_directory, f"{nifti_filename}.nii.gz")

    if not (os.path.exists(output_nifti)):

        dcm_cmd = f"{dcm2niix} -o {output_directory} -f {nifti_filename} {input_directory}"

        if skip == False: 
            run_command(dcm_cmd)
    
    return output_nifti