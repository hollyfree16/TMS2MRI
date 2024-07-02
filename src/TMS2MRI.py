#!/usr/bin/python3


import os
import shutil
import argparse
from datetime import datetime
import csv
import nibabel as nib
from read_csv import process_csv
from dicom_check import check_directory_for_dicom
from convert_dicom import convert_dicom
from convert_coordinates import staging
from query_atlas import atlas_location


def make_directories(directory, name=None):
    directory_path = os.path.join(directory, name) if name else directory
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def process_subject(subject_id, x, y, z, label, sourcedata, output_directory):
    print("*" * 60)
    print(f"Processing:\n   Subject ID: {subject_id} \n   Coordinates: ({x}, {y}, {z}) \n   Label: {label}\n   Source Data: {sourcedata}\n")

    subject_data_source = sourcedata
    is_valid_data, nifti_file_path = check_directory_for_dicom(subject_data_source)

    if not is_valid_data:
        print(f"Skipping subject {subject_id} as no valid DICOM or NIfTI files found.")
        return

    subject_output_directory = make_directories(output_directory, name=subject_id)
    exists = subject_id in data_dict

    if not exists:
        if nifti_file_path:
            base_filename = os.path.basename(nifti_file_path)
            file_name, file_extension = os.path.splitext(base_filename)

            if file_extension == ".nii":
                nii_img = nib.load(nifti_file_path)
                new_nifti_path = os.path.join(subject_output_directory, f"{subject_id}_T1w.nii.gz")
                nib.save(nii_img, new_nifti_path)
            else:
                new_nifti_path = os.path.join(subject_output_directory, f"{subject_id}_T1w.nii.gz")
                shutil.copy2(nifti_file_path, new_nifti_path)

            print(f"Copied NIfTI file '{nifti_file_path}' to '{new_nifti_path}'")

        t1_data = convert_dicom(subject_id, subject_data_source, subject_output_directory, skip=False)
        data_dict[subject_id] = t1_data
    else:
        t1_data = convert_dicom(subject_id, subject_data_source, subject_output_directory, skip=True)

    staging_result = staging(t1_data, subject_output_directory, subject_id, x, y, z, label_name=label.replace(" ", "-"))

    if staging_result is None:
        print(f"Skipping subject {subject_id} due to out of bounds center of mass.")
        return

    mni_center_of_label, mni_inverted_center_of_label, native_center_of_label, native_inverted_center_of_label = staging_result

    atlas_region = atlas_location(mni_center_of_label)
    inverted_atlas_region = atlas_location(mni_inverted_center_of_label)

    print("Cleaning up intermediate files")
    for file_name in os.listdir(subject_output_directory):
        if "tmp" in file_name:
            file_path = os.path.join(subject_output_directory, file_name)
            try:
                os.remove(file_path)
                print(f"Removed intermediate file: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")


    return native_center_of_label, native_inverted_center_of_label, mni_center_of_label, mni_inverted_center_of_label, atlas_region, inverted_atlas_region


def main(csv_filename, output_directory, sid=None):
    output_directory = make_directories(output_directory, name="TMS2MRI_output")
    data = process_csv(csv_filename)

    global data_dict
    data_dict = {}

    current_date = datetime.now().strftime('%Y%m%d')

    if sid is not None: 
        output_csv = os.path.join(output_directory, f"{current_date}-{sid}.csv")
    else: 
        output_csv = os.path.join(output_directory, f"{current_date}.csv")

    try:
        with open(output_csv, 'a', newline='') as csvfile:
            fieldnames = ['Subject ID', 'Stimulation Target', 'X (nexstim)', 'Y (nexstim)', 'Z (nexstim)',
                          'X', 'Y', 'Z', 'Inverted X', 'Atlas Region', 'MNI X', 'MNI Y', 'MNI Z',
                          'MNI Inverted X', 'Inverted Atlas Region']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


            csvfile.seek(0, 2) 
            file_empty = csvfile.tell() == 0

            if file_empty:
                print("CSV file is empty. Writing header.")
                writer.writeheader()

  
            if not data:
                print("Data is empty. Nothing to write.")
            else:
                for subject_id, x, y, z, label, sourcedata in data:
                    if sid and subject_id != sid:
                        continue

                    process_result = process_subject(subject_id, x, y, z, label, sourcedata, output_directory)

                    if process_result is None: 
                        print(f"Processing result for subject {subject_id} is None. Skipping.")
                        continue

                    native_center_of_label, native_inverted_center_of_label, mni_center_of_label, mni_inverted_center_of_label, atlas_region, inverted_atlas_region =   process_result

                    row_data = {
                        'Subject ID': subject_id,
                        'Stimulation Target': label,
                        'X (nexstim)': x,
                        'Y (nexstim)': y,
                        'Z (nexstim)': z,
                        'X': native_center_of_label[0] if native_center_of_label.any() else '',
                        'Y': native_inverted_center_of_label[1] if native_inverted_center_of_label.any() else '',
                        'Z': native_center_of_label[2] if native_center_of_label.any() else '',
                        'Inverted X': native_inverted_center_of_label[0] if native_inverted_center_of_label.any() else '',
                        'Atlas Region': atlas_region,
                        'MNI X': mni_center_of_label[0] if mni_center_of_label.any() else '',
                        'MNI Y': mni_inverted_center_of_label[1] if mni_inverted_center_of_label.any() else '',
                        'MNI Z': mni_center_of_label[2] if mni_center_of_label.any() else '',
                        'MNI Inverted X': mni_inverted_center_of_label[0] if mni_inverted_center_of_label.any() else '',
                        'Inverted Atlas Region': inverted_atlas_region
                    }

                    print(f"Writing row for subject {subject_id}: {row_data}")
                    writer.writerow(row_data)

        print(f"Data appended to '{output_csv}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TMS2MRI data")
    parser.add_argument('--csv', type=str, required=True, help="Path to the CSV file")
    parser.add_argument('--output_directory', type=str, required=True, help="Directory to store the output data")
    parser.add_argument('--subject', type=str, help="Process only the specified subject-id from the CSV")

    args = parser.parse_args()

    main(args.csv, args.output_directory, args.subject)
