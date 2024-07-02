TMS2MRI Data Processing Pipeline

This repository contains a Python script designed to process TMS (Transcranial Magnetic Stimulation) stimulation target coordinates to MRI (Magnetic Resonance Imaging) subject space. It handles DICOM to NIfTI conversion, extracts coordinates, and queries MNI atlas locations for given brain regions. The results are saved in a CSV file.


Features

•	Process subject data from a CSV file.

•	Convert DICOM images to NIfTI format.

•	Extract and convert coordinates.

•	Query brain atlas regions based on coordinates.

•	Generate output in a structured CSV format.



Requirements

•	Python 3.x

•	nibabel


Installation
1.	Clone the repository:
git clone https://github.com/hollyfree16/TMS2MRI.git

3.	Install the required Python packages:
pip install nibabel


Usage

To run the script, use the following command:

python3 TMS2MRI.py --csv <path_to_csv_file> --output_directory <output_directory> [--subject <subject_id>]


Arguments
•	--csv: Path to the input CSV file containing subject data.
•	--output_directory: Directory where the output data will be stored.
•	--subject: (Optional) Process only the specified subject ID from the CSV file.


CSV File Format
The input CSV file should have the following columns:

1.	Subject ID
2.	X (nexstim)
3.	Y (nexstim)
4.	Z (nexstim)
5.	Label
6.	Source Data


Output

The script generates an output CSV file named with the current date and optional subject ID. The output file contains the following columns:
•	Subject ID
•	Stimulation Target
•	X (nexstim)
•	Y (nexstim)
•	Z (nexstim)
•	X
•	Y
•	Z
•	Inverted X
•	Atlas Region
•	MNI X
•	MNI Y
•	MNI Z
•	MNI Inverted X
•	Inverted Atlas Region


Please note, inverted X coordinates are provided in case there is L/R swapping present in the data. You will need to examine the results to verify which orientation best matches your data.


Contact

For any questions or suggestions, feel free to contact hjfreeman@mgh.harvard.edu

