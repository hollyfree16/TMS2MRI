#!/usr/bin/python3


import numpy as np
from nilearn import datasets
import nibabel as nib
import os

def atlas_location (mni_voxel_coordinates):
    custom_data_dir = mni_template = os.getcwd()

    cortical_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', data_dir=custom_data_dir)

    cortical_atlas_img = cortical_atlas['maps']
    cortical_labels = cortical_atlas['labels']

    voxel_coord_mni = mni_voxel_coordinates

    print(f'MNI coordinates: {voxel_coord_mni}')

    def within_bounds(coord, shape):
        return all(0 <= c < s for c, s in zip(coord, shape))

    if not within_bounds(voxel_coord_mni, cortical_atlas_img.shape[:3]):
        print(f'MNI coordinates {voxel_coord_mni} are out of bounds for atlas shape {cortical_atlas_img.shape[:3]}.')
        exit()

    region_name = 'Unknown'
    label_index = cortical_atlas_img.get_fdata()[tuple(voxel_coord_mni)]
    if label_index > 0:
        region_name = cortical_labels[int(label_index)]

    print(f'Region: {region_name}')

    return region_name
