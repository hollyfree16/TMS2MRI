#!/usr/bin/python3


import ants
import shutil
import numpy as np
import nibabel as nib
import os
import sys

def create_sphere_mask(center, shape, radius):
    """
    Create a spherical mask with the given center, shape, and radius.
    """
    grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((grid[0] - center[0])**2 + (grid[1] - center[1])**2 + (grid[2] - center[2])**2)
    mask = distance <= radius
    return mask.astype(np.uint8)

def mni_operations(fixed_image_path, moving_image_path, label_image_path, output_directory, output_transform_image_name=None, output_transformed_label_name=None, output_sphere_image_name=None, sphere_radius=None):

    # Load the fixed and moving images
    fixed = ants.image_read(fixed_image_path)
    moving = ants.image_read(moving_image_path)

    # Perform registration
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')

    # Apply the transform to the moving image
    transformed_image = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'])

    output_transform_image = os.path.join(output_directory, output_transform_image_name)
    # Save the transformed image
    ants.image_write(transformed_image, output_transform_image)

    # Get the list of forward transforms
    fwd_transforms = reg['fwdtransforms']

    for i, transform_path in enumerate(fwd_transforms):
        print(f"Forward Transform {i+1}: {transform_path}")
        destination_path = os.path.join(output_directory, f"transform_{i+1}.mat")
        shutil.copy(transform_path, destination_path)
        print(f"Copied to: {destination_path}")

    # Load and transform the label volume
    label_volume = ants.image_read(label_image_path)
    output_transformed_label = os.path.join(output_directory, output_transformed_label_name)
    transformed_label = ants.apply_transforms(fixed=fixed, moving=label_volume, transformlist=reg['fwdtransforms'])
    ants.image_write(transformed_label, output_transformed_label)

    # Convert ANTs image to numpy array for easier manipulation
    transformed_label_np = transformed_label.numpy()

    # Find non-zero voxel coordinates
    nonzero_indices = np.transpose(np.nonzero(transformed_label_np))

    # Compute center of mass of non-zero voxels
    center_of_mass = np.mean(nonzero_indices, axis=0)
    print(f"Center of mass: {center_of_mass}")

    # Load the reference image to get its geometry
    ref_image = ants.image_read(output_transform_image)

    # Center of mass coordinates (in voxel space)
    com_voxel = center_of_mass.astype(int)

    # Ensure the coordinates are within the volume bounds
    if all(0 <= v < s for v, s in zip(com_voxel, ref_image.shape)):
        # Create a spherical mask at the center of mass
        sphere_mask = create_sphere_mask(com_voxel, ref_image.shape, sphere_radius)
    else:
        print("Center of mass coordinates are out of bounds")
        return None  # Skip this subject

    # Convert the NumPy array back to an ANTsImage
    sphere_image = ants.from_numpy(sphere_mask, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction)

    # Save the new volume
    output_sphere_image = os.path.join(output_directory, output_sphere_image_name)
    ants.image_write(sphere_image, output_sphere_image)

    print("Sphere image saved successfully.")

    return com_voxel


def ras_operations(t1_img, output_directory, subject, x, y, z, label_name, sphere_radius=None):
    # Define your T1w image file path
    t1w_image_file = t1_img

    # Load the T1w image
    img = nib.load(t1w_image_file)

    dimensions = img.shape
    x_dim, y_dim, z_dim = dimensions[:3]
    print(f"x dimension: {x_dim}")
    print(f"y dimension: {y_dim}")
    print(f"z dimension: {z_dim}")

    # Extract the RAS 2 VOX matrix
    ras2vox_matrix = img.affine[0:3, 0:3]

    # Print the RAS 2 VOX matrix
    print("RAS to VOX matrix:")
    print(ras2vox_matrix)

    # Convert voxel coordinates to RAS coordinates
    x_vox, y_vox, z_vox = x, z, y

    # Convert 1mm voxel coordinates to original voxel coordinates
    x_vox_original = x_vox / img.header.get_zooms()[0]
    y_vox_original = y_vox / img.header.get_zooms()[1]
    z_vox_original = z_vox / img.header.get_zooms()[2]

    # Convert 1mm voxel coordinates to inverted x coordinates
    inverted_x = x_dim - x_vox_original
    
    # Convert original voxel coordinates to RAS coordinates
    voxel_coords_original = np.array([x_vox_original, y_vox_original, z_vox_original])
    ras_coords = np.dot(img.affine, np.append(voxel_coords_original, 1))[:3]

    # Convert inverted x coordinates to RAS coordinates
    voxel_coords_inverted = np.array([inverted_x, y_vox_original, z_vox_original])
    ras_coords_inverted = np.dot(img.affine, np.append(voxel_coords_inverted, 1))[:3]

    # Print the converted RAS coordinates
    print(f"Original Voxel coordinates: ({x_vox_original}, {y_vox_original}, {z_vox_original})")
    print(f"RAS coordinates: ({ras_coords[0]}, {ras_coords[1]}, {ras_coords[2]})")
    print(f"Inverted X Coordinate: ({inverted_x}, {y_vox_original}, {z_vox_original})")
    print(f"Inverted x RAS coordinates: ({ras_coords_inverted[0]}, {ras_coords_inverted[1]}, {ras_coords_inverted[2]})")

    # Load the reference image to get its geometry
    ref_image = ants.image_read(t1w_image_file)

    # Center of mass coordinates (in voxel space)
    com_voxel = voxel_coords_original.astype(int)
    com_inverted_voxel = voxel_coords_inverted.astype(int)

    # Ensure the coordinates are within the volume bounds
    if all(0 <= v < s for v, s in zip(com_voxel, ref_image.shape)):
        # Create a spherical mask at the center of mass
        print(com_voxel, ref_image.shape, sphere_radius)
        sphere_mask = create_sphere_mask(com_voxel, ref_image.shape, sphere_radius)
    else:
        print("Center of mass coordinates are out of bounds")
        return None, None, None, None  # Skip this subject

    # Same for inverted
    if all(0 <= v < s for v, s in zip(com_inverted_voxel, ref_image.shape)):
        # Create a spherical mask at the center of mass
        print(com_inverted_voxel, ref_image.shape, sphere_radius)
        sphere_mask_inverted = create_sphere_mask(com_inverted_voxel, ref_image.shape, sphere_radius)
    else:
        print("Center of mass coordinates are out of bounds")
        return None, None, None, None  # Skip this subject

    # Convert the NumPy array back to an ANTsImage
    sphere_image = ants.from_numpy(sphere_mask, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction)
    sphere_image_inverted = ants.from_numpy(sphere_mask_inverted, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction)

    # Save the new volume
    output_sphere_image = os.path.join(output_directory, f"{subject}_desc-native_{label_name}.nii.gz")
    ants.image_write(sphere_image, output_sphere_image)
    print("Sphere image saved successfully.")

    # Save the new volume
    output_inverted_sphere_image = os.path.join(output_directory, f"{subject}_desc-native_{label_name}-invertedX.nii.gz")
    ants.image_write(sphere_image_inverted, output_inverted_sphere_image)
    print("Inverted sphere image saved successfully.")

    return output_sphere_image, output_inverted_sphere_image, com_voxel, com_inverted_voxel


def staging(input_t1, output_directory, subject, x, y, z, label_name=None):
    mni_template = os.path.join(os.getcwd(), "share/MNI152_T1_1mm.nii.gz")

    native_label_image, native_inverted_label_image, native_label_center, native_inverted_label_center = ras_operations(input_t1, output_directory, subject, x, y, z, label_name, sphere_radius=2)

    if native_label_image is None:
        print(f"Skipping subject {subject} due to out of bounds center of mass.")
        return

    mni_label_center = mni_operations(mni_template, input_t1, native_label_image, output_directory, output_transform_image_name=f"{subject}_desc-MNI_T1w.nii.gz", output_transformed_label_name=f"{subject}_desc-tmp_{label_name}.nii.gz", output_sphere_image_name=f"{subject}_desc-MNI_{label_name}.nii.gz", sphere_radius=2)

    if mni_label_center is None:
        print(f"Skipping MNI processing for subject {subject} due to out of bounds center of mass.")
        return 

    mni_inverted_label_center = mni_operations(mni_template, input_t1, native_inverted_label_image, output_directory, output_transform_image_name=f"{subject}_desc-MNI_T1w.nii.gz", output_transformed_label_name=f"{subject}_desc-tmp_{label_name}-invertedX.nii.gz", output_sphere_image_name=f"{subject}_desc-MNI_{label_name}-invertedX.nii.gz", sphere_radius=2)

    if mni_inverted_label_center is None:
        print(f"Skipping MNI processing for inverted X for subject {subject} due to out of bounds center of mass.")
        return 

    return mni_label_center, mni_inverted_label_center, native_label_center, native_inverted_label_center