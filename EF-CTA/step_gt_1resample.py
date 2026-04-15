import os
import pydicom as dicom
import numpy as np
import SimpleITK
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import scipy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, feature
from os.path import join

batch_names = ['batch1-NATURE-mr', 'batch2+3']

for batch_name in batch_names:
    in_path = join("/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/", batch_name, "step2_image")
    out_path = join("/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/", batch_name, "step_gt")
    gt_path = join("/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/", batch_name, "gt")
    folder_name = join("/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/", batch_name, "DICOM")

    # Load the scans in given folder path
    def load_scan(path):
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    def get_pixels_hu(slices):
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)


    def resample(image, scan, new_spacing=[0.5, 0.5, 0.5]):
        spacing = [0.5, 0.5, 0.5]
        # print(scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1])
        new_spacing_arr = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]])
        # print("spacing  ", spacing)
        
        resize_factor = new_spacing_arr / spacing
        # print(resize_factor)
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        # new_spacing = spacing / real_resize_factor # Unused variable
        
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing_arr # Returned updated spacing just in case

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    folders = os.listdir(in_path)

    for ii, file in enumerate(folders):
        print(ii, "#######################")
        
        # -------------------------------------------------------------
        # Determine GT filename based on batch_name
        # -------------------------------------------------------------
        if batch_name == 'batch2+3':
            # Assumes 'file' is something like 'ID.nii.gz', converts to 'ID_Merge.nii'
            gt_filename = file.replace('.nii.gz', '_Merge.nii')
        else:
            # Keep original name (e.g., 'ID.nii.gz')
            gt_filename = file

        gt_file_full_path = join(gt_path, gt_filename)
        # -------------------------------------------------------------

        if not os.path.exists(gt_file_full_path):
            print("[SKIP] missing gt:", gt_file_full_path)
            continue
        
        # Load the image using the determined path
        img = np.array(nib.load(gt_file_full_path).dataobj) 

        # Transpose logic
        if img.shape[0] == img.shape[1] and img.shape[1] == 512:
            img = np.transpose(img, (2, 0, 1))[::-1, :, :]
        
        assert img.shape[1] == img.shape[2]
        print(img.shape)

        # Determine DICOM path (stripping extension from the original file name)
        dicom_path = folder_name + "/" + file.split(".")[0]
        print(dicom_path)
        
        try:
            patient = load_scan(dicom_path)
        except Exception as e:
            print(f"[ERROR] Could not load DICOM from {dicom_path}: {e}")
            continue

        print("len(patient):   ", len(patient))
        
        pix_resampled, spacing = resample(img, patient, [0.5, 0.5, 0.5])
        print(pix_resampled.shape)
        
        pix_resampled = np.transpose(pix_resampled, (1, 2, 0))
        new_img = nib.Nifti1Image(pix_resampled, np.eye(4))
        
        # Save output
        nib.save(new_img, join(out_path, file.replace('.nii.gz', '_mask.nii.gz')))