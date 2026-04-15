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

## 重采样
batch_names = ['batch1-NATURE-mr', 'batch2+3']
step_names = ["step1_image", "step2_image"]
for batch_name in batch_names:
    data_root = join('/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/', batch_name)
    folder_name = join(data_root, "DICOM")
    for step_name in step_names:
        in_path = join(data_root, step_name)
        if step_name == "step1_image":
            out_path = join(data_root, "step3_image_withskull")
        else:
            out_path = join(data_root, "step3_image_withoutskull")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        folders = os.listdir(in_path)


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
            # print(image.max())
            # print(image.min())
            # Convert to int16 (from sometimes int16),
            # should be possible as values should always be low enough (<32k)
            image = image.astype(np.int16)
            # Set outside-of-scan pixels to 0
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0
            # Convert to Hounsfield units (HU)
            for slice_number in range(len(slices)):

                intercept = slices[slice_number].RescaleIntercept
                slope = slices[slice_number].RescaleSlope

                if slope != 1:
                    image[slice_number] = slope * image[slice_number].astype(np.float64)
                    image[slice_number] = image[slice_number].astype(np.int16)

                image[slice_number] += np.int16(intercept)

            return np.array(image, dtype=np.int16)


        def resample(image, scan, new_spacing=[0.5, 0.5, 0.5]):
            # Determine current pixel spacing
            # spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
            # spacing = np.array(list(spacing))
            spacing = [0.5, 0.5, 0.5]
            print(scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1])
            new_spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]])
            print("spacing  ", spacing)
            # resize_factor = spacing / new_spacing
            resize_factor = new_spacing / spacing
            print(resize_factor)
            new_real_shape = image.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / image.shape
            new_spacing = spacing / real_resize_factor
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
            return image, new_spacing


        for ii, file in enumerate(folders):
            print(ii, "#######################")
            img = np.array(nib.load(join(in_path, file)).dataobj)  # 去骨后的nii
            print(img.shape)
            print(folder_name + '/' + file.split(".")[0])
            patient = load_scan(folder_name + '/' + file.split(".")[0])
            print("len(patient):   ", len(patient))
            # patient_pixels = np.transpose(img,(2,0,1))
            pix_resampled, spacing = resample(img, patient, [0.5, 0.5, 0.5])
            print(pix_resampled.shape)
            pix_resampled = np.transpose(pix_resampled, (1, 2, 0))
            new_img = nib.Nifti1Image(pix_resampled, np.eye(4))
            nib.save(new_img, join(out_path, file))

