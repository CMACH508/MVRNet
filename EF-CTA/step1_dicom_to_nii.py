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
import time
start_time = time.time()

data_root = '/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/batch2+3'
folder_name = join(data_root, "DICOM/")
out_path = join(data_root, "step1_image")
if not os.path.exists(out_path):
    os.mkdir(out_path)


folders = os.listdir(folder_name)
# folders = open('./nature_test.txt').readlines()
# folders = open('/xxxxx-xxxx/xxxxxxxxx/program/_Aneurysm_/FE-Aneurysm/NATURE/nature_train.txt').readlines()
# folders = [f.strip() for f in folders]

# 扫描一个患者的目录，加载所有的切片，按切换的z方向排序切片，并获取切片厚度
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

# ICME不经过这步
# nature 经过
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    ####################################################BIBM没进过####################################
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept) #（-1024）
    ################################################BIBM没经过######################################
    return np.array(image, dtype=np.int16)

for i, folder in enumerate(folders):
    print(i, folder)
    try:
        patient = load_scan(join(folder_name, folder))
        patient_pixels = get_pixels_hu(patient)
        print(patient_pixels.shape)  # (279, 512, 512)
        # print(patient_pixels.max())
        # print(patient_pixels.min())

        # plt.hist(patient_pixels.flatten(), bins=80, color='c')
        # plt.xlabel("Hounsfield Units (HU)")
        # plt.ylabel("Frequency")
        # plt.show()
        # patient_pixels = np.where(patient_pixels<150, 0, patient_pixels)
        # plt.imshow(patient_pixels[12], cmap=plt.cm.gray)
        # plt.show()

        image_refer = patient_pixels.copy()
        c, w, h = patient_pixels.shape
        for i in range(c):
            patient_pixels[i] = image_refer[c - i - 1]  # 倒一遍

        new_img = nib.Nifti1Image(patient_pixels, np.eye(4))  #
        nib.save(new_img, out_path + "/" + folder + ".nii.gz")
        # print('save:', out_path + "/" + folder + ".nii.gz")
    except:
        continue

print('Finished')
end_time = time.time()
run_time = round(end_time-start_time)
hour = run_time//3600
minute = (run_time-3600*hour)//60
second = run_time-3600*hour-60*minute
print('cost time：{}h:{}m:{}s'.format(hour, minute, second))