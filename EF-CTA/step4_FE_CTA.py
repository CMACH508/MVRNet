import os
from multiprocessing import Process
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage import morphology
import time
from os.path import join
import nibabel as nib

def read_nii(file_path):
    try:
        # 尝试读取数据
        img = np.array(nib.load(file_path).dataobj)
        
        # 检查数据是否有效：
        # 1. img.size == 0 表示数据为空
        # 2. 0 in img.shape 表示某个维度长为0 (导致了你的报错 [1, 1, 0])
        if img.size == 0 or 0 in img.shape:
            return None
            
        return img
    except:
        # 如果文件损坏连 nib.load 都报错，直接返回 None
        return None


def preprocess(read_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sharpen = Sharpen().to(device)

    brain = read_nii(read_path)
    
    # --- 新增：如果是坏文件，直接结束函数，什么都不做 ---
    if brain is None:
        print(f"Skip Invalid File: {read_path}")
        return
    # ------------------------------------------------

    brain = torch.tensor(brain, dtype=torch.float32).to(device)
    brain = sharpen(brain)
    brain = brain.cpu().numpy().astype(np.int32)

    brain = dilate(brain)
    brain = erode(brain)
    
    brain = erode(brain)
    brain = denoise(brain)
    brain = dilate(brain)

    save_nii(brain, save_path)
    print("Saving: {}".format(save_path))





def save_nii(img, save_path):
    # out = sitk.GetImageFromArray(img)
    # sitk.WriteImage(out, save_path)
    new_img = nib.Nifti1Image(img.astype(np.float32), np.eye(4))
    nib.save(new_img, save_path)


class Sharpen(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, padding=1):
        super(Sharpen, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        kernel = torch.tensor([[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                               [[0, -1, 0], [-1, 7, -1], [0, -1, 0]],
                               [[0, 0, 0], [0, -1, 0], [0, 0, 0]]], dtype=torch.float32)
        self.conv.weight.data = kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, img, iter=1):
        # print("sharpening...")
        for _ in range(iter):
            img = img.unsqueeze(0).unsqueeze(0)
            sharp_img = self.conv(img)
            tmp = abs(sharp_img - img) / (img + 0.01)
            img = torch.where(tmp > 0.8, torch.min(img), img)
            img = img.squeeze(0).squeeze(0)
        return img


def get_kernel():
    kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                       [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    return kernel


def erode(img):
    # print("eroding...")
    kernel = get_kernel()
    eroded_img = ndimage.grey_erosion(img, footprint=kernel)
    return eroded_img


def dilate(img):
    # print("dilating...")
    kernel = get_kernel()
    dilated_img = ndimage.grey_dilation(img, footprint=kernel)
    return dilated_img


def denoise(img):
    # print("denoising...")
    min_area = 5000
    img = np.where(img < 100, 0, img)
    kernel = get_kernel()
    labeled_img, _ = ndimage.label(img, structure=kernel)
    labeled_img = morphology.remove_small_objects(labeled_img, min_area)
    img[labeled_img == 0] = 0
    return img
    

# def preprocess(read_path, save_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     sharpen = Sharpen().to(device)

#     brain = read_nii(read_path)
#     brain = torch.tensor(brain, dtype=torch.float32).to(device)
#     brain = sharpen(brain)
#     brain = brain.cpu().numpy().astype(np.int32)

#     brain = dilate(brain)
#     brain = erode(brain)
    
#     brain = erode(brain)
#     brain = denoise(brain)
#     brain = dilate(brain)

#     save_nii(brain, save_path)
#     print("Saving: {}".format(save_path))

# root = '/xxxxx-xxxx/xxxxxxxxx/program/_Aneurysm_/dataset/nature/with_skull/'
batch_names = ['batch1-NATURE-mr', 'batch2+3']
root = '/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/'
for batch_name in batch_names:
    source_dir = join(root, batch_name, 'step3_image_withskull')
    save_dir = join(root, batch_name, 'step4_image_FECTA')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    def my_thread(thd_id, start_index, idx_list, source_dir, save_dir):
        for ii, file in enumerate(idx_list):
            print("Thd-{} Processing: {}/{} {}".format(thd_id, ii, len(idx_list), file))
            with open(join(root, 'enhanced_{}.out'.format(thd_id)), 'a') as f:
                f.write('{} {}\n'.format(ii+start_index, file.replace('.nii.gz', '')))
            preprocess(join(source_dir, file), join(save_dir, file))


    if __name__ == '__main__':
        torch.multiprocessing.set_start_method('spawn', force=True)  # <== 这行！
        os.environ['CUDA_DEVICES_VISIBLE'] = '0,1'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sharpen = Sharpen().to(device)

        s = time.time()

        # file_list = open('/data3/xxxxxxxxx/nature/with_skull/test.txt').readlines()
        # file_list = [_file.strip()+'.nii.gz' for _file in file_list]
        file_list = os.listdir(source_dir)
        file_list.sort()

        thread_num = 2
        start = 0
        interval = 600
        thread_list = []
        for thd_id in range(thread_num):
            t = Process(target=my_thread, args=(thd_id, start, file_list[start: start + interval], source_dir, save_dir))
            thread_list.append(t)
            start += interval

        for i in range(thread_num):
            thread_list[i].start()
        for j in range(thread_num):
            thread_list[j].join()

        t = time.time()
        print("Time elapse: {}".format(t - s))

