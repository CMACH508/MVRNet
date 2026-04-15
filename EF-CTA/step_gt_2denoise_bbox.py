import os
import skimage
import numpy as np
import nibabel as nib
from os.path import join
import matplotlib.pyplot as plt
pid_list = []


cnt_multi_mask = 0
def denoise_one(mask_dir, save_dir, pid, denoise_bbox_file = None):
    valid_large_list = ['93660544+80', '93650714','94022263', '93794930', '93737392', '1916358', '94022263+392', '94026433', '94035988', '93736052+125', '93607728', '93732348', '93946367+318', '93836499', '2582996', '93829935', '93592578',
                        '3252170+132', '3237339+136', '2666889', '32ccf', '2685022', '4925251', '2943312', '3173614', '1507211',
                        'R1035291', '2745344+39', '54wcx', '2913634', '06zxy', '989030', '1895923', '850369', '902869',
                        '2092662', '93668311', '3190450', '93467510', '1884464', '16fxj', '1969960', '93479017', '2702781',
                        '2682297', '46jjx', 'R1054481', '51lcl']
    
    path_mask = join(mask_dir, '{}_mask.nii.gz'.format(pid))
    nii_img = nib.load(path_mask)
    img = nii_img.get_fdata()
    img = np.where(img>0.3, 1, 0)
    img = np.transpose(img, (2, 0, 1))
    # print(img.shape)
    labeled = skimage.measure.label(img, connectivity=2)
    region_props = skimage.measure.regionprops(labeled)
    print(' pid={} region={}'.format(pid, len(region_props)))

    tmp_img = img.copy()
    valid_region = len(region_props)
    is_denoised = False
    is_too_large = False
    for ii in range(len(region_props)):
        print(' {}-label:'.format(ii), region_props[ii].label)
        print(' {}-area:'.format(ii), region_props[ii].area)
        print(' {}-bbox:'.format(ii), region_props[ii].bbox)
        print(' {}-center:'.format(ii), region_props[ii].centroid)
        area = region_props[ii].area

        if area < 10:  # 面积为10的区域视为噪声点，用0来替换
            valid_region -= 1
            is_denoised = True 
            coords = region_props[ii].coords
            print(' coords.shape=', coords.shape)
            for (aa, bb, cc) in coords:
                tmp_img[aa, bb, cc] = 0
        elif area > 4000 and pid not in valid_large_list:
            # [idx pid area]
            # [24 3252170+132 5391] [68 3237339+136 31325] [101 2666889 6866] [102 32ccf 4231]
            # [182 3226752 6273] del [228 2685022 8988] [269 4925251 26089] [287 2943312 4082] [288 3173614 8886]
            # [319 1507211 16551]
            # [325 R1035291 8420] [338 2745344+39 5250] [341 54wcx 7005] [379 93830208] 不连通 reg=4
            # [436 2913634 8599] [440 06zxy 7189] [457 989030 4230] [482 1895923 8910]
            # [502 850369 9257] [508 902869 11369] [516 2092662 8978] [520 93668311 4227] [566 94035988 7424]
            # [579 3190450 6732] [595 93467510 4797] [605 1884464 7945] [727 16fxj 5082] [783 1969960 6129]
            # [789 93479017 9823] [790 2702781 14317] [802 2682297 4853] [814 46jjx 20344] [822 R1054481 6300]
            # [826 51lcl 4149]
            is_too_large = True
    if is_too_large:
        print('too large, skip pid={}'.format(pid))
        return False

    tmp_labeled = skimage.measure.label(tmp_img, connectivity=2)
    tmp_region_props = skimage.measure.regionprops(tmp_labeled)
    if is_denoised:
        if len(tmp_region_props) == valid_region:
            print(' ---denoise successfully---, new_valid_region=', valid_region)
        else:
            raise AssertionError('denoise failed')

    if valid_region == 0 or valid_region>5:
        print('valid_region=', valid_region)
        raise AssertionError('no valid mask or too many mask')
    if valid_region > 1:
        global cnt_multi_mask
        cnt_multi_mask += 1

    ### save tmp_img
    new_img = nib.Nifti1Image(np.transpose(tmp_img, (1, 2, 0)).astype(np.int8), np.eye(4))
    nib.save(new_img, join(save_dir, pid+"_mask.nii.gz"))
    # ### save bbox
    for ii in range(len(tmp_region_props)):
        a, b, c, aa, bb, cc = tmp_region_props[ii].bbox
        denoise_bbox_file.write('{} {} {} {} {} {} {}\n'.format(pid, a, b, c, aa-a, bb-b, cc-c))

    return is_denoised



def denoise(mask_dir, save_dir, denoise_bbox_file=None):
    # '3226752', '93830208'(不连通4联通其实是2),
    invalid_list = ['3226752', '93830208']
    cnt_denoise = 0
    all_list_ = os.listdir(mask_dir)
    all_list = [file.split('_')[0] for file in all_list_]
    all_list.sort()
    total_num = len(all_list)
    print(total_num)
    if denoise_bbox_file is not None:
        denoise_bbox_file = open(denoise_bbox_file, 'w')
    for i in range(total_num):
        print('\n[{}/{}] {}'.format(i, total_num, all_list[i]))
        pid = all_list[i]
        if pid in invalid_list:
            continue
        is_denoised = denoise_one(mask_dir, save_dir, pid, denoise_bbox_file)
        cnt_denoise += is_denoised
    denoise_bbox_file.close()
    print('cnt_denoised=', cnt_denoise) 
    print('cnt_multi_mask=', cnt_multi_mask)


if __name__ == '__main__':
    denoise_bbox_file = '/home/xxxxxxxxx/program/_Aneurysm_/MVRNet_zhong/data_prepare/lesion_bbox.txt'

    batch_names = ['batch1-NATURE-mr', 'batch2+3']
    for batch_name in batch_names:
        mask_dir = f'/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/{batch_name}/step_gt'
        save_dir = f'/data2/xxxxxxxxx/xxxxx/dataset/from_zhou/{batch_name}/step_gt_denoised'

        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        denoise(mask_dir, save_dir, denoise_bbox_file)
