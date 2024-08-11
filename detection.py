import numpy as np
import nibabel as nib
from os.path import join


def IOU_3d(Reframe, GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为 up down left right front back
    """
    up1 = Reframe[0]
    left1 = Reframe[2]
    front1 = Reframe[4]

    height1 = Reframe[1] - Reframe[0]
    width1 = Reframe[3] - Reframe[2]
    thickness1 = Reframe[5] - Reframe[4]

    up2 = GTframe[0]
    left2 = GTframe[2]
    front2 = GTframe[4]

    height2 = GTframe[1] - GTframe[0]
    width2 = GTframe[3] - GTframe[2]
    thickness2 = GTframe[5] - GTframe[4]

    end_up = max(up1 + height1, up2 + height2)
    start_up = min(up1, up2)
    height = height1 + height2 - (end_up - start_up)

    end_left = max(left1 + width1, left2 + width2)
    start_left = min(left1, left2)
    width = width1 + width2 - (end_left - start_left)

    end_front = max(front1 + thickness1, front2 + thickness2)
    start_front = min(front1, front2)
    thickness = thickness1 + thickness2 - (end_front - start_front)

    if width <= 0 or height <= 0 or thickness <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width * height * thickness;  # 两立方体相交体积
        Area1 = width1 * height1 * thickness1;
        Area2 = width2 * height2 * thickness2;
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    # print(ratio)
    return ratio

def bound(brain, index):
    c, h, w = brain.shape
    print('--> gt_brain.shape={}'.format(brain.shape))  # (320, 359, 359)
    up = 0
    down = c
    left = 0
    right = h
    front = 0
    back = w
    for i in range(c):
        # print(i)
        if np.all(brain[i] < index):
            continue
        else:
            up = i
            break
    for i in range(c):
        if np.all(brain[c - i - 1] < index):
            continue
        else:
            down = c - i - 1
            break

    brain_1 = brain.transpose((2, 1, 0))
    for i in range(w):
        if np.all(brain_1[i] < index):
            continue
        else:
            left = i
            break
    for i in range(w):
        if np.all(brain_1[w - i - 1] < index):
            continue
        else:
            right = w - i - 1
            break

    brain_2 = brain.transpose((1, 0, 2))
    for i in range(w):
        if np.all(brain_2[i] < index):
            continue
        else:
            front = i
            break
    for i in range(w):
        if np.all(brain_2[w - i - 1] < index):
            continue
        else:
            back = w - i - 1
            break
    return up, down, left, right, front, back


gt = np.array(nib.load(join(gt_nii_dir, im_name_dir+ ".nii.gz")).dataobj)
gt_up, gt_down, gt_left, gt_right, gt_front, gt_back = bound(gt, 1)
gt_box = [gt_up, gt_down, gt_left, gt_right, gt_front, gt_back]
print("--> gt_box:", gt_box)
for box in result_box:
    print('--> IOU_3d=', IOU_3d(gt_box, box))
    if (IOU_3d(gt_box, box) >= 0.01):
        detected = "yes"