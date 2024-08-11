import torch
import numpy as np


def get_seg_TP_TN_FP_FN(result, target):
    true = target.astype(np.bool)
    pred = result.astype(np.bool)
    TP = (true & pred).sum()
    TN = (~true & ~pred).sum()
    FP = (~true & pred).sum()
    FN = (true & ~pred).sum()
    return TP, TN, FP, FN

def Accuracy(TP, TN, FP, FN):
    if (TP+TN+FP+FN == 0):
        return 0
    return (TP+TN)/(TP+TN+FP+FN)

def Miou(TP, TN, FP, FN):
    if (FP + FN + TP) == 0:
        return 0
    # print('TP={} TN={} FP={} FN={}'.format(TP, TN, FP, FN))
    return float(TP) / float(FP + FN + TP)

def Dice(TP, TN, FP, FN):
    if (FP + FN + 2 * TP == 0):
        return 0
    return float(2 * TP) / float(FP + FN + 2 * TP)

def Precision(TP, TN, FP, FN):
    if (FP + TP == 0):
        return 0
    return float(TP) / float(FP + TP)

def TPR(TP, TN, FP, FN):
    if (TP + FN == 0):
        return 0
    return float(TP) / float(TP + FN)


# def Miou(result, target):
#     true = target.astype(np.bool)
#     pred = result.astype(np.bool)
#     TP = (true & pred).sum()
#     TN = (~true & ~pred).sum()
#     FP = (~true & pred).sum()
#     FN = (true & ~pred).sum()
#     if (FP + FN + TP) == 0:
#         return 0
#     # print('TP={} TN={} FP={} FN={}'.format(TP, TN, FP, FN))
#     return float(TP) / float(FP + FN + TP)


# def Dice(result, target):
#     true = target.astype(np.bool)
#     pred = result.astype(np.bool)
#     TP = (true & pred).sum()
#     TN = (~true & ~pred).sum()
#     FP = (~true & pred).sum()
#     FN = (true & ~pred).sum()
#     return float(2 * TP) / float(FP + FN + 2 * TP)


# def Precision(result, target):
#     true = target.astype(np.bool)
#     pred = result.astype(np.bool)
#     TP = (true & pred).sum()
#     TN = (~true & ~pred).sum()
#     FP = (~true & pred).sum()
#     FN = (true & ~pred).sum()
#     if FP + TP == 0:
#         return 0
#     return float(TP) / float(FP + TP)


# def TPR(result, target):
#     true = target.astype(np.bool)
#     pred = result.astype(np.bool)
#     TP = (true & pred).sum()
#     TN = (~true & ~pred).sum()
#     FP = (~true & pred).sum()
#     FN = (true & ~pred).sum()
#     return float(TP) / float(TP + FN)

def F1(Precision, Recall):
    if (Precision + Recall) == 0:
        F1_score = 0
    else:
        F1_score = (2 * Precision * Recall) / (Precision + Recall)
    return F1_score


''' detection '''
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
        Area1 = width1 * height1 * thickness1
        Area2 = width2 * height2 * thickness2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    # print(ratio)
    return ratio

def get_dect_TP_TN_FP_FN(mask_bboxs, pred_bboxs, thrd_iou): # thrd_iou=0.01
    TP, TN, FP, FN = 0, 0, 0, 0
    has_matched = []
    pred_num = len(pred_bboxs)
    for ii, mask_box in enumerate(mask_bboxs):
        max_iou, max_iou_index = -1, 0
        for jj, pred_box in enumerate(pred_bboxs):
            if jj in has_matched:
                continue
            tmp_iou = IOU_3d(pred_box, mask_box)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                max_iou_index = jj
        if max_iou >= thrd_iou:
            TP += 1
            has_matched.append(max_iou_index)
        else:
            FN += 1
    FP = pred_num - TP
    return TP, TN, FP, FN



