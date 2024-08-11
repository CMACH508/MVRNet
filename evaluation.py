# _date_:2022/6/5 9:49
import os
import cv2
import time
import datetime
import numpy as np
import nibabel as nib
from glob import glob
from skimage import morphology
from skimage import measure
from utils.config import cfg
from tasks.aneurysm.datasets.aneurysm_dataset import ANEURYSM_SEG
import os, sys
from os.path import join
import argparse
from torch import cat
sys.path.extend([join(os.path.abspath('.'), 'tasks/aneurysm/nets'),
                 join(os.path.abspath('.'), 'tasks/aneurysm/nets_global'),
                 join(os.path.abspath('.'), 'tasks/aneurysm')])
import torch
import torch.backends.cudnn as cudnn
from metrics_lpy import *
from lpy_test import read_size


def seg_metrics_add(acc_dict, iou, dice, precision, tpr, f1):
    acc_dict['iou'] += iou
    acc_dict['dice'] += dice
    acc_dict['precision'] += precision
    acc_dict['tpr'] += tpr
    acc_dict['f1'] += f1
    return acc_dict

def detect_metrics_add(acc_dict, precision, tpr, f1, fp_percase):
    acc_dict['precision'] += precision
    acc_dict['tpr'] += tpr
    acc_dict['f1'] += f1
    acc_dict['fp_percase'] += fp_percase
    return acc_dict

def img2bboxs(img):
    # img = morphology.remove_small_objects(img, min_size=4, connectivity=2, in_place=False)
    label_img = measure.label(img, connectivity=2)
    # print('regions number:', label_img.max() + 1)
    regions = measure.regionprops(label_img)
    bbox_list = []
    for reg in regions:
        if reg.area < 4:
            continue
        bbox = reg.bbox  # 边界外接框(min_row, min_col, max_row, max_col)
        # print(bbox)
        # (115, 212, 175, 129, 231, 184)
        # (116, 213, 175, 129, 231, 186)
        # 转成 gt_up, gt_down, gt_left, gt_right, gt_front, gt_back
        bbox_list.append([bbox[0], bbox[3], bbox[1], bbox[4], bbox[2], bbox[5]])
    return bbox_list

def evaluate_results(BATCH_SIZE, gpus, model_type, model_path, image_root, mask_root, test_pids_file=None):
    # params init
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    print('USE GPU: ', gpus)

    WORKERS = 8
    # BATCH_SIZE = 16
    # BATCH_SIZE = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WL, WW = 450, 900
    print('USE WL_WW:', WL, WW)

    # evaluation data list
    test_pids_list = None
    
    print('mask_root=', mask_root)
    print('image_root=', image_root)
    print('test_pids_file=', test_pids_file)
    # print('test_pids_list=', len(test_pids_list))

    # segmentation save
    # STORE_MASK = False
    # SAVE_DIR = './raws/output'
    # VIS_ROOT = './raws/test_vis'

    # if not os.path.exists(SAVE_DIR):
    #     os.mkdir(SAVE_DIR)
    # if not os.path.exists(VIS_ROOT):
    #     os.mkdir(VIS_ROOT)

    cfg.TASK.STATUS = 'test'
    cfg.TEST.DATA.NII_FOLDER = image_root
    cfg.TRAIN.DATA.WL_WW = WL, WW

    cudnn.benchmark = True
    cudnn.deterministic = True
    ## model init
    # net = DAResUNet(segClasses=2, k=32, input_channel=1)
    net = None
    k = 32
    cfg.merge_from_list(['MODEL.NAME', args.model_type])
    if model_type == 'dlia':
        k=16
        from tasks.aneurysm.nets.aneurysm_net_dlia import DAResUNet
    if model_type == 'dlia_EFCTA':
        k=16
        from tasks.aneurysm.nets.aneurysm_net_dlia import DAResUNet
        net = DAResUNet(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL*2)
    if model_type == 'dlia_refine_v2':
        k = 16
        from tasks.aneurysm.nets.aneurysm_net_dlia_refine_v2 import DAResUNet
    elif model_type == 'dlia_multi_view_add' or model_type == 'dlia_multi_view_cat' or \
            model_type == 'dlia_refine_multi_view_add' or model_type == 'dlia_refine_multi_view_cat':
        k = 16
        from tasks.aneurysm.nets.aneurysm_multi_view_3d import MultiView
        net = MultiView(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL, name=model_type)
    elif model_type == 'datacat_refine_multi_add':
        k = 16
        from nets.aneurysm_multi_view_3d import MultiView
        net = MultiView(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL * 2, name=model_type)
    elif model_type == 'dlia_datacat_add':
        k = 16
        from nets.aneurysm_multi_view_3d import MultiView
        net = MultiView(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL * 2, name=model_type)
    elif model_type == 'dlia_datacat4_add':
        k = 16
        from nets.aneurysm_multi_view_3d import MultiView
        net = MultiView(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL * 4, name=model_type)
    elif model_type == 'dlia_dual_u':
        k = 16
        from tasks.aneurysm.nets.aneurysm_net_dlia_dual_u import Dual_U
        net = Dual_U(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL, name='dlia_multi_view_add')
    if net is None:
        net = DAResUNet(cfg.MODEL.NCLASS, k=k, psp=False, input_channel=cfg.MODEL.INPUT_CHANNEL)
    print('k={}'.format(k))
    print('------------- {} --------------'.format(model_type))

    params = torch.load(model_path, map_location='cpu')
    net.load_state_dict(params['model'])
    print('current model epoch=%d' % params['epoch'])
    print('load successfully, model_path={}'.format(model_path))
    net = net.to(device)
    gpus = gpus.split(',')
    if len(gpus) > 1:
        net = torch.nn.DataParallel(net, device_ids=range(len(gpus)))
    net.eval()

    ## data init
    EVAL_FILES = None
    EVAL_FILES = test_pids_file
    EVAL_LIST = []
    EVAL_LIST = test_pids_list
    if EVAL_FILES is not None and os.path.exists(EVAL_FILES):
        with open(EVAL_FILES, 'r') as f:
            lines = f.readlines()
            subjects = [line.strip() for line in lines]
    elif EVAL_LIST is not None:
        subjects = EVAL_LIST
    else:
        subjects = []
    # subjects.sort()
    print("the eval data count: %d" % len(subjects))

    ## eval
    eval_results = {}
    cuda_times = []
    seg_dict = {'iou': 0, 'dice': 0, 'precision': 0, 'tpr': 0, 'f1': 0}
    seg_first_50_dic = {'iou': 0, 'dice': 0, 'precision': 0, 'tpr': 0, 'f1': 0}
    seg_last_50_dic = {'iou': 0, 'dice': 0, 'precision': 0, 'tpr': 0, 'f1': 0}
    detect_dict = {'precision': 0, 'tpr': 0, 'f1': 0, 'fp_percase': 0}
    detect_first_50_dic = {'precision': 0, 'tpr': 0, 'f1': 0, 'fp_percase': 0}
    detect_last_50_dic = {'precision': 0, 'tpr': 0, 'f1': 0, 'fp_percase': 0}
    print('--------------- is_enhanced = {} -----------------'.format(args.is_enhanced))

    with torch.no_grad():
        for step, subject in enumerate(subjects, start=1):
            print('[{}/{}] {}'.format(step, len(subjects), subject))
            para_dict = {"subject": subject}  # subject_id
            eval_set = ANEURYSM_SEG(para_dict, "test", is_enhanced=args.is_enhanced)
            kwargs = {'shuffle': False, 'pin_memory': True,
                      'drop_last': False, 'batch_size': BATCH_SIZE, 'num_workers': WORKERS}
            data_loader = torch.utils.data.DataLoader(eval_set, **kwargs)
            v_x, v_y, v_z = eval_set.volume_size()  # 296,512,512
            # print('v_x, v_y, v_z={},{},{}'.format(v_x, v_y, v_z))
            other_infos = eval_set.get_other_infos()  # for seg eval

            seg = torch.FloatTensor(v_x, v_y, v_z).zero_()
            seg = seg.to(device)

            time_start = time.time()
            for i, (image, enhanced, coord) in enumerate(data_loader):
                image = image.to(device)
                # print('image.shape={}'.format(image.shape))  # [b, 1, 80, 80, 80]
                # print('enhanced.shape={}'.format(enhanced.shape))  # [b, 1, 80, 80, 80]
                if model_type == 'datacat_refine_multi_add' or model_type == 'dlia_datacat_add' or model_type == 'dlia_EFCTA':
                    enhanced = enhanced.to(device)
                    out = net(cat([image, enhanced], dim=1))
                # if cfg.TEST.DATA.VESSEL_FOLDER is not None:
                #     out = net(image, vessel)
                else:
                    out = net(image)
                # print('out.shape={}'.format(out['y'].shape))
                pred = torch.nn.functional.softmax(out['y'], dim=1)
                for idx in range(image.size(0)):
                    sx, ex = coord[idx][0][0], coord[idx][0][1]
                    sy, ey = coord[idx][1][0], coord[idx][1][1]
                    sz, ez = coord[idx][2][0], coord[idx][2][1]

                    seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]  # accsum

            time_end = time.time()
            seg = (seg >= 0.30).cpu().numpy().astype(np.uint8)  # binary, mask
            seg = np.transpose(seg, (1, 2, 0))

            # print('pred.shape=', seg.shape)  # 512,512,296

            mask_path = '%s/%s_mask.nii.gz' % (mask_root, subject)
            # if not os.path.exists(mask_path):
            #     mask_path = '%s/%s.nii.gz' % (mask_root, subject)
            mask = nib.load(mask_path).get_fdata().astype(np.uint8)
            # print(mask.max(), mask.min(), mask.shape)  # (512, 512, 296)
            print('pred.shape={} mask.shape={} pred_sum={} mask_sum={}'.format(
                seg.shape, mask.shape, seg.sum(), mask.sum()))  # (512, 512, 296)
            TP, TN, FP, FN = get_seg_TP_TN_FP_FN(seg, mask)
            iou = Miou(TP, TN, FP, FN).__round__(6)
            dice = Dice(TP, TN, FP, FN).__round__(6)
            precision = Precision(TP, TN, FP, FN).__round__(6)
            tpr = TPR(TP, TN, FP, FN).__round__(6)
            f1 = F1(precision, tpr).__round__(6)
            seg_dict = seg_metrics_add(seg_dict, iou, dice, precision, tpr, f1)

            print('seg tmp iou={:.6f} dice={:.6f} precision={:.6f} tpr={:.6f} f1={:.6f}'.format(iou, dice, precision, tpr, f1))
            print('seg avg iou={:.6f} dice={:.6f} precision={:.6f} tpr={:.6f} f1={:.6f}'.format(seg_dict['iou'] / step, seg_dict['dice'] / step,
                                                                                            seg_dict['precision'] / step, seg_dict['tpr'] / step,
                                                                                            seg_dict['f1'] / step))
            if step <= 50:
                seg_first_50_dic = seg_metrics_add(seg_first_50_dic, iou, dice, precision, tpr, f1)
            if step > 50:
                seg_last_50_dic = seg_metrics_add(seg_last_50_dic, iou, dice, precision, tpr, f1)
            # if STORE_MASK:
            #     eval_set.save(seg.copy(), SAVE_DIR)
            # eval_results[subject] = np.transpose(seg, (1, 2, 0)), other_infos  # (z.x,y) => (x,y,z)
            cuda_time = time_end - time_start
            cuda_times += [cuda_time]
            # print(datetime.datetime.now(), \
            #       '%d/%d: %s finished! cuda time=%.2f s!' % (step, len(subjects), subject, cuda_time))
            # torch.cuda.empty_cache()

            ''' detection '''
            mask_bboxs = img2bboxs(mask)
            pred_bboxs = img2bboxs(seg)
            print('aneurysm_num: Label={} pred={}'.format(len(mask_bboxs), len(pred_bboxs)))
            TP, TN, FP, FN = get_dect_TP_TN_FP_FN(mask_bboxs, pred_bboxs, thrd_iou=0.01)
            precision = Precision(TP, TN, FP, FN).__round__(6)
            tpr = TPR(TP, TN, FP, FN).__round__(6)
            f1 = F1(precision, tpr).__round__(6)
            detect_dict = detect_metrics_add(detect_dict, precision, tpr, f1, FP)
            print('detected num={}'.format(TP))
            print('detect tmp precision={:.6f} tpr={:.6f} f1={:.6f} FP={}'.format(precision, tpr, f1, FP))
            print('detect avg precision={:.6f} tpr={:.6f} f1={:.6f} FP={}'.format(detect_dict['precision'] / step, detect_dict['tpr'] / step,
                                                                                      detect_dict['f1'] / step, detect_dict['fp_percase'] / step))
            if step <= 50:
                detect_first_50_dic = detect_metrics_add(detect_first_50_dic, precision, tpr, f1, FP)
            if step > 50:
                detect_last_50_dic = detect_metrics_add(detect_last_50_dic, precision, tpr, f1, FP)

        summary = '\nResult:\nSegmentation: '
        for k, v in seg_dict.items():
            summary += '{}={:.6f} '.format(k, v / len(subjects))
        summary += '\nDetection:    '
        for k, v in detect_dict.items():
            summary += '{}={:.6f} '.format(k, v / len(subjects))

        first_50 = '\nfirst_50:\nSegmentation: '
        for k, v in seg_first_50_dic.items():
            first_50 += '{}={:.6f} '.format(k, v / 50)
        first_50 += '\nDetection:    '
        for k, v in detect_first_50_dic.items():
            first_50 += '{}={:.6f} '.format(k, v / 50)

        last_50 = '\nlast_50:\nSegmentation: '
        for k, v in seg_last_50_dic.items():
            last_50 += '{}={:.6f} '.format(k, v / 50)
        last_50 += '\nDetection:    '
        for k, v in detect_last_50_dic.items():
            last_50 += '{}={:.6f} '.format(k, v / 50)

        print(summary)
        print(first_50)
        print(last_50)

        print('\ntotal data: %.2s,avg cuda time: %.2s' % \
              (len(cuda_times), 1. * sum(cuda_times) / len(cuda_times)))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0,1')
parser.add_argument('-b', '--batch_size', type=int, default=6)
parser.add_argument('-p','--model_path', type=str, default=None)
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--img_dir', type=str, default=None)
parser.add_argument('--mask_dir', type=str, default=None)
parser.add_argument('--enhanced_dir', type=str, default=None)
parser.add_argument('--test_file', type=str, default=None)
parser.add_argument('--is_enhanced', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    # %%
    # dlia, dlia_refine_v2, dlia_multi_view_add, dlia_multi_view_cat, dlia_refine_multi_view_add, dlia_refine_multi_view_cat
    model_type = args.model_type
    model_path = args.model_path
    if model_path is None:
        model_type = 'dlia'
        model_path = 'model.pth.tar'

    BATCH_SIZE = args.batch_size
    gpus = args.gpu
    print('model_type=', model_type)
    print('model_path=', model_path)
    print('enhanced_dir=', args.enhanced_dir)
    print('is_enhanced=', args.is_enhanced)
    cfg_path = 'tasks/configs/aneurysm_seg.daresunet.yaml'
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(['TEST.DATA.ENHANCED_FOLDER', args.enhanced_dir])
    evaluate_results(BATCH_SIZE, gpus, model_type, model_path, args.img_dir, args.mask_dir, args.test_file)

    print('\nmodel_type=', model_type)
    print('model_path=', model_path)
