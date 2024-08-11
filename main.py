import os
import shutil
import argparse
import os, sys
from os.path import join
sys.path.extend([join(os.path.abspath('.'), 'tasks/aneurysm'), join(os.path.abspath('.'), 'tasks/aneurysm/nets')])
parser = argparse.ArgumentParser()
from utils.config import cfg
from utils.tools.logger import Logger as Log
from tasks.main import main_worker
from utils.tools.file import mkdir_safe
parser.add_argument('--train', action='store_true', help='command for train', default=True)
parser.add_argument('--test', action='store_true', help='command for my_test')
parser.add_argument('--config', help='configure file path',
                    default='tasks/configs/aneurysm_seg.daresunet.yaml')
parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint pth')
# parser.add_argument('--gpu', nargs='+', type=int, default=[0,1], help='which gpu to select')
parser.add_argument('--check_point', default=None, help='the check point path to store')
parser.add_argument('--init_lr', default=0.0001, type=float, help='init learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--gpu', type=str, default='2,3', help='which gpu to select')
parser.add_argument('--model_type', default='dlia', type=str, help='model type')
# parser.add_argument('--model_type', default='multi_view', type=str, help='model type')
parser.add_argument('--output_dir', default='./output_del', type=str, help='output dir')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--img_dir', default='', type=str, help='path of img_dir')
parser.add_argument('--mask_dir', default='', type=str, help='mask_dir')
parser.add_argument('--enhanced_dir', default=None, type=str, help='enhanced_dir')
parser.add_argument('--train_list', default=None, type=str, help='train_list')
parser.add_argument('--valid_list', default=None, type=str, help='valid_list')
parser.add_argument('--is_enhanced', default=0, type=int, help='input contains enhanced data')
parser.add_argument('--box', default='', type=str, help='box')
parser.add_argument('--start_valid', default=1, type=int, help='start_valid')
parser.add_argument('--start_epoch', default=0, type=int, help='start_epoch')
parser.add_argument('--end_epoch', default=150, type=int, help='end_epoch')
parser.add_argument('--validate_freq', default=4, type=int, help='validate_freq')

def main():
    args = parser.parse_args()
    print(args)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['TRAIN.RESUME', args.resume, 'TRAIN.DATA.BATCH_SIZE', args.batch_size,
                         'MODEL.NAME', args.model_type, 'OUTPUT_DIR', args.output_dir,
                         'SOLVER.BASE_LR', args.init_lr, 'MODEL.PRETRAIN',
                         args.pretrain, 'TRAIN.DATA.NII_FOLDER', args.img_dir,
                         'TRAIN.DATA.ENHANCED_FOLDER', args.enhanced_dir,
                         'TEST.DATA.ENHANCED_FOLDER', args.enhanced_dir,
                         'TRAIN.DATA.IS_ENHANCED', args.is_enhanced,
                         'TRAIN.DATA.ANEURYSM_FOLDER', args.mask_dir,
                         'TRAIN.DATA.ANEURYSM_BBOX', args.box,
                         'SOLVER.START_EPOCHS', args.start_epoch, 'SOLVER.EPOCHS', args.end_epoch,
                         'TRAIN.VALIDATE_FREQUENCE', args.validate_freq,
                         'TRAIN.DATA.TRAIN_LIST', args.train_list, 'TRAIN.DATA.VAL_LIST', args.valid_list,
                         'TRAIN.START_VALIDATE', args.start_valid,
                         'MODEL.INPUT_CHANNEL', 1])

    print('cfg.MODEL.NAME={}'.format(cfg.MODEL.NAME))
    if args.train:
        cfg.TASK.STATUS = 'train'
        # cfg.OUTPUT_DIR = os.path.join('./results', cfg.TASK.NAME, 'train', cfg.OUTPUT_DIR)
        mkdir_safe(cfg.OUTPUT_DIR)

        Log.init(logfile_level='info',
                 log_file=os.path.join(cfg.OUTPUT_DIR, 'logger.log'),
                 stdout_level='info')

        cfg.TRAIN.RESUME = args.resume

        if args.config != os.path.join(cfg.OUTPUT_DIR, os.path.basename(args.config)):
            # shutil.copyfile(args.config, os.path.join(cfg.OUTPUT_DIR, os.path.basename(args.config)))
            with open(os.path.join(cfg.OUTPUT_DIR, os.path.basename(args.config)), 'w') as ff:
                ff.write(str(cfg))

    elif args.test:
        cfg.TASK.STATUS = 'test'
        mkdir_safe(cfg.TEST.SAVE_DIR)

        Log.init(logfile_level='info',
                 log_file=os.path.join(cfg.TEST.SAVE_DIR, 'logger.log'),
                 stdout_level='info')

        ct_pth = cfg.TEST.MODEL_PTH
        if args.check_point:
            ct_pth = args.check_point

        if os.path.exists(os.path.join('./results', cfg.TASK.NAME, 'train', cfg.OUTPUT_DIR, ct_pth)):
            ct_pth = os.path.join('./results', cfg.TASK.NAME, 'train', cfg.OUTPUT_DIR, ct_pth)
        cfg.TEST.MODEL_PTH = ct_pth

    cfg.freeze()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(cfg)
    main_worker(args)


if __name__ == '__main__':
    main()
