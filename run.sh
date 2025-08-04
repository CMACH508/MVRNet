data_root='./dataset/'
enhanced_dir=$data_root'with_skull/dicom2nifti_enhanced_v2/'
image_dir=$data_root'without_skull/image/'
mask_dir=$data_root'without_skull/mask/'
box=$data_root'without_skull/lesion_bbox.txt'
train_list=$data_root'without_skull/train.txt'
valid_list=$data_root'without_skull/valid.txt'

# MVRNet
gpu='0'
init_lr=0.0001
config='tasks/configs/aneurysm_seg.daresunet.yaml'
batch_size=3
model_type='datacat_refine_multi_add'
is_enhanced=1
start_epoch=0
end_epoch=300
pretrain='none'
output_dir='./exp/'
start_valid=0
validate_freq=500
out=$output_dir'log.out'
python -u main.py --gpu $gpu --validate_freq $validate_freq --start_valid $start_valid \
    --start_epoch $start_epoch --end_epoch $end_epoch --init_lr $init_lr --output_dir $output_dir \
    --model_type $model_type --batch_size $batch_size --config $config \
    --img_dir $image_dir --mask_dir $mask_dir \
    --is_enhanced $is_enhanced --enhanced_dir $enhanced_dir --box $box \
    --train_list $train_list --valid_list $valid_list\
    # > $out 2>&1 &

