data_root='./dataset/'
### MVRNet
model_type='datacat_refine_multi_add'
img_dir=$data_root'without_skull/image/'
mask_dir=$data_root'without_skull/mask/'
test_file=$data_root'without_skull/test.txt'
enhanced_dir=$data_root'with_skull/dicom2nifti_enhanced_v2/'
model_path='ckpt/MVRNet/model.pth.tar'
gpu='0,1'
batch_size=12
is_enhanced=1
out=$model_path'.del.out'
python -u evaluation.py --test_file $test_file --is_enhanced $is_enhanced --enhanced_dir $enhanced_dir \
    -b $batch_size --img_dir $img_dir --mask_dir $mask_dir -p $model_path --model_type $model_type \
    --gpu $gpu
