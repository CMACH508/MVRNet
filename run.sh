# dlia_refine_multi_view_add
gpu='0,1,2'
init_lr=0.0001
config='tasks/configs/aneurysm_seg.daresunet.yaml'
batch_size=8
model_type='dlia_refine_multi_view_add'
start_epoch=0
end_epoch=250
pretrain='none'
output_dir='output_dir'
start_valid=0
validate_freq=1
out='output.out'
nohup python -u main.py --gpu $gpu --validate_freq $validate_freq --start_valid $start_valid --start_epoch $start_epoch --end_epoch $end_epoch --init_lr $init_lr --output_dir $output_dir --model_type $model_type --batch_size $batch_size --config $config > $out 2>&1 &
