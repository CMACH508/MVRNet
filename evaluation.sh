#### dlia_refine_multi_view_add
model_type='dlia_refine_multi_view_add'
data_root='dataset/nature/without_skull/'
model_path='model.pth.tar'
gpu='2,3'
batch_size=12
out=$model_path'.out'
python -u fm_evaluation.py --data_root $data_root -p $model_path --model_type $model_type --gpu $gpu
nohup python -u fm_evaluation.py -b $batch_size --data_root $data_root -p $model_path --model_type $model_type --gpu $gpu > $out 2>&1 &
