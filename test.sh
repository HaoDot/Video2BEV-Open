# cd /media/xgroup/data/xgroup/hao/codes/others/university_etc/itm/rebuttal/itm_backbones/;
cd ~/Ours/Video2BEV;
gpu_ids='1';
# python_path='/home/xgroup/anaconda3/envs/torch1131/bin/python';
python_path=/home/juhao/miniconda3/envs/torch220/bin/python
# 45
dataset_path='xxxxxx'
test_dir=$dataset_path'/45/2fps/test'
train_dir=$dataset_path'/45/2fps/train';

model_dir_name='model_2024-10-05-02_49_11';
#
model_dir_name_epoch=$model_dir_name'_059';
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --batchsize 1024 --debug ;
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --batchsize 1024 --debug ;
# similarly you can  organize datapath for 30 subset
# ...
# test_dir=$dataset_path'/30/2fps/test'
