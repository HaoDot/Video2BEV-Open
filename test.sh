cd /data/juhao/code/q1/Ours/first_stage/sd_geo_localization_contrastive/;
gpu_ids='1';
python_path='/home/juhao/miniconda3/envs/torch1131/bin/python';
test_dir='/data/juhao/dataset/UniV/45/2/test';
train_dir='path/to/train';


model_dir_name='model_2024-08-20-19_19_36';

model_dir_name_epoch=$model_dir_name'_059';
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug --batchsize 1024 ;
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug --batchsize 1024 ;
#$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;