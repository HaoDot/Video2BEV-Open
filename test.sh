cd /media/xgroup/data/xgroup/hao/codes/others/university_etc/itm/rebuttal/itm_backbones/;
gpu_ids='2';
python_path='/home/xgroup/anaconda3/envs/torch1131/bin/python';
# 45
test_dir='/media/xgroup/data/xgroup/hao/datasets/university/drone_testset/test';
#test_dir='/media/xgroup/data/xgroup/hao/datasets/university/30_drone_testset/test';
train_dir='/media/xgroup/data/xgroup/hao/datasets/university/University-Release/train';

model_dir_name='model_2025-01-26-04:38:57';
#
#model_dir_name_epoch=$model_dir_name'_039';
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
###$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;
#model_dir_name_epoch=$model_dir_name'_059';
##topk_values=(8 16 32 64)
##for topk in "${topk_values[@]}"; do
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
##done
##$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_drone' --gallery_name 'gallery_satellite' --debug --first_stage_only True ;
##$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_drone' --debug --first_stage_only True ;
##$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;
#
#model_dir_name_epoch=$model_dir_name'_079';
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
###$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;
##model_dir_name_epoch=$model_dir_name'_099';
##$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_drone' --gallery_name 'gallery_satellite' --debug --first_stage_only True ;
##$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_drone' --debug --first_stage_only True ;
###$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;
##
##model_dir_name_epoch=$model_dir_name'_119';
##$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_drone' --gallery_name 'gallery_satellite' --debug --first_stage_only True ;
##$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_drone' --debug --first_stage_only True ;

model_dir_name='model_2025-01-26-05:51:43';
#
model_dir_name_epoch=$model_dir_name'_059';
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' ;
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
##$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;
model_dir_name_epoch=$model_dir_name'_119';
#topk_values=(8 16 32 64)
#for topk in "${topk_values[@]}"; do
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
#model_dir_name_epoch=$model_dir_name'_079';
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;

#model_dir_name='model_2025-01-26-06:19:14';
##
#model_dir_name_epoch=$model_dir_name'_059';
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
###$python_path test_bev_group_feat_fusion_two_stage_train.py --gpu_ids $gpu_ids --test_dir $train_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'satellite' --gallery_name 'bev' --debug ;
#model_dir_name_epoch=$model_dir_name'_119';
##topk_values=(8 16 32 64)
##for topk in "${topk_values[@]}"; do
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
#model_dir_name_epoch=$model_dir_name'_079';
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_bev' --gallery_name 'gallery_satellite' --debug ;
#$python_path test_bev_group_feat_fusion_two_stage.py --gpu_ids $gpu_ids --test_dir $test_dir --model_dir_name_epoch $model_dir_name_epoch --model_dir_name $model_dir_name --query_name 'query_satellite' --gallery_name 'gallery_bev' --debug ;
