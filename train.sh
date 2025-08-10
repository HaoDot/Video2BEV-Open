#!/bin/bash
cd path/to/repo/;

python_path='~/miniconda3/envs/lff/torch1131/bin/python'
gpu_ids='0';

backbone_name='vit-base'
first_stage_weight_path='path/to/first-stage-weight'
dataset_path='./UniV/45/2fps/train'
batchsize=140

$python_path train_bev_paired_fsra.py \
--backbone_name $backbone_name --first_stage_weight_path $first_stage_weight_path \
--lr 3e-4 --gpu_ids $gpu_ids --batchsize $batchsize \
--rendered_BEV --vit --vit_itc --vit_itm \
--lr_itm 2e-4 --vit_itm_share \
--two_stage_training --name two_view_long_share_d0.75_256_s1 \
--epoch 120 \
--neg_sample_number 3 \
--wd 0.5 \
--views 2  --droprate 0.75 --share \
--stride 1 --h 256  --w 256 --fp16 \
--sample_num 1 --optimizer 'AdamW' \
--num_warmup_steps 39 --sd_negative_sample \
--scheduler 'linearlr_warmup_34_epoch' --num_worker 16 --data_dir $dataset_path ;



