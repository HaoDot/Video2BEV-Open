cd /path/to/repo/;
python_path='/home/juhao/miniconda3/envs/torch1131/bin/python';

#gpu_ids='1';
gpu_ids='0';
loss_lambda=1.0
dataset_path='path/to/train'


# 20240920
$python_path train_bev_paired_fsra.py --data_dir $dataset_path \
--lr 2e-4 --lr_instance 2e-4 --lr_itc 2e-4 --gpu_ids $gpu_ids --batchsize 140 --num_worker 16 \
--epoch 140 \
--rendered_BEV --vit --lpn --vit_itc --loss_lambda $loss_lambda \
--name two_view_long_share_d0.75_256_s1 --views 2 \
--droprate 0.75  --share  --stride 1 --h 256  --w 256 --fp16 --sample_num 1 \
--optimizer 'AdamW' --scheduler 'steplr';
