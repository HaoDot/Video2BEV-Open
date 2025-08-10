t_name=tb22
tmux new-session -d -s $t_name -n window0;
tmux send -t $t_name:window0 "cd /mnt/share/hao/codes/others/university/itm/0509v2/model_2024-05-11-10:04:29/two_view_long_share_d0.75_256_s1/" ENTER;
#tmux send -t $t_name:window0 "cd /media/xgroup/data/xgroup/hao/codes/others/university_etc/bev_baseline/vit_bkb/vit_0411_itc_same_head/model_2024-04-11-16:08:51/three_view_long_share_d0.75_256_s1_google/" ENTER;
tmux send -t $t_name:window0 "/home/hao/miniconda3/envs/pytorch190_timm/bin/tensorboard --logdir=tb_logger --port=6022" ENTER;
