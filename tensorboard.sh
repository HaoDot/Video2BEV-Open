#!/usr/bin/expect
set host "10.7.18.249"
set tb_name "6022"
set one "1"
set pre ":127.0.0.1:"
set usrname "hao"
set timeout 5
#set $password

spawn ssh -L $one$tb_name$pre$tb_name $usrname@$host
expect "*password*" {send " \r"}
# interact
expect $usrname  {send "sh /mnt/share/hao/codes/others/university/itm/0509v2/run-tensorboard.sh\r"}
interact
# /media/omnisky/Data1/juhao/iccv22/codes/ape_sr/run-tensorboard.sh
# expect "*password*" {send "$password\r"}
# ssh -L 16006:127.0.0.1:6006 omnisky@10.7.167.84;
# sleep 3s;
# send "tmux new-session -d -s tb6 -n window0;\r"
# send "tmux send -t tb6:window0 "cd /media/omnisky/Data1/juhao/iccv22/codes/ape_sr/tblogger" ENTER;"
# send "tmux send -t tb6:window0 "/home/omnisky/anaconda3/envs/mmp/bin/tensorboard --logdir=EDSR" ENTER;"
# tmux new-session -d -s tb6 -n window0;
# tmux send -t tb6:window0 "cd /media/omnisky/Data1/juhao/iccv22/codes/ape_sr/tblogger" ENTER;
# tmux send -t tb6:window0 "/home/omnisky/anaconda3/envs/mmp/bin/tensorboard --logdir=EDSR" ENTER;

