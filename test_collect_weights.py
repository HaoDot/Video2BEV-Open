import os
from os.path import join
from glob import glob

src_path = '/18141169908/hao/itc/itc20250123/model_2025-01-23-04:20:59'
# epoch_l = [59, 79, 99, 139, 239, 339]
# epoch_l = [419, 479]
# epoch_l = [39, 59, 79, 99, 119, 139]
epoch_l = [39, 59, 79, 99, 119]
# epoch_l = [39]

exp_name = os.listdir(src_path)[0]
save_path_root = join(src_path, exp_name, os.path.basename(src_path), exp_name)
for epoch_number in epoch_l:
	cur_save_path = join(save_path_root, '{}_{:03d}'.format(os.path.basename(src_path),epoch_number))
	os.makedirs(cur_save_path)
	pth_path = join(src_path, exp_name, 'net_{:03d}.pth'.format(epoch_number))
	config_path = join(src_path, exp_name, 'opts.yaml')
	os.symlink(pth_path,join(cur_save_path, 'net_{:03d}.pth'.format(epoch_number)))
	os.symlink(config_path,join(cur_save_path, 'opts.yaml'))

