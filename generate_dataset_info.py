from glob import glob
import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from os.path import join
import random
from tqdm import tqdm
import pickle
import multiprocessing
from multiprocessing import Pool

def oneseq(params):
	sample_idx, sample_label   = params
	file_list = []
	# one sample only
	satellite_sample_path = glob(join(root_path, 'satellite', sample_idx, '*.jpg'))[0]
	street_sample_path = glob(join(root_path, 'street', sample_idx, '*.jpg'))[0]
	if rendered_BEV:
		for bev_sample_idx, bev_sample_name in enumerate(tqdm(sorted(os.listdir(join(root_path, 'bev', sample_idx))))):
			bev_sample_path = join(root_path, 'bev', sample_idx, bev_sample_name)
			for google_sample_name in sorted(os.listdir(join(root_path, 'google', sample_idx))):
				tmp = {}
				google_sample_path = join(root_path, 'google', sample_idx, google_sample_name)
				tmp['class_label'] = sample_label
				tmp['satellite'] = satellite_sample_path
				tmp['street'] = street_sample_path
				# many samples
				tmp['drone'] = glob(join(root_path, 'drone', sample_idx, '*.jpeg'))[bev_sample_idx]
				tmp['google'] = google_sample_path
				if rendered_BEV:
					tmp['bev'] = bev_sample_path
				file_list.append(tmp)

	else:
		raise Exception(' have not implented yet')
	return file_list

root_path = '/a_hao_working_space/datasets/university/University-Release/train'
rendered_BEV = True
save_path = '/a_hao_working_space/codes/others/university_etc/bev_baseline/bev0219'


# set(sorted(os.listdir(join(root_path, 'street')))) -set(sorted(os.listdir(join(root_path, 'bev'))))
# set(sorted(os.listdir(join(root_path, 'bev')))) - set(sorted(os.listdir(join(root_path, 'street'))))
street_l = os.listdir(join(root_path, 'street'))
street_l.pop(street_l.index('1440'))
file_list_l = []
result_list = []
# with multiprocessing.Pool(32) as pool:
# 	pool.map(file_list_l.append(oneseq),
# 			 [(sample_idx)
# 			  for sample_idx in sorted(street_l)])
pool = Pool(processes=32)
for sample_idx, sample_name in enumerate(street_l):
	# oneseq((sample_name,sample_idx))
	result_list.append(pool.apply_async(oneseq, ((sample_name,sample_idx),)))
pool.close()
pool.join()
for result in result_list:
	# k, v = result._value
	file_list_l.extend(result._value)

# for sample_idx in (sorted(os.listdir(join(root_path, 'street')))):
# 	# if self.opt.rendered_BEV:
# 	# 	sample_number = len(os.listdir(join(root_path,'bev',sample_idx)))
# 	# else:
# 	# 	sample_number = len(os.listdir(join(root_path, 'bev', sample_idx)))
# 	# one sample only
# 	satellite_sample_path = glob(join(root_path, 'satellite', sample_idx, '*.jpg'))[0]
# 	street_sample_path = glob(join(root_path, 'street', sample_idx, '*.jpg'))[0]
# 	if rendered_BEV:
# 		for bev_sample_idx, bev_sample_name in enumerate(sorted(os.listdir(join(root_path, 'bev', sample_idx)))):
# 			bev_sample_path = join(root_path, 'bev', sample_idx, bev_sample_name)
# 			for google_sample_name in sorted(os.listdir(join(root_path, 'google', sample_idx))):
# 				tmp = {}
# 				google_sample_path = join(root_path, 'google', sample_idx, google_sample_name)
# 				tmp['satellite'] = satellite_sample_path
# 				tmp['street'] = street_sample_path
# 				# many samples
# 				tmp['drone'] = glob(join(root_path, 'drone', sample_idx, '*.jpeg'))[bev_sample_idx]
# 				tmp['google'] = google_sample_path
# 				if rendered_BEV:
# 					tmp['bev'] = bev_sample_path
# 				file_list.append(tmp)
# 	else:
# 		raise Exception(' have not implented yet')


save_path = join(save_path,"mylist.pkl")
with open(save_path, 'wb') as f:
	pickle.dump(file_list_l, f)
