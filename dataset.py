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


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


class Paired_University1652(Dataset):
	def __init__(self, opt, root_path, transforms,
				 data_path='/media/xgroup/data/xgroup/hao/codes/others/university_etc/bev_baseline/bev0219/mylist.pkl'):
		self.opt = opt
		self.transform_dict = transforms
		self.root_path = root_path
		# bev 有一类没生成上
		self.class_number = 700
		with open(data_path, 'rb') as f:
			self.file_list = pickle.load(f)
		self.fag = True

	# for sample_idx in tqdm(sorted(os.listdir(join(root_path,'street')))):
	# 	# if self.opt.rendered_BEV:
	# 	# 	sample_number = len(os.listdir(join(root_path,'bev',sample_idx)))
	# 	# else:
	# 	# 	sample_number = len(os.listdir(join(root_path, 'bev', sample_idx)))
	# 	# one sample only
	# 	satellite_sample_path = glob(join(root_path, 'satellite', sample_idx, '*.jpg'))[0]
	# 	street_sample_path = glob(join(root_path, 'street', sample_idx,'*.jpg'))[0]
	# 	if self.opt.rendered_BEV:
	# 		for bev_sample_idx, bev_sample_name in enumerate(sorted(os.listdir(join(root_path,'bev',sample_idx)))):
	# 			bev_sample_path = join(root_path,'bev',sample_idx, bev_sample_name)
	# 			for google_sample_name in sorted(os.listdir(join(root_path,'google',sample_idx))):
	# 				tmp ={}
	# 				google_sample_path = join(root_path,'google',sample_idx,google_sample_name)
	# 				tmp['satellite'] = satellite_sample_path
	# 				tmp['street'] = street_sample_path
	# 				# many samples
	# 				tmp['drone'] = glob(join(root_path, 'drone', sample_idx,'*.jpeg'))[bev_sample_idx]
	# 				tmp['google'] = google_sample_path
	# 				if self.opt.rendered_BEV:
	# 					tmp['bev'] = bev_sample_path
	# 				self.file_list.append(tmp)
	# 	else:
	# 		raise Exception(' have not implented yet')
	# self.fag = True
	def pil_loader(self,path):
		# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
		with open(path, 'rb') as f:
			img = Image.open(f)
			return img.convert('RGB')

	def __getitem__(self, index):
		# self.transform_dict
		sample_path_d = self.file_list[index]

		# sample = self.pil_loader(sample_path)
		if self.opt.rendered_BEV:
			bev_path = sample_path_d['bev']
		google_path = sample_path_d['google']
		drone_path = sample_path_d['drone']
		street_path = sample_path_d['street']
		satellite_path = sample_path_d['satellite']
		bev_img = self.pil_loader(bev_path)
		bev_sample = self.transform_dict['bev'](bev_img)
		google_img = self.pil_loader(google_path)
		google_sample = self.transform_dict['train'](google_img)
		drone_img = self.pil_loader(drone_path)
		drone_sample = self.transform_dict['train'](drone_img)
		street_img = self.pil_loader(street_path)
		street_sample = self.transform_dict['train'](street_img)
		satellite_img = self.pil_loader(satellite_path)
		satellite_sample = self.transform_dict['satellite'](satellite_img)

		class_label = sample_path_d['class_label']


		# dataloaders['satellite'], dataloaders['street'], dataloaders['drone'], dataloaders['google'],dataloaders['bev']
		return satellite_sample, street_sample, drone_sample,google_sample, bev_sample, class_label

	def __len__(self):
		return len(self.file_list)
