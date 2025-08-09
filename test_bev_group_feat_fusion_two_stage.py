
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import h5py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from torch.nn import functional as F
import scipy.io
import yaml
import math
from model import ft_net, two_view_net, three_view_net
from utils import *
from evaluate_gpu_api_group_feat_fusion_two_stage import evaluate_root

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
def str_to_bool(value):
    """Convert string to boolean."""
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise ValueError(f'Invalid truth value {value}')
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='4', type=str, help='0,1,2,3...or last')
# parser.add_argument('--test_dir',default='/media/xgroup/data/xgroup/hao/datasets/university/University-Release/test',type=str, help='./test_data')
# 这个数据集是统一了drone和bev在gallery 数目，都是950个样本（将drone的0281丢掉了）
# parser.add_argument('--test_dir',default='/media/xgroup/data/xgroup/hao/datasets/university/University-Release/train',type=str, help='./test_data')
# parser.add_argument('--test_dir',default='/media/xgroup/data/xgroup/hao/datasets/university/drone_testset/test',type=str, help='./test_data')
# parser.add_argument('--test_dir',default='/data/hao/dataset/UniV/45/2/test',type=str, help='./test_data')
parser.add_argument('--test_dir',default='/data/hao/dataset/UniV_SUES_160k_mix/test',type=str, help='./test_data')
# parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
# parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--name', default='two_view_long_share_d0.75_256_s1', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=1024, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# parser.add_argument('--topk_two_stage',default=16, type=int,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# parser.add_argument('--model_dir_name',default=None, type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--model_dir_name',default='model_2024-10-05-02_49_11', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--model_dir_name_epoch',default='model_2024-10-05-02_49_11_059', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

parser.add_argument('--query_name',default='query_satellite', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# parser.add_argument('--query_name',default='query_bev', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# parser.add_argument('--query_name',default=None, type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--gallery_name',default='gallery_bev', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# parser.add_argument('--gallery_name',default='gallery_satellite', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# parser.add_argument('--gallery_name',default=None, type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--result_analysis', action='store_true',default=True, help='use fp16.' )
parser.add_argument('--debug', action='store_false',default=True, help='use fp16.' )
# parser.add_argument('--debug', action='store_false',default=False, help='use fp16.' )
parser.add_argument('--topk', type=int,default=32, help='use fp16.' )
parser.add_argument('--first_stage_only',type=str_to_bool, default=False, help='use first stage for evaluation')

opt = parser.parse_args()
###load config###
# load the training config
# config_path = os.path.join('./model_retrain',opt.name,'opts.yaml')
# model_dir_name = 'model_2024-01-29-23:31:20'
# bev
# model_dir_name = 'model_2024-04-11-16:08:51'
# drone
# model_dir_name = 'model_2024-05-24-19:50:25'
model_dir_name = opt.model_dir_name
model_dir_name_epoch = opt.model_dir_name_epoch
# model_dir_name = 'model_2024-05-03-00:04:33'
# model_dir_name = 'model_2024-04-29-22:01:24'
# while not os.path.exists(os.path.join('/mnt/share/hao/codes/others/university/itm/0513v2/model_2024-05-13-23:08:36/two_view_long_share_d0.75_256_s1/net_119.pth')):
#     print('waiting for the last epoch')
#     time.sleep(60*5)
#     continue
# config_path = os.path.join('./{}'.format(model_dir_name),opt.name,'opts.yaml')
config_path = os.path.join('./{}'.format(model_dir_name),opt.name, model_dir_name_epoch,'opts.yaml')
# config_path = os.path.join('./{}'.format(model_dir_name),opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
    # config = yaml.load(stream)
    config = yaml.safe_load(stream)
opt.fp16 = config['fp16']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 729

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone','gallery_bev', 'gallery_street', 'query_satellite', 'query_bev', 'query_drone', 'query_street']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16) for x in ['gallery_satellite', 'gallery_drone','gallery_bev','gallery_street', 'query_satellite', 'query_bev', 'query_drone', 'query_street']}

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
    #                   ['satellite', 'bev']}
    # dataloaders = {
    #     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16) for x in ['satellite', 'bev']}
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' or 'bev' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model, dataloaders, opt, view_index = 1, video_frame_d = None,name_l=None):
    
    out_dir = f'./tmp/{generate_random_string()}'

    # 创建文件夹
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # features = torch.FloatTensor()
    # features_full_seq = torch.FloatTensor()
    count = 0
    
    if video_frame_d==None:
        label_to_save = None
    else:
        label_to_save = 0
    feature_in_last_batch = None
    full_feature_in_last_batch = None
    
    for data_idx, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        
        if model.backbone_name == 'swin-base':
            ff = torch.FloatTensor(n, 1024).zero_().cuda()
        else:
            ff = torch.FloatTensor(n, 768).zero_().cuda()
        if opt.lpn:
            # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
            if opt.itc:
                if model.backbone_name == 'swin-base':
                    ff = torch.FloatTensor(n, 1024 * model.block).zero_().cuda()
                else:
                    ff = torch.FloatTensor(n, 768 * model.block).zero_().cuda()
            else:
                ff = torch.FloatTensor(n, 512, opt.block).zero_().cuda()
        ff_full_seq = torch.FloatTensor(n, 257, 768).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                
                # #  ++++++++++++++++++++++++++++++++++++++++(Effficiency)
                # mb = count_params_mb(model)                   # 参数 MB
                # gflops = calc_flops_gpu(model, input_img)        # 理论 GFLOPs
                # t_sec = measure_forward_time(model, input_img)    # 前向秒数
                
                # print(f'Model details: {mb} MB {gflops} GFLOPs {t_sec} s')
                # exit()
                # # ++++++++++++++++++++++++++++++++++++++++
                
                if opt.views ==2:
                    if view_index == 1:
                        # # TODO: debug only
                        # outputs_full_seq = torch.zeros([input_img.shape[0], 257, 768]).cuda()
                        # outputs = torch.zeros([input_img.shape[0], 3840]).cuda()
                        
                        outputs, _ = model(input_img, None, None)
                        if opt.itm:
                            outputs, outputs_full_seq = outputs[1], outputs[2]
                        else:
                            if opt.lpn:
                                if opt.itc:
                                    outputs = outputs[1]
                            else:
                                # 选norm 后的cls token
                                assert torch.equal(F.normalize(outputs[0], dim=-1), outputs[1])
                                outputs = outputs[1]
                        # outputs_full_seq = outputs[2]
                    elif view_index == 3:
                        # # TODO: debug only
                        # outputs_full_seq = torch.zeros([input_img.shape[0], 257, 768]).cuda()
                        # outputs = torch.zeros([input_img.shape[0], 3840]).cuda()
                        
                        
                        _, outputs = model(None, None, input_img)
                        if opt.itm:
                            outputs, outputs_full_seq = outputs[1], outputs[2]
                        else:
                            if opt.lpn:
                                if opt.itc:
                                    outputs = outputs[1]
                            else:
                                assert torch.equal(F.normalize(outputs[0], dim=-1), outputs[1])
                                outputs = outputs[1]
                    else:
                        raise Exception('Only support 1(satellite) and 3(drone(bev))')
                    # elif view_index ==2:
                    #     _, outputs = model(None, input_img)
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                        outputs = outputs[1]
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                        outputs = outputs[1]
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                        outputs = outputs[1]

                if opt.itm:
                    if not opt.lpn:
                        cls_feat = outputs_full_seq[:,0,:]
                        cls_feat = model.classifier(cls_feat)
                        if not torch.equal(cls_feat / cls_feat.norm(dim=-1, keepdim=True),outputs):
                        # if not torch.equal(F.normalize(outputs_full_seq[:,0,:], dim=-1),outputs):
                            raise Exception('cls token is not equal to the one in the full seq')
                ff += outputs
                if opt.itm:
                    # ff_full_seq += outputs_full_seq
                    if (i == 0):
                        ff_full_seq = outputs_full_seq
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            # # ff: [B,768]
            # TODO: norm
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            # ff_full_seq_norm = torch.norm(ff_full_seq, p=2, dim=2, keepdim=True)
            # ff_full_seq = ff_full_seq.div(ff_full_seq_norm.expand_as(ff_full_seq))
            # ff_full_seq = ff_full_seq / 2

        
        if video_frame_d==None:
            save_batch_features_ordered(ff, ff_full_seq, count, out_dir, name_l)
        else:
            feature_in_last_batch, full_feature_in_last_batch, feature_split_l, full_seq_split_l, label_to_save_end, to_save_dir_name_l = \
                split_feature(video_frame_d, label_to_save, ff, ff_full_seq, feature_in_last_batch, full_feature_in_last_batch)
            # if feature_in_last_batch != None:
            #     # 上个batch有遗留的，要保存的feature数目 = 当前存到第几个（label_to_save_end）-进去前存到第几个（label_to_save）
            #     assert len(feature_split_l) == (label_to_save_end - label_to_save)
            # to_save_name_l = sorted(video_frame_d.keys())[label_to_save:label_to_save_end]
            # print(to_save_dir_name_l)
            # 从label_to_save开始保存到len(feature_split_l)
            # save_batch_features_ff(ff_l: list, out_dir: str, to_save_name_l: list, name:str)
            save_batch_features_ff(feature_split_l, out_dir, to_save_dir_name_l, name='ff')
            save_batch_features_ff(full_seq_split_l, out_dir, to_save_dir_name_l, name='ff_full_seq')
            
            
            label_to_save = label_to_save_end
            
        count += n
        print(count)
        # del ff, ff_full_seq
        # gc.collect()
                
        # features = torch.cat((features,ff.data.cpu()), 0)
        # if opt.itm:
        #     features_full_seq = torch.cat((features_full_seq, ff_full_seq.data.cpu()), 0)
    # features, features_full_seq = load_all_features(out_dir, device='cpu')
    # return features, features_full_seq
    return out_dir

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch, mssg = load_network(opt.name, opt,model_dir_name, model_dir_name_epoch)
# model.classifier.classifier = nn.Sequential()
# print('!!! keep the classifier ')
# model.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

topk_two_stage = int(opt.topk)
# topk_two_stage = 128

query_name = '{}'.format(opt.query_name)
gallery_name = '{}'.format(opt.gallery_name)

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))

gallery_path = image_datasets[gallery_name].imgs
# txt_path_ = './model/%s/gallery_name.txt'%opt.name
# config_path = os.path.join('./{}'.format(model_dir_name),opt.name,'opts.yaml')
txt_path_ = './{}'.format(model_dir_name)+'/%s/gallery_name.txt'%opt.name
f = open(txt_path_,'w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
# txt_path_ = './model/%s/query_name.txt'%opt.name
txt_path_ = './{}/'.format(model_dir_name)+'%s/query_name.txt'%opt.name
f = open(txt_path_,'w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    if opt.debug:
        print('==============================================')
        print('debug mode')
        print('==============================================')
    else:
        with torch.no_grad():
            # query_feature, query_feature_full_seq = extract_feature(model,dataloaders[query_name],_, which_query)
            if 'bev' not in query_name:
                name_l = []
                for q_path_id in query_path:
                    name_l.append(os.path.basename(os.path.dirname(q_path_id)))
                out_dir_query = extract_feature(model,dataloaders[query_name],_, which_query, video_frame_d=None, name_l= name_l)
            else:
                video_frame_d = collect_video_number(query_path)
                out_dir_query = extract_feature(model,dataloaders[query_name],_, which_query, video_frame_d=video_frame_d)
            # [N,512] 701
            # 37855
            # gallery_feature, gallery_feature_full_seq = extract_feature(model,dataloaders[gallery_name],_, which_gallery)
            if 'bev' not in gallery_name:
                name_l = []
                for g_path_id in gallery_path:
                    name_l.append(os.path.basename(os.path.dirname(g_path_id)))
                out_dir_gallery = extract_feature(model,dataloaders[gallery_name],_, which_gallery, video_frame_d=None, name_l= name_l)
            else:
                video_frame_d = collect_video_number(gallery_path)
                out_dir_gallery = extract_feature(model,dataloaders[gallery_name],_, which_gallery, video_frame_d=video_frame_d)
            # [N,512] 51355
            # [N,512] 950

        # For street-view image, we use the avg feature as the final feature.
        '''
        if which_query == 2:
            new_query_label = np.unique(query_label)
            new_query_feature = torch.FloatTensor(len(new_query_label) ,512).zero_()
            for i, query_index in enumerate(new_query_label):
                new_query_feature[i,:] = torch.sum(query_feature[query_label == query_index, :], dim=0)
            query_feature = new_query_feature
            fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
            query_feature = query_feature.div(fnorm.expand_as(query_feature))
            query_label   = new_query_label
        elif which_gallery == 2:
            new_gallery_label = np.unique(gallery_label)
            new_gallery_feature = torch.FloatTensor(len(new_gallery_label), 512).zero_()
            for i, gallery_index in enumerate(new_gallery_label):
                new_gallery_feature[i,:] = torch.sum(gallery_feature[gallery_label == gallery_index, :], dim=0)
            gallery_feature = new_gallery_feature
            fnorm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
            gallery_feature = gallery_feature.div(fnorm.expand_as(gallery_feature))
            gallery_label   = new_gallery_label
        '''

        # Save to Matlab for check
        # result = {'gallery_f': gallery_feature.numpy(), 'query_f': query_feature.numpy(),
        #           'gallery_label': gallery_label, 'gallery_path': gallery_path,
        #           'query_label': query_label, 'query_path': query_path}
        result = {'gallery_f_path': out_dir_gallery, 'query_f_path': out_dir_query,
                  'gallery_label': gallery_label, 'gallery_path': gallery_path,
                  'query_label': query_label, 'query_path': query_path}

        scipy.io.savemat('./pytorch_result.mat',result)

        # if not _.first_stage_only:
        #     if _.itm:
        #         with h5py.File('./pytorch_result.h5', 'w') as h5file:
        #             h5file.create_dataset('gallery_f_full_seq', data=np.array(gallery_feature_full_seq))
        #             h5file.create_dataset('query_f_full_seq', data=np.array(query_feature_full_seq))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # exit()
    print(opt.name)
    # result = './model/%s/result.txt'%opt.name
    # result = './{}'.format(model_dir_name)+'/%s/result.txt'%opt.name
    result = './{}'.format(model_dir_name) + '/{}/{}/result.txt'.format(opt.name, opt.model_dir_name_epoch)
    with open(result, 'a+') as f:
        f.writelines([mssg])
        f.write('\n')
        f.write('choose top {} in first stage for the second stage'.format(topk_two_stage))
    # result = './model_retrain/%s/result.txt'%opt.name
    # os.system('python evaluate_gpu.py | tee -a %s'%result)
    current_dir = os.getcwd()  # 获取当前工作目录
    absolute_path = os.path.abspath(current_dir)  # 将相对路径转换为绝对路径
    # exit()
    torch.cuda.empty_cache()
    evaluate_root(query_name,query_path,gallery_name,gallery_path,os.path.join(absolute_path,'{}'.format(model_dir_name),'{}'.format(opt.name), '{}'.format(opt.model_dir_name_epoch))
                  ,topk_two_stage=topk_two_stage,model=model,opt=_)


