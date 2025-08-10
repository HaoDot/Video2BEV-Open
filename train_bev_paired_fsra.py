# -*- coding: utf-8 -*-
from __future__ import print_function, division
# import sys
# sys.path.append('/a_hao_working_space/codes/others/university_etc/bev_baseline/bev0219/')
import argparse
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from folder import ImageFolder
import torch.backends.cudnn as cudnn
import matplotlib
import torchvision
matplotlib.use('agg')
import time
import matplotlib.pyplot as plt
# from PIL import Image
import copy
import time
import os
from model import two_view_net, three_view_net
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
import yaml
import torch.nn.functional as F
from shutil import copyfile
import random
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning
from circle_loss import CircleLoss, convert_label_to_similarity
from dataset import Paired_University1652
from tool.make_dataloader import make_dataset

version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='output model name')#'two_view'
parser.add_argument('--resume', action='store_true', help='use resume trainning')
#data
parser.add_argument('--data_dir', default='/university1652k/University-Release/train', type=str, help='training dir path')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')

parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
# parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--sample_num', default=1, type=int, help='num of repeat sampling' )
# parser.add_argument('--sample_num', default=2, type=int, help='num of repeat sampling' )
parser.add_argument('--num_worker', default=8, type=int, help='num of worker' )
#backbone
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--epoch', default=120, type=int, help='stride')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')

# parser.add_argument('--rendered_BEV', action='store_true',default=False, help='using extra noise Google')#,default=False
parser.add_argument('--rendered_BEV', action='store_true',default=True, help='using extra noise Google')#,default=False

# parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--share', action='store_true', default='True',help='share weight between different view')

parser.add_argument('--ibn', action='store_true',default=False, help='use NAS')
# parser.add_argument('--ibn', action='store_true',default=True, help='use NAS')

parser.add_argument('--vit', action='store_true',default=True, help='use NAS')
# parser.add_argument('--vit', action='store_true',default=False, help='use NAS')
#
parser.add_argument('--vit_itc', action='store_true',default=True, help='use NAS')
# parser.add_argument('--vit_itc', action='store_true',default=False, help='use NAS')

# parser.add_argument('--vit_itm', action='store_true',default=True, help='use NAS')
parser.add_argument('--vit_itm', action='store_true',default=False, help='use NAS')

parser.add_argument('--vit_itm_share', action='store_true',default=True, help='use NAS')
# parser.add_argument('--vit_itm_share', action='store_true',default=False, help='use NAS')

parser.add_argument('--lpn', action='store_true',default=True, help='use NAS')
# parser.add_argument('--lpn', action='store_true',default=False, help='use NAS')

#optimizer
parser.add_argument('--optimizer', default='SGD', help='use all training data')
# parser.add_argument('--optimizer', action='store_true',default='AdamW', help='use all training data')
parser.add_argument('--scheduler', default='steplr', help='use all training data')
# parser.add_argument('--scheduler', action='store_true',default='cosine', help='use all training data')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
# parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
parser.add_argument('--lr', default=0.04, type=float, help='learning rate')
parser.add_argument('--lr_itm', default=0.004, type=float, help='learning rate')
parser.add_argument('--lr_instance', default=0.04, type=float, help='learning rate')
parser.add_argument('--lr_itc', default=0.04, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--fp16', action='store_true',
                    help='use float16 instead of float32, which will save about 50% memory')
# extra losses (default is cross-entropy loss. You can fuse different losses for further performance boost.)
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
parser.add_argument('--cosface', action='store_true', help='use CosFace loss')
parser.add_argument('--contrast', action='store_true', help='use contrast loss')
parser.add_argument('--triplet', action='store_true', help='use triplet loss')
parser.add_argument('--lifted', action='store_true', help='use lifted loss')
parser.add_argument('--sphere', action='store_true', help='use sphere loss')
parser.add_argument('--loss_merge', action='store_true', help='combine perspectives to calculate losses')
# lambda for loss balance
parser.add_argument('--loss_lambda', default=1.0, type=float, help='loss balance')
# parser.add_argument('--auxiliary', type=bool, required=False,default=False,help='losses between bev and satellite')#, default=False
parser.add_argument('--auxiliary', action='store_true',default=False,help='losses between bev and satellite')#, default=False
opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

def write_txt(record_file_path,content):
    with open(record_file_path, "a") as file:
        file.write("{}\n".format(content))
fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    print('gpu id:{}'.format(gpu_ids[0]))
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

# transform_train_list = [
#     # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
#     transforms.Resize((opt.h, opt.w), interpolation=3),
#     transforms.Pad(opt.pad, padding_mode='edge'),
#     transforms.RandomCrop((opt.h, opt.w)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]
#
# transform_satellite_list = [
#     transforms.Resize((opt.h, opt.w), interpolation=3),
#     transforms.Pad(opt.pad, padding_mode='edge'),
#     transforms.RandomAffine(360),
#     # transforms.RandomAffine(90),
#     transforms.RandomCrop((opt.h, opt.w)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]
#
# transform_bev_list = [
#     transforms.Resize((opt.h, opt.w), interpolation=3),
#     transforms.Pad(opt.pad, padding_mode='edge'),
#     transforms.RandomAffine(360),
#     # transforms.CenterCrop((3*(opt.h//4), 3*(opt.w//4))),
#     # transforms.Pad((opt.h-3*(opt.h//4))//2, fill=(0, 0, 0), padding_mode='constant'),
#     # transforms.CenterCrop((2*(opt.h//4), 2*(opt.w//4))),
#     # transforms.Pad((opt.h-2*(opt.h//4))//2, fill=(0, 0, 0), padding_mode='constant'),
#     # transforms.RandomAffine(360),
#     transforms.RandomCrop((opt.h, opt.w)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]
#
# transform_val_list = [
#     transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]
#
# if opt.erasing_p > 0:
#     transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
#
# if opt.color_jitter:
#     transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
#                                                    hue=0)] + transform_train_list
#     transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
#                                                        hue=0)] + transform_satellite_list
#     transform_bev_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
#                                                        hue=0)] + transform_bev_list
#
# if opt.DA:
#     transform_train_list = [ImageNetPolicy()] + transform_train_list
#
# print(transform_train_list)
# data_transforms = {
#     'train': transforms.Compose(transform_train_list),
#     'val': transforms.Compose(transform_val_list),
#     'satellite': transforms.Compose(transform_satellite_list)}
# if opt.rendered_BEV:
#     data_transforms.update({'bev':transforms.Compose(transform_bev_list)})
#
# train_all = ''
# if opt.train_all:
#     train_all = '_all'

# paired_dataset = Paired_University1652(opt,data_dir,data_transforms)
# dataloader = torch.utils.data.DataLoader(paired_dataset, batch_size=opt.batchsize,
#                                               shuffle=True, num_workers=2, pin_memory=True)  # 8 workers may work faster
# dataset_size = len(paired_dataset)
# class_names = paired_dataset.class_number
dataloader,class_names,dataset_size, drone_based_dataloader = make_dataset(opt)
opt.nclasses = len(class_names)
print(dataset_size)
# currentTime = time.localtime()
# model_name = 'model_{}'.format(time.strftime("%Y-%m-%d-%H:%M:%S", currentTime))
# current_file = __file__
# absolute_path = os.path.dirname(os.path.abspath(current_file))
# dir_name = os.path.join(absolute_path,model_name, name)
# txt_path = os.path.join(dir_name,'training_record.txt')
# write_txt(txt_path,dataset_size)

# image_datasets = {}
# image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'),
#                                                    data_transforms['satellite'])
# if opt.rendered_BEV:
#     image_datasets['bev'] = datasets.ImageFolder(os.path.join(data_dir, 'bev'),
#                                                        data_transforms['bev'])
# image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'),
#                                                 data_transforms['train'])
# image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'),
#                                                data_transforms['train'])
# image_datasets['google'] = ImageFolder(os.path.join(data_dir, 'google'),
#                                        # google contain empty subfolder, so we overwrite the Folder
#                                        data_transforms['train'])
#
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                               shuffle=True, num_workers=2, pin_memory=True)  # 8 workers may work faster
#                for x in ['satellite', 'street', 'drone', 'google']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'street', 'drone', 'google']}
# if opt.rendered_BEV:
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                                   shuffle=True, num_workers=2, pin_memory=True)
#                    # 8 workers may work faster
#                    for x in ['satellite', 'bev','street', 'drone', 'google']}
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'bev','street', 'drone', 'google']}
#
# # class_names = image_datasets['street'].classes
# # print(dataset_sizes)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    import os
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger

def tb_logger_add_images(tb_logger,tensors,current_iter):
    mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tmp_l = []
    for tensor_ in tensors:
        # tmp = 255*(tensor_*torch.from_numpy(np.array(std)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1))+torch.from_numpy(np.array(mean)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1)
        tmp = (255*(tensor_.mul_(torch.from_numpy(np.array(std)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1)).add_(torch.from_numpy(np.array(mean)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1)))).type(torch.uint8)
        tmp_l.append(tmp)
    # pass
    img = torchvision.utils.make_grid(torch.cat(tmp_l,dim=0),nrow=tensor_.shape[0],padding=2,pad_value=255)
    tb_logger.add_image('input_images',img,current_iter)
def get_contrastive_loss(modality_one, modality_two, temper_parameter):
    """
    computing contrastive loss. note: modality_one, modality_two should be paired data
    Args:
        modality_one: [B, C]
        modality_two: [B, C]
        temper_parameter: [B, B]
    Returns:

    """
    assert modality_one.size(-1) == modality_two.size(-1)

    itc_label = torch.arange(len(modality_one), device=modality_one.device)
    logits_one_two = modality_one @ modality_two.t() / temper_parameter
    # logits_one_two = modality_one @ modality_two.t()
    logits_two_one = modality_two @ modality_one.t() / temper_parameter
    # logits_two_one = modality_two @ modality_one.t()
    loss_i2t = F.cross_entropy(logits_one_two, itc_label)
    loss_t2i = F.cross_entropy(logits_two_one, itc_label)
    return (loss_i2t + loss_t2i) / 2

def compose_pairs_for_contrastive_loss(source_l):
    """
    compose pairs (sample_0, sample_1) from source_l
    Args:
        source_l: [ground, satellite, drone or bev, google]

    Returns:
        [
            [ground, satellite],
            [ground, drone or bev],
            [ground, google],
            [satellite, drone or bev],
            [satellite, google]
            [drone or bev, google]
        ]

    """
    pair_l = []
    for idx in range(len(source_l)):
        anchor_sample = source_l[idx]
        for idx_ in range((idx + 1), len(source_l), 1):
            # print('{}_{}'.format(idx,idx_))
            pair_l.append([anchor_sample, source_l[idx_]])
            # if idx_ == (len(source_l)-1):
            #     continue
            # else:
            #     pair_l.extend([anchor_sample, source_l[idx_]])
    return pair_l



def get_contrastive_loss_different_sources(feat_groups, temper_parameter, lpn=False):
    """

    Args:
        feat_groups:[satellite_based_samples, drone_based_samples]
            satellite_based_samples:[ground, satellite, drone or bev, google]
        temper_parameter: learnable parameter for similarity

    Returns:

    """
    satellite_based_sample_l = feat_groups[0]
    drone_based_sample_l = feat_groups[1]

    # satellite_based_paired_l = compose_pairs_for_contrastive_loss(satellite_based_sample_l)
    satellite_based_paired_l = [satellite_based_sample_l]
    # drone_based_paired_l = compose_pairs_for_contrastive_loss(drone_based_sample_l)
    drone_based_paired_l = [drone_based_sample_l]
    itc_loss = 0.
    for modality_one, modality_two in satellite_based_paired_l:
        itc_loss += get_contrastive_loss(modality_one, modality_two, temper_parameter)
    # itc_loss /= len(satellite_based_paired_l)

    for modality_one, modality_two in drone_based_paired_l:
        itc_loss += get_contrastive_loss(modality_one, modality_two, temper_parameter)
    itc_loss /= (len(drone_based_paired_l)+len(satellite_based_paired_l))
    # itc_loss /= 2
    return itc_loss

def compose_neg_pair_from_similarity(similarity, full_seq_l_other_source):
    """

    Args:
        similarity: 相似性
        full_seq_l_other_source: 另一个领域的feature

    Returns:

    """
    neg_pair_l = []
    bs = similarity.shape[0]
    for b in range(bs):
        # 从多项式分布抽取样本
        neg_idx = torch.multinomial(similarity[b], 1).item()
        neg_pair_l.append(full_seq_l_other_source[neg_idx])
    other_source_neg = torch.stack(neg_pair_l, dim=0)
    return other_source_neg


def has_duplicates(tensor):
    return len(set(tensor.numpy())) < len(tensor)

def one_LPN_output(outputs, labels, criterion, block):
    """
    For feature from the last stage in ResNet, operate square-ring partition
    Args:
        outputs:
        labels:
        criterion:
        block:

    Returns:

    """
    # part = {}
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss


def train_model(model, model_test, criterion, optimizer, scheduler, record_file_path,tb_logger, model_name,num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    # warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
    warm_iteration = round(dataset_size['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.cosface:
        criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32)  # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
    if opt.contrast:
        criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=512, margin=4)
    if opt.auxiliary:
        criterion_auxiliary = nn.MSELoss()

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        write_txt(record_file_path,'Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        write_txt(record_file_path, ('-' * 10))

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_itm = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0
            # Iterate over data.
            # for iters, (data, data2, data3, data4, data5) in enumerate(zip(dataloaders['satellite'], dataloaders['street'], dataloaders['drone'],
            #                                      dataloaders['google'],dataloaders['bev'])):
            for iters, ((data, data2, data3, data4),(d_data, d_data2, d_data3, d_data4)) in enumerate(zip(dataloader, drone_based_dataloader)):
                if opt.rendered_BEV:
                    (data4, data5) = data4
                    (d_data4, d_data5) = d_data4
                cur_iter = (epoch*len(dataloader)+iters)
                # get the inputs
                # labels,labels2,labels3,labels4,labels5 = [label_all]*5
                # satellite
                inputs, labels = data
                d_inputs, d_labels = d_data
                # street
                inputs2, labels2 = data2
                d_inputs2, d_labels2 = d_data2
                # drone
                inputs3, labels3 = data3
                d_inputs3, d_labels3 = d_data3
                # google
                inputs4, labels4 = data4
                d_inputs4, d_labels4 = d_data4
                if opt.rendered_BEV:
                    # bev
                    inputs5, labels5 = data5
                    d_inputs5, d_labels5 = d_data5
                if cur_iter%500==0:
                    print('add images')
                    if opt.rendered_BEV:
                        tb_logger_add_images(tb_logger, [inputs,inputs2,inputs3,inputs4,inputs5],cur_iter)
                    else:
                        tb_logger_add_images(tb_logger, [inputs, inputs2, inputs3, inputs4], cur_iter)
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    # satellite-anchored sampling
                    inputs = Variable(inputs.cuda().detach())
                    inputs2 = Variable(inputs2.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels2 = Variable(labels2.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                    # drone-anchored sampling
                    d_inputs = Variable(d_inputs.cuda().detach())
                    d_inputs2 = Variable(d_inputs2.cuda().detach())
                    d_inputs3 = Variable(d_inputs3.cuda().detach())
                    d_labels = Variable(d_labels.cuda().detach())
                    d_labels2 = Variable(d_labels2.cuda().detach())
                    d_labels3 = Variable(d_labels3.cuda().detach())
                    if opt.extra_Google:
                        inputs4 = Variable(inputs4.cuda().detach())
                        labels4 = Variable(labels4.cuda().detach())

                        d_inputs4 = Variable(d_inputs4.cuda().detach())
                        d_labels4 = Variable(d_labels4.cuda().detach())
                    if opt.rendered_BEV:
                        inputs5 = Variable(inputs5.cuda().detach())
                        labels5 = Variable(labels5.cuda().detach())
                        d_inputs5 = Variable(d_inputs5.cuda().detach())
                        d_labels5 = Variable(d_labels5.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, outputs2 = model(inputs, inputs2)
                else:
                    if opt.views == 2:
                        # outputs, outputs2 = model(inputs, inputs2)
                        if opt.rendered_BEV:
                            # replace inputs3(drone) with inputs5(BEV)
                            outputs, outputs3 = model(inputs, None, inputs5)
                            d_outputs, d_outputs3 = model(d_inputs, None, d_inputs5)
                        else:
                            outputs, outputs3 = model(inputs, None, inputs3)
                            d_outputs, d_outputs3 = model(d_inputs, None, d_inputs3)
                    elif opt.views == 3:
                        if opt.extra_Google:
                            if opt.rendered_BEV:
                                # replace inputs3(drone) with inputs5(BEV)
                                # satellite, street, bev, google
                                outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs5, inputs4)
                                d_outputs, d_outputs2, d_outputs3, d_outputs4 = model(d_inputs, d_inputs2, d_inputs5, d_inputs4)
                            else:
                                outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs3, inputs4)
                                d_outputs, d_outputs2, d_outputs3, d_outputs4 = model(d_inputs, d_inputs2, d_inputs3, d_inputs4)
                        else:
                            if opt.rendered_BEV:
                                # replace inputs3(drone) with inputs5(BEV)
                                outputs, outputs2, outputs3 = model(inputs, inputs2, inputs5)
                                d_outputs, d_outputs2, d_outputs3 = model(d_inputs, d_inputs2, d_inputs5)
                            else:
                                outputs, outputs2, outputs3 = model(inputs, inputs2, inputs3)
                                d_outputs, d_outputs2, d_outputs3 = model(d_inputs, d_inputs2, d_inputs3)

                return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere or opt.auxiliary#+ 0.5 * criterion_auxiliary()

                if opt.views == 2:
                    # _, preds = torch.max(outputs.data, 1)
                    # _, preds2 = torch.max(outputs2.data, 1)
                    # loss = criterion(outputs, labels) + criterion(outputs2, labels2)
                    if model.itc:
                        [outputs, cls_feat], [outputs3, cls_feat3] = outputs, outputs3
                        [d_outputs, d_cls_feat], [d_outputs3, d_cls_feat3] = d_outputs, d_outputs3
                        satellite_based_feat_group = [cls_feat, cls_feat3]
                        drone_based_feat_group = [d_cls_feat, d_cls_feat3]

                        if not opt.lpn:
                            itc_loss = get_contrastive_loss_different_sources(
                                [satellite_based_feat_group, drone_based_feat_group], model.logit_scale, opt.lpn)
                        else:

                            itc_loss = get_contrastive_loss_different_sources(
                                [satellite_based_feat_group, drone_based_feat_group], model.logit_scale, opt.lpn)


                    if not opt.lpn:
                        _, preds = torch.max(outputs.data, 1)
                        # _, preds2 = torch.max(outputs2.data, 1)
                        _, preds3 = torch.max(outputs3.data, 1)

                        _, d_preds = torch.max(d_outputs.data, 1)
                        # _, d_preds2 = torch.max(d_outputs2.data, 1)
                        _, d_preds3 = torch.max(d_outputs3.data, 1)
                        if opt.rendered_BEV:
                            # replace labels3(drone) with labels5(BEV)
                            loss = criterion(outputs, labels) + criterion(outputs3, labels5)
                            # write_txt(record_file_path,'satellite-based: satellite_label:{}'.format(labels.cpu().data))
                            # write_txt(record_file_path,'satellite-based: bev_label:{}'.format(labels5.cpu().data))
                            if has_duplicates(labels.cpu()):
                                raise Exception('satellite-based: satellite_label has duplicated label')
                            if has_duplicates(labels5.cpu()):
                                raise Exception('satellite-based: bev_label has duplicated label')
                            d_loss = criterion(d_outputs, d_labels) + criterion(d_outputs3, d_labels5)
                            if has_duplicates(d_labels.cpu()):
                                raise Exception('bev-based: satellite_label has duplicated label')
                            if has_duplicates(d_labels5.cpu()):
                                raise Exception('bev-based: bev_label has duplicated label')
                            # write_txt(record_file_path, 'bev-based: satellite_label:{}'.format(d_labels.cpu().data))
                            # write_txt(record_file_path, 'bev-based: bev_label:{}'.format(d_labels5.cpu().data))
                        else:
                            loss = criterion(outputs, labels) + criterion(outputs3, labels3)
                            d_loss = criterion(d_outputs, d_labels) + criterion(d_outputs3, d_labels3)
                    else:
                        preds, loss = one_LPN_output(outputs, labels, criterion, 5)
                        d_preds, d_loss = one_LPN_output(d_outputs, d_labels, criterion, 5)
                        if opt.rendered_BEV:
                            preds3, loss3 = one_LPN_output(outputs3, labels5, criterion, 5)
                            d_preds3, d_loss3 = one_LPN_output(d_outputs3, d_labels5, criterion, 5)

                        if opt.views == 2:  # no implement this LPN model
                            loss = loss + loss3
                            d_loss = d_loss + d_loss3
                elif opt.views == 3:
                    if return_feature:
                        logits, ff = outputs
                        logits2, ff2 = outputs2
                        logits3, ff3 = outputs3
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        fnorm2 = torch.norm(ff2, p=2, dim=1, keepdim=True)
                        fnorm3 = torch.norm(ff3, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))  # 8*512,tensor
                        ff2 = ff2.div(fnorm2.expand_as(ff2))
                        ff3 = ff3.div(fnorm3.expand_as(ff3))
                        if opt.rendered_BEV:
                            # replace labels3(drone) with labels5(BEV)
                            loss = criterion(logits, labels) + criterion(logits2, labels2) + criterion(logits3, labels5) + 0.5 * criterion_auxiliary(ff3,ff)
                        else:
                            loss = criterion(logits, labels) + criterion(logits2, labels2) + criterion(logits3, labels3)

                        _, preds = torch.max(logits.data, 1)
                        _, preds2 = torch.max(logits2.data, 1)
                        _, preds3 = torch.max(logits3.data, 1)
                        # Multiple perspectives are combined to calculate losses, please join ''--loss_merge'' in run.sh
                        if opt.loss_merge:
                            ff_all = torch.cat((ff, ff2, ff3), dim=0)
                            if opt.rendered_BEV:
                                # replace labels3(drone) with labels5(BEV)
                                labels_all = torch.cat((labels, labels2, labels5), dim=0)
                            else:
                                labels_all = torch.cat((labels, labels2, labels3), dim=0)
                        if opt.extra_Google:
                            logits4, ff4 = outputs4
                            fnorm4 = torch.norm(ff4, p=2, dim=1, keepdim=True)
                            ff4 = ff4.div(fnorm4.expand_as(ff4))
                            if opt.rendered_BEV:
                                # replace labels3(drone) with labels5(BEV)
                                loss = criterion(logits, labels) + criterion(logits2, labels2) \
                                       + criterion(logits3,labels5) + criterion(logits4, labels4)
                            else:
                                loss = criterion(logits, labels) + criterion(logits2, labels2) + criterion(logits3, labels3) +criterion(logits4, labels4)
                            if opt.loss_merge:
                                ff_all = torch.cat((ff_all, ff4), dim=0)
                                labels_all = torch.cat((labels_all, labels4), dim=0)
                        if opt.arcface:
                            if opt.loss_merge:
                                loss += criterion_arcface(ff_all, labels_all)
                            else:
                                if opt.rendered_BEV:
                                # replace labels3(drone) with labels5(BEV)
                                    loss += criterion_arcface(ff, labels) + criterion_arcface(ff2, labels2) \
                                            + criterion_arcface(ff3, labels5)  # /now_batch_size
                                else:
                                    loss += criterion_arcface(ff, labels) + criterion_arcface(ff2, labels2) + criterion_arcface(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_arcface(ff4, labels4)  # /now_batch_size
                        if opt.cosface:
                            if opt.loss_merge:
                                loss += criterion_cosface(ff_all, labels_all)
                            else:
                                if opt.rendered_BEV:
                                    # replace labels3(drone) with labels5(BEV)
                                    loss += criterion_cosface(ff, labels) + criterion_cosface(ff2,labels2) \
                                            + criterion_cosface(ff3, labels5)  # /now_batch_size
                                else:
                                    loss += criterion_cosface(ff, labels) + criterion_cosface(ff2, labels2) + criterion_cosface(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_cosface(ff4, labels4)  # /now_batch_size
                        if opt.circle:
                            if opt.loss_merge:
                                loss += criterion_circle(*convert_label_to_similarity(ff_all, labels_all)) / now_batch_size
                            else:
                                if opt.rendered_BEV:
                                    # replace labels3(drone) with labels5(BEV)
                                    loss += criterion_circle(
                                        *convert_label_to_similarity(ff, labels)) / now_batch_size + criterion_circle(
                                        *convert_label_to_similarity(ff2, labels2)) / now_batch_size + criterion_circle(
                                        *convert_label_to_similarity(ff3, labels5)) / now_batch_size
                                else:
                                    loss += criterion_circle(*convert_label_to_similarity(ff, labels)) / now_batch_size + criterion_circle(*convert_label_to_similarity(ff2, labels2)) / now_batch_size + criterion_circle(*convert_label_to_similarity(ff3, labels3)) / now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_circle(*convert_label_to_similarity(ff4, labels4)) / now_batch_size
                        if opt.triplet:
                            if opt.loss_merge:
                                hard_pairs_all = miner(ff_all, labels_all)
                                loss += criterion_triplet(ff_all, labels_all, hard_pairs_all)
                            else:
                                hard_pairs = miner(ff, labels)
                                hard_pairs2 = miner(ff2, labels2)
                                if opt.rendered_BEV:
                                    # replace labels3(drone) with labels5(BEV)
                                    hard_pairs3 = miner(ff3, labels5)
                                    loss += criterion_triplet(ff, labels, hard_pairs) + criterion_triplet(ff2, labels2,hard_pairs2) \
                                            + criterion_triplet(ff3, labels5, hard_pairs3)  # /now_batch_size
                                else:
                                    hard_pairs3 = miner(ff3, labels3)
                                    loss += criterion_triplet(ff, labels, hard_pairs) + criterion_triplet(ff2, labels2, hard_pairs2) + criterion_triplet(ff3, labels3, hard_pairs3)# /now_batch_size
                                if opt.extra_Google:
                                    hard_pairs4 = miner(ff4, labels4)
                                    loss += criterion_triplet(ff4, labels4, hard_pairs4)
                        if opt.lifted:
                            if opt.loss_merge:
                                loss += criterion_lifted(ff_all, labels_all)
                            else:
                                if opt.rendered_BEV:
                                    # replace labels3(drone) with labels5(BEV)
                                    loss += criterion_lifted(ff, labels) + criterion_lifted(ff2,labels2) \
                                            + criterion_lifted(ff3, labels5)  # /now_batch_size
                                else:
                                    loss += criterion_lifted(ff, labels) + criterion_lifted(ff2, labels2) + criterion_lifted(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_lifted(ff4, labels4)
                        if opt.contrast:
                            if opt.loss_merge:
                                loss += criterion_contrast(ff_all, labels_all)
                            else:
                                if opt.rendered_BEV:
                                    # replace labels3(drone) with labels5(BEV)
                                    loss += criterion_contrast(ff, labels) + criterion_contrast(ff2,labels2) \
                                            + criterion_contrast(ff3, labels5)  # /now_batch_size
                                else:
                                    loss += criterion_contrast(ff, labels) + criterion_contrast(ff2,labels2) \
                                            + criterion_contrast(ff3, labels3)  # /now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_contrast(ff4, labels4)
                        if opt.sphere:
                            if opt.loss_merge:
                                loss += criterion_sphere(ff_all, labels_all) / now_batch_size
                            else:
                                if opt.rendered_BEV:
                                    # replace labels3(drone) with labels5(BEV)
                                    loss += criterion_sphere(ff, labels) / now_batch_size + criterion_sphere(ff2,labels2) / now_batch_size \
                                            + criterion_sphere(ff3, labels5) / now_batch_size
                                else:
                                    loss += criterion_sphere(ff, labels) / now_batch_size + criterion_sphere(ff2, labels2) / now_batch_size \
                                            + criterion_sphere(ff3, labels3) / now_batch_size
                                if opt.extra_Google:
                                    loss += criterion_sphere(ff4, labels4)

                    else:
                        if model.itc and not(model.itm):
                            if opt.extra_Google:
                                [outputs, cls_feat], [outputs2, cls_feat2], [outputs3, cls_feat3], [outputs4, cls_feat4]  = outputs, outputs2, outputs3, outputs4
                                [d_outputs, d_cls_feat], [d_outputs2, d_cls_feat2], [d_outputs3, d_cls_feat3], [d_outputs4, d_cls_feat4]  = d_outputs, d_outputs2, d_outputs3, d_outputs4
                                satellite_based_feat_group = [cls_feat, cls_feat2, cls_feat3, cls_feat4]
                                drone_based_feat_group = [d_cls_feat, d_cls_feat2, d_cls_feat3, d_cls_feat4]
                            else:
                                [outputs, cls_feat], [outputs2, cls_feat2], [outputs3, cls_feat3] = outputs, outputs2, outputs3
                                [d_outputs, d_cls_feat], [d_outputs2, d_cls_feat2], [d_outputs3, d_cls_feat3] = d_outputs, d_outputs2, d_outputs3
                                satellite_based_feat_group = [cls_feat, cls_feat2, cls_feat3]
                                drone_based_feat_group = [d_cls_feat, d_cls_feat2, d_cls_feat3]

                            itc_loss = get_contrastive_loss_different_sources([satellite_based_feat_group, drone_based_feat_group], model.logit_scale)
                        elif model.itc and model.itm:
                            if opt.extra_Google:
                                [outputs, cls_feat, full_seq_1], [outputs2, cls_feat2, full_seq_2], [outputs3, cls_feat3, full_seq_3], [outputs4,
                                                                                                    cls_feat4 , full_seq_4] = outputs, outputs2, outputs3, outputs4
                                [d_outputs, d_cls_feat, d_full_seq_1], [d_outputs2, d_cls_feat2, d_full_seq_2], [d_outputs3, d_cls_feat3, d_full_seq_3], [
                                    d_outputs4, d_cls_feat4, d_full_seq_4] = d_outputs, d_outputs2, d_outputs3, d_outputs4
                                satellite_based_feat_group = [cls_feat, cls_feat2, cls_feat3, cls_feat4]
                                satellite_based_seq_feat_group = [full_seq_1, full_seq_2, full_seq_3, full_seq_4]
                                drone_based_feat_group = [d_cls_feat, d_cls_feat2, d_cls_feat3, d_cls_feat4]
                                drone_based_seq_feat_group = [d_full_seq_1, d_full_seq_2, d_full_seq_3, d_full_seq_4]
                            else:
                                [outputs, cls_feat, full_seq_1], [outputs2, cls_feat2, full_seq_2], [outputs3, cls_feat3, full_seq_3] = outputs, outputs2, outputs3
                                [d_outputs, d_cls_feat, d_full_seq_1], [d_outputs2, d_cls_feat2, d_full_seq_2], [d_outputs3, d_cls_feat3, d_full_seq_3] = d_outputs, d_outputs2, d_outputs3
                                satellite_based_feat_group = [cls_feat, cls_feat2, cls_feat3]
                                satellite_based_seq_feat_group = [full_seq_1, full_seq_2, full_seq_3]
                                drone_based_feat_group = [d_cls_feat, d_cls_feat2, d_cls_feat3]
                                drone_based_seq_feat_group = [d_full_seq_1, d_full_seq_2, d_full_seq_3]

                            itc_loss = get_contrastive_loss_different_sources(
                                [satellite_based_feat_group, drone_based_feat_group], model.logit_scale)
                            # satellite, street, drone(bev), google
                            # seq_feature_group, feat_groups, temper_parameter, model
                            itm_loss = get_matching_loss([satellite_based_seq_feat_group, drone_based_seq_feat_group],
                                                         [satellite_based_feat_group, drone_based_feat_group],
                                                         model.logit_scale,
                                                         model
                                                         )


                        _, preds = torch.max(outputs.data, 1)
                        _, preds2 = torch.max(outputs2.data, 1)
                        _, preds3 = torch.max(outputs3.data, 1)

                        _, d_preds = torch.max(d_outputs.data, 1)
                        _, d_preds2 = torch.max(d_outputs2.data, 1)
                        _, d_preds3 = torch.max(d_outputs3.data, 1)
                        if opt.loss_merge:
                            outputs_all = torch.cat((outputs, outputs2, outputs3), dim=0)
                            if opt.rendered_BEV:
                                # replace labels3(drone) with labels5(BEV)
                                labels_all = torch.cat((labels, labels2, labels5), dim=0)
                            else:
                                labels_all = torch.cat((labels, labels2, labels3), dim=0)
                            if opt.extra_Google:
                                outputs_all = torch.cat((outputs_all, outputs4), dim=0)
                                labels_all = torch.cat((labels_all, labels4), dim=0)
                            loss = 4*criterion(outputs_all, labels_all)
                        else:
                            if opt.rendered_BEV:
                                # replace labels3(drone) with labels5(BEV)
                                loss = criterion(outputs, labels) + criterion(outputs2, labels2) + criterion(outputs3, labels5)
                                d_loss = criterion(d_outputs, d_labels) + criterion(d_outputs2, d_labels2) + criterion(d_outputs3, d_labels5)
                            else:
                                loss = criterion(outputs, labels) + criterion(outputs2, labels2) + criterion(outputs3, labels3)
                                d_loss = criterion(d_outputs, d_labels) + criterion(d_outputs2, d_labels2) + criterion(d_outputs3, d_labels3)
                            if opt.extra_Google:
                                loss += criterion(outputs4, labels4)
                                d_loss += criterion(d_outputs4, d_labels4)
                if model.itc:
                    if opt.vit_itm:
                        loss = (d_loss + loss) / 2 + opt.loss_lambda * (itc_loss + itm_loss)
                    else:
                        # follow tingyu to average
                        loss = (d_loss + loss)/2 + opt.loss_lambda * itc_loss

                else:
                    loss = (d_loss + loss) / 2
                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16:  # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    ##########
                    if opt.moving_avg < 1.0:
                        update_average(model_test, model, opt.moving_avg)

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                    if opt.vit_itm:
                        running_loss_itm += itm_loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                try:
                    running_corrects2 += float(torch.sum(preds2 == labels2.data))
                except:
                    pass
                if cur_iter % 500 == 0:
                    tb_logger.add_scalar('iter/loss', running_loss/ (now_batch_size*(cur_iter+1)), cur_iter)
                    # tb_logger.add_scalar('iter/loss_itm', running_loss_itm/ (now_batch_size*(cur_iter+1)), cur_iter)
                    # tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                    if opt.vit_itc:
                        tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                    if opt.vit_itm:
                        tb_logger.add_scalar('loss/loss_itm', itm_loss, cur_iter)
                    tb_logger.add_scalar('iter/Satellite_Acc', running_corrects/ (now_batch_size*(cur_iter+1)), cur_iter)
                    tb_logger.add_scalar('iter/Street_Acc', running_corrects2/ (now_batch_size*(cur_iter+1)), cur_iter)
                    tb_logger.add_scalar('iter/Drone_Acc', running_corrects3/ (now_batch_size*(cur_iter+1)), cur_iter)
                if opt.views == 3 or opt.views == 2:
                    if opt.rendered_BEV:
                        # replace labels3(drone) with labels5(BEV)
                        running_corrects3 += float(torch.sum(preds3 == labels5.data))
                    else:
                        running_corrects3 += float(torch.sum(preds3 == labels3.data))




            epoch_loss = running_loss / dataset_size['satellite']
            epoch_acc = running_corrects / dataset_size['satellite']
            try:
                epoch_acc2 = running_corrects2 / dataset_size['satellite']
            except:
                pass

            # epoch_loss = running_loss / dataset_size
            # epoch_acc = running_corrects / dataset_size
            # epoch_acc2 = running_corrects2 / dataset_size

            if opt.views == 2:
                epoch_acc3 = running_corrects3 / dataset_size['satellite']
                # print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                #                                                                          epoch_acc2))
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                         epoch_acc3))
                write_txt(record_file_path, ('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                         epoch_acc3)))
                tb_logger.add_scalar('loss/loss_total', running_loss / dataset_size['satellite'], cur_iter)
                if opt.vit_itc:
                    tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                if opt.vit_itm:
                    tb_logger.add_scalar('loss/loss_itm', itm_loss, cur_iter)
                tb_logger.add_scalar('Satellite_Acc', running_corrects / dataset_size['satellite'], cur_iter)
                tb_logger.add_scalar('Street_Acc', running_corrects2 / dataset_size['satellite'], cur_iter)
                tb_logger.add_scalar('Drone_Acc', running_corrects3 / dataset_size['satellite'], cur_iter)
            elif opt.views == 3:
                epoch_acc3 = running_corrects3 / dataset_size['satellite']
                # epoch_acc3 = running_corrects3 / dataset_size
                print('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase,
                                                                                                           epoch_loss,
                                                                                                           epoch_acc,
                                                                                                           epoch_acc2,
                                                                                                           epoch_acc3))
                write_txt(record_file_path, ('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase,
                                                                                                           epoch_loss,
                                                                                                           epoch_acc,
                                                                                                           epoch_acc2,
                                                                                                           epoch_acc3)))
                tb_logger.add_scalar('loss/loss_total', running_loss/ dataset_size['satellite'], cur_iter)
                if opt.vit_itc:
                    tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                if opt.vit_itm:
                    tb_logger.add_scalar('loss/loss_itm', itm_loss, cur_iter)
                tb_logger.add_scalar('Satellite_Acc', running_corrects/ dataset_size['satellite'], cur_iter)
                tb_logger.add_scalar('Street_Acc', running_corrects2/ dataset_size['satellite'], cur_iter)
                tb_logger.add_scalar('Drone_Acc', running_corrects3/ dataset_size['satellite'], cur_iter)
                # tb_logger.add_scalar('epoch/loss', running_loss / dataset_size, cur_iter)
                # tb_logger.add_scalar('epoch/Satellite_Acc', running_corrects / dataset_size, cur_iter)
                # tb_logger.add_scalar('epoch/Street_Acc', running_corrects2 / dataset_size, cur_iter)
                # tb_logger.add_scalar('epoch/Drone_Acc', running_corrects3 / dataset_size, cur_iter)



            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                scheduler.step()
                lr_l = scheduler.get_lr()
                for i, lr in enumerate(lr_l):
                    tb_logger.add_scalar('lr/lr_{}'.format(i), lr, cur_iter)

            # last_model_wts = model.state_dict()
            if epoch % 20 == 19:
            # if epoch % 2 == 0:
                # save_network(model, opt.name, epoch)
                save_network(model, model_name, opt.name, epoch)
            # draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        write_txt(record_file_path, ('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    write_txt(record_file_path, ('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)))
    # print('Best val Acc: {:4f}'.format(best_acc))
    # save_network(model_test, opt.name+'adapt', epoch)

    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere or opt.auxiliary

if opt.views == 2:
    model = two_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                           share_weight=opt.share, circle=return_feature,ibn=opt.ibn,vit=opt.vit,itc = opt.vit_itc,itm=opt.vit_itm,itm_share=opt.vit_itm_share,lpn=opt.lpn)
elif opt.views == 3:
    raise Exception('not surpport')
    model = three_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                           share_weight=opt.share, circle=return_feature,ibn=opt.ibn,vit=opt.vit,itc = opt.vit_itc,itm=opt.vit_itm,itm_share=opt.vit_itm_share)

opt.nclasses = len(class_names)
# opt.nclasses = (class_names)

print(model)
# For resume:
if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

if opt.moving_avg < 1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    # num_epochs = 120
    num_epochs = opt.epoch
    # num_epochs = 1

optimizer_grouped_parameters = [
        # base parameters
        {"params": [], "weight_decay": opt.wd, "lr": 0.1 * opt.lr},
        # classification head for the instance loss
        {"params": [], "weight_decay": opt.wd, "lr": opt.lr_instance},
        # {"params": [], "weight_decay": opt.wd, "lr": opt.lr},
        ]
if opt.vit_itm or opt.vit_itc:
    if (not opt.vit_itm) and (opt.vit_itc):#lr_itm
        # itc only
        # logit_scale
        optimizer_grouped_parameters.append({"params": [], "weight_decay": opt.wd, "lr": opt.lr_itc})
    else:
        raise Exception('only for the first stage')




no_decay = {"bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2_1.bias",
            "norm2_1.weight",
            "norm2.bias",
            "norm2.weight",
            "norm3.bias",
            "norm3.weight",}

if not opt.lpn:
    # ignored_params = list(map(id, model.classifier.parameters()))
    mssg_l = []
    l_lr_l = []
    for n, p in model.classifier.named_parameters():
        optimizer_grouped_parameters[1]['params'].append(p)
        l_lr_l.append('classifier.{}'.format(n))
        mssg_l.append('{} use larger lr: {}'.format(n, opt.lr_instance))
else:
    mssg_l = []
    l_lr_l = []
    ignored_params = []
    for i in range(5):
        name_ = 'classifier' + str(i)
        c = getattr(model, name_)
        ignored_params.append(map(id, c.parameters()))
        for n, p in c.named_parameters():
            optimizer_grouped_parameters[1]['params'].append(p)
            l_lr_l.append('{}.{}'.format(name_, n))
            mssg_l.append('{} use larger lr: {}'.format(n, opt.lr_instance))


# ignored_params_1 = list(map(id, model.model_1_fusion.parameters()))
# ignored_params_2 = list(map(id, model.model_2_fusion.parameters()))
# todo: check from here
if opt.vit_itc:
    l_lr_l.append('{}'.format('logit_scale'))
    optimizer_grouped_parameters[2]['params'].append(model.logit_scale)
    mssg_l.append('Note: {} use lr: {}'.format('logit_scale', opt.lr))


for n, p in model.named_parameters():
    if n in l_lr_l:
        continue
    else:
        optimizer_grouped_parameters[0]['params'].append(p)

tmp = 0
for i in optimizer_grouped_parameters:
    tmp=tmp+len(i['params'])
# if all parameters are in optimizer
assert len(dict(model.named_parameters()).keys()) == tmp


for name_, param in model.named_parameters():
    # if id(param) in ignored_params:
    if name_ in l_lr_l:
        # print('{} use larger lr: {}'.format(name_, opt.lr))
        print('{} use larger lr'.format(name_))
        # mssg_l.append('{} use larger lr: {}'.format(name_, opt.lr))
        mssg_l.append('{} use larger lr'.format(name_))

if opt.optimizer=='SGD':

    optimizer_ft = optim.SGD(optimizer_grouped_parameters, momentum=0.9, nesterov=True)
elif opt.optimizer=='AdamW':

    from torch.optim import AdamW

    optimizer_ft = AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.98))

else:
    raise Exception('Not support {} optimizer'.format(opt.optimizer))
# Decay LR by a factor of 0.1 every 40 epochs
##### scheduler
if opt.scheduler== 'steplr':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=120, gamma=0.1)


elif opt.scheduler == 'linearlr_warmup':

    from torch.optim.lr_scheduler import LambdaLR

    opt.num_training_steps = num_epochs
    opt.num_warmup_steps = int(0.1 * opt.num_training_steps)
    # opt.step_size = num_epochs * opt.step_size
    mssg_l.append('total epoches:{}'.format(opt.num_training_steps))
    mssg_l.append('iterations:{} per epoch'.format(len(dataloader)))
    mssg_l.append(
        'warmup epoches:{} ({} iterations)'.format(opt.num_warmup_steps, opt.num_warmup_steps * len(dataloader)))


    # mssg_l.append('decay every {} epoches ({} iterations)'.format(opt.step_size, opt.step_size*len(dataloader)))

    def lr_lambda(current_step: int):
        if current_step < opt.num_warmup_steps:
            return float(current_step) / float(max(1, opt.num_warmup_steps))
        elif current_step < opt.num_warmup_steps * 4:
            tt = 1
        elif current_step < opt.num_warmup_steps * 7:
            tt = 0.5
        else:
            tt = 0.2

        return tt * max(
            0.0,
            float(opt.num_training_steps - current_step) /
            float(max(1, opt.num_training_steps - opt.num_warmup_steps))
        )

    exp_lr_scheduler = LambdaLR(optimizer_ft, lr_lambda, last_epoch=-1)
elif opt.scheduler== 'cosine':
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_ft,T_max=num_epochs,eta_min=1e-6)
else:
    raise Exception('Not support {} scheduler'.format(opt.scheduler))


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
# print(dataset_size)
currentTime = time.localtime()
model_name = 'model_{}'.format(time.strftime("%Y-%m-%d-%H:%M:%S", currentTime))
# opt.name = model_name
current_file = __file__
absolute_path = os.path.dirname(os.path.abspath(current_file))
dir_name = os.path.join(absolute_path,model_name, name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    txt_path = os.path.join(dir_name,'training_record.txt')
    tb_path = os.path.join(dir_name,'tb_logger')
    if not os.path.exists(tb_path):
        os.mkdir(tb_path)
    tb_logger = init_tb_logger(tb_path)
    write_txt(txt_path,'BEV:{}'.format(opt.rendered_BEV))
    write_txt(txt_path,'auxiliary loss:{}'.format(opt.auxiliary))
    if opt.ibn:
        write_txt(txt_path,'backbone:{}'.format('ResNet50-IBN'))
    elif opt.vit:
        write_txt(txt_path, 'backbone:{}'.format('vit-small'))
    else:
        write_txt(txt_path, 'backbone:{}'.format('ResNet50'))
    write_txt(txt_path,dataset_size)

    write_txt(txt_path, 'Use Image text contrastive loss:{}'.format(opt.vit_itc))
    write_txt(txt_path, 'Use Image text matching loss:{}'.format(opt.vit_itm))
    write_txt(txt_path, 'Use Image text matching loss with a shared head:{}'.format(opt.vit_itm_share))
    write_txt(txt_path, 'lambda for loss:{}'.format(opt.loss_lambda))
    for mssg in mssg_l:
        write_txt(txt_path, '{}'.format(mssg))

    write_txt(txt_path,'optimizer {}'.format(opt.optimizer))
    write_txt(txt_path,'scheduler {}'.format(opt.scheduler))
    write_txt(txt_path,'learning rate {}'.format(opt.lr))

    write_txt(txt_path,model)

    # record every run
    copyfile('train_bev_paired_fsra.py', dir_name + '/train_bev_paired_fsra.py')
    # copyfile('./model.py', dir_name + '/model.py')
    copyfile(os.path.join(absolute_path,'model.py'), dir_name + '/model.py')
    copyfile(os.path.join(absolute_path,'train.sh'), dir_name + '/train.sh')
    copyfile(os.path.join(absolute_path,'backbone','vit_pytorch.py'), dir_name + '/vit_pytorch.py')
    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

criterion = nn.CrossEntropyLoss()
# if opt.moving_avg < 1.0:
#     model_test = copy.deepcopy(model)
#     num_epochs = 140
# else:
#     model_test = None
#     num_epochs = 120
#     # num_epochs = 1

model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=num_epochs,tb_logger=tb_logger,record_file_path= txt_path,model_name=model_name)

