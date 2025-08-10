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
from functools import partial
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
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str,
                    help='output model name')  # 'two_view'
parser.add_argument('--resume', action='store_true', help='use resume trainning')
# data
parser.add_argument('--data_dir', default='/university1652k/University-Release/train', type=str,
                    help='training dir path')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')

# parser.add_argument('--rendered_BEV', action='store_true', default=False, help='using extra noise Google')  # ,default=False
parser.add_argument('--rendered_BEV', action='store_true', default=True,
                    help='using extra noise Google')  # ,default=False

parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
# parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--sample_num', default=1, type=int, help='num of repeat sampling')
# parser.add_argument('--sample_num', default=2, type=int, help='num of repeat sampling' )
parser.add_argument('--num_worker', default=12, type=int, help='num of worker')
parser.add_argument('--gamma', default=0.9, type=float, help='num of worker')
parser.add_argument('--step_size', default=20, type=float, help='num of worker')
parser.add_argument('--save_epoch', default=20, type=float, help='num of worker')
# backbone

parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')

parser.add_argument('--ibn', action='store_true', default=False, help='use NAS')
# parser.add_argument('--ibn', action='store_true',default=True, help='use NAS')

# parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--share', action='store_true', default=True, help='share weight between different view')

parser.add_argument('--vit', action='store_true', default=True, help='use NAS')
# parser.add_argument('--vit', action='store_true', default=False, help='use NAS')

parser.add_argument('--backbone_name', type=str, default='swin-base', help='use NAS')
# 94.58;94.29
parser.add_argument('--first_stage_weight_path', type=str, default='/18141169908/hao/itm/rebuttal/backbones_weight/swin-base/net_139.pth', help='use NAS')

#
parser.add_argument('--vit_itc', action='store_true', default=True, help='use NAS')
# parser.add_argument('--vit_itc', action='store_true', default=False, help='use NAS')

parser.add_argument('--vit_itm', action='store_true', default=True, help='use NAS')
# parser.add_argument('--vit_itm', action='store_true', default=False, help='use NAS')

parser.add_argument('--vit_itm_share', action='store_true', default=True, help='use NAS')
# parser.add_argument('--vit_itm_share', action='store_true', default=False, help='use NAS')

# parser.add_argument('--fusion_cls_loss', action='store_true',default=True,help='losses between bev and satellite')#, default=False
parser.add_argument('--fusion_cls_loss', action='store_true', default=False,
                    help='losses between bev and satellite')  # , default=False

# parser.add_argument('--sd_negative_sample', action='store_true', default=False, help='use NAS')
parser.add_argument('--sd_negative_sample', action='store_true', default=True, help='use NAS')

parser.add_argument('--two_stage_training', action='store_true', default=True,
                    help='losses between bev and satellite')  # , default=False
# parser.add_argument('--two_stage_training', action='store_true', default=False, help='losses between bev and satellite')  # , default=False
# matching label_smooth
# parser.add_argument('--matching_label_smooth', action='store_true',default=True,help='losses between bev and satellite')#, default=False
parser.add_argument('--matching_label_smooth', action='store_true', default=False,
                    help='losses between bev and satellite')  # , default=False
parser.add_argument('--lpn', action='store_true', default=True, help='use NAS')
# parser.add_argument('--lpn', action='store_true',default=False, help='use NAS')


parser.add_argument('--num_training_steps', default=None, help='losses between bev and satellite')  # , default=False
parser.add_argument('--num_warmup_steps', type=int, default=200, help='losses between bev and satellite')  # , default=False
parser.add_argument('--epoch', default=0, type=int, help='losses between bev and satellite')  # , default=False

# optimizer
parser.add_argument('--optimizer', default='AdamW', help='use all training data')
# parser.add_argument('--optimizer', action='store_true',default='AdamW', help='use all training data')
parser.add_argument('--scheduler', default='linearlr_warmup', help='use all training data')
# parser.add_argument('--scheduler', action='store_true',default='cosine', help='use all training data')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--neg_sample_number', default=3, type=int, help='the first K epoch that needs warm up')
# parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
parser.add_argument('--lr', default=0.04, type=float, help='learning rate')
parser.add_argument('--lr_itm', default=0.04, type=float, help='learning rate')
parser.add_argument('--lr_instance', default=0.04, type=float, help='learning rate')
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
parser.add_argument('--loss_lambda', default=1, help='loss balance')


opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0


def write_txt(record_file_path, content):
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
if opt.backbone_name=='vit-small':
    pass
elif opt.backbone_name=='swin-tiny' or opt.backbone_name=='swin-base' or opt.backbone_name=='vit-base':
    opt.h = 224
    opt.w = 224
else:
    raise NotImplementedError
# Load Data
dataloader, class_names, dataset_size, drone_based_dataloader = make_dataset(opt)
opt.nclasses = len(class_names)
print(dataset_size)

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


def tb_logger_add_images(tb_logger, tensors, current_iter):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tmp_l = []
    for tensor_ in tensors:
        # tmp = 255*(tensor_*torch.from_numpy(np.array(std)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1))+torch.from_numpy(np.array(mean)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1)
        tmp = (255 * (tensor_.mul_(
            torch.from_numpy(np.array(std)).unsqueeze(0).repeat(tensor_.shape[0], 1).unsqueeze(-1).unsqueeze(-1)).add_(
            torch.from_numpy(np.array(mean)).unsqueeze(0).repeat(tensor_.shape[0], 1).unsqueeze(-1).unsqueeze(
                -1)))).type(torch.uint8)
        tmp_l.append(tmp)
    # pass
    img = torchvision.utils.make_grid(torch.cat(tmp_l, dim=0), nrow=tensor_.shape[0], padding=2, pad_value=255)
    tb_logger.add_image('input_images', img, current_iter)


def save_tensor(tensors, save_path, name):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tmp_l = []
    for tensor_ in tensors:
        # tmp = 255*(tensor_*torch.from_numpy(np.array(std)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1))+torch.from_numpy(np.array(mean)).unsqueeze(0).repeat(tensor_.shape[0],1).unsqueeze(-1).unsqueeze(-1)
        tmp = (255 * (tensor_.mul_(
            torch.from_numpy(np.array(std)).unsqueeze(0).repeat(tensor_.shape[0], 1).unsqueeze(-1).unsqueeze(-1)).add_(
            torch.from_numpy(np.array(mean)).unsqueeze(0).repeat(tensor_.shape[0], 1).unsqueeze(-1).unsqueeze(
                -1)))).type(torch.uint8)
        tmp_l.append(tmp)
    # pass
    img = torchvision.utils.make_grid(torch.cat(tmp_l, dim=0), nrow=tensor_.shape[0], padding=2, pad_value=255)
    import cv2
    img = img.permute(1, 2, 0).byte().numpy()
    cv2.imwrite(os.path.join(save_path, name), img)


def save_feature(tensors, save_path, name):
    tmp_l = []
    for tensor_ in tensors:
        b_l = []
        for b_idx in range(tensor_.shape[0]):
            tensor = tensor_[b_idx].unsqueeze(0)
            # tmp = tensor
            tmp = (tensor - torch.min(tensor_)) / (torch.max(tensor) - torch.min(tensor))
            # tmp *= 255
            # tensorboard 会自动归一化 所以不用手动归一化
            # tmp = tensor_
            import torch.nn.functional as F
            tmp = F.interpolate(tmp, size=[256, 256], mode='bilinear', align_corners=True)
            b_l.append(tmp)
        tmp = torch.cat(b_l, dim=0)
        tmp_l.append((tmp.repeat(1, 3, 1, 1)))
    # pass
    img = torchvision.utils.make_grid(torch.cat(tmp_l, dim=0), nrow=tensor_.shape[0], padding=2, pad_value=255)
    return img

def unflatten(x):
    bs, n, c = x[:, 1:, :].shape
    cls = x[:, :1, :]

    tmp_l = []
    for i in range(0, n, 16):
        chunk = x[:, 1:, :][:, i:(i + 16), :]
        tmp_l.append(chunk)
    x_unflatten = torch.stack(tmp_l, dim=1).permute(0, 3, 1, 2)
    x_unflatten = torch.mean(x_unflatten, dim=1).unsqueeze(1).detach().cpu()
    return x_unflatten


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
    logits_two_one = modality_two @ modality_one.t() / temper_parameter
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


def get_contrastive_loss_different_sources(feat_groups, temper_parameter):
    """

    Args:
        feat_groups:[satellite_based_samples, drone_based_samples]
            satellite_based_samples:[ground, satellite, drone or bev, google]
        temper_parameter: learnable parameter for similarity

    Returns:

    """
    satellite_based_sample_l = feat_groups[0]
    drone_based_sample_l = feat_groups[1]
    satellite_based_paired_l = compose_pairs_for_contrastive_loss(satellite_based_sample_l)
    # satellite_based_paired_l = [satellite_based_sample_l]
    drone_based_paired_l = compose_pairs_for_contrastive_loss(drone_based_sample_l)
    # drone_based_paired_l = [drone_based_sample_l]
    itc_loss = 0.
    for modality_one, modality_two in satellite_based_paired_l:
        itc_loss += get_contrastive_loss(modality_one, modality_two, temper_parameter)
    # itc_loss /= len(satellite_based_paired_l)

    for modality_one, modality_two in drone_based_paired_l:
        itc_loss += get_contrastive_loss(modality_one, modality_two, temper_parameter)
    itc_loss /= (len(drone_based_paired_l) + len(satellite_based_paired_l))
    # itc_loss /= 2
    return itc_loss


import cv2


def pad_color(src_tsr, color, pad_width=4, gt=False):
    vertical_strip = torch.zeros_like(src_tsr[:, :pad_width, :])
    horizontal_strip = torch.zeros([pad_width, (pad_width * 2 + src_tsr.shape[1]), 3])
    if color:
        # green
        vertical_strip[:, :, 1] = 255
        horizontal_strip[:, :, 1] = 255
    else:
        if gt:
            # red
            vertical_strip[:, :, 2] = 255
            horizontal_strip[:, :, 2] = 255
        else:
            # red
            vertical_strip[:, :, 0] = 255
            horizontal_strip[:, :, 0] = 255
    src_tsr_ = torch.cat([vertical_strip, src_tsr, vertical_strip], dim=1)
    src_tsr_ = torch.cat([horizontal_strip, src_tsr_, horizontal_strip], dim=0)
    return src_tsr_


def add_text(image, mssg):
    image = cv2.putText(image, "{}".format(mssg), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    return image


def visualization_img_q_g(query_path, gallery_path_l, matching_result, score_topk):
    """
    visualization samples: query and gallery
    matched gallery sample : pad green
    missed gallery sample : pad red
    Args:
        query_path: should one sample
        gallery_path_l: should be a list of sample path
        matching_result：bool, denotes matched (True) or missed (False)

    Returns:

    """
    # query pad green
    small_size = [128,128]
    q_tsr = torch.from_numpy(cv2.resize(cv2.imread(query_path),small_size))
    q_tsr_pad = pad_color(q_tsr, True)

    gallery_pad_l = []
    for idx, (gallery_sample_path, matching_result_i) in enumerate(zip(gallery_path_l, matching_result)):
        g_tsr = torch.from_numpy(add_text(cv2.resize(cv2.imread(gallery_sample_path),small_size), score_topk[idx]))
        if g_tsr.shape != q_tsr.shape:
            g_tsr = g_tsr.to(torch.float32).resize_as_(q_tsr.to(torch.float32))
        assert g_tsr.shape == q_tsr.shape
        if matching_result_i:
            # pad green
            g_tsr_pad = pad_color(g_tsr, matching_result_i)
        else:
            # pad red
            g_tsr_pad = pad_color(g_tsr, matching_result_i)
        if len(gallery_pad_l) != 0:
            assert g_tsr_pad.shape == gallery_pad_l[-1].shape
        gallery_pad_l.append(g_tsr_pad)

    # gt_tsr = torch.from_numpy(add_text(np.array(cv2.imread(gt_abs_path)), '{}'.format(g_equal_q_score)))
    # # pad blue
    # gt_tsr = pad_color(gt_tsr, False, gt=True)
    # if gt_tsr.shape != gallery_pad_l[-1].shape:
    #     gt_tsr = torch.from_numpy(np.array(
    #         torchvision.transforms.Resize(size=(gallery_pad_l[-1].shape[0], gallery_pad_l[-1].shape[1]))((np.uint8(gt_tsr)))))
    # gallery_pad_l.append(gt_tsr)

    # list: query, zero, [gallery]
    white_ = torch.zeros_like(q_tsr_pad) + 255
    gallery_pad_l.insert(0, white_)
    gallery_pad_l.insert(0, q_tsr_pad)
    # torchvision
    v_img = torchvision.utils.make_grid((torch.stack(gallery_pad_l, dim=0)).permute(0, 3, 1, 2),
                                        nrow=len(gallery_pad_l), padding=2, pad_value=255).permute(1, 2, 0)
    return v_img


def visualize_anchor_and_negative_sample(anchor_name, gallery_name, anchor_id, gallery_id, save_path, save_name,
                                         **kwargs):
    save_name = '{}_{}2{}_{}'.format(save_name, anchor_name, gallery_name, anchor_id)
    for gallery_id_ in gallery_id:
        save_name += '_{}'.format(gallery_id_)
    save_name += '.jpg'
    dataset_root_path = '/university1652k/University-Release/train'
    anchor_path = os.path.join(os.path.join(dataset_root_path, anchor_name, anchor_id),
                               os.listdir(os.path.join(dataset_root_path, anchor_name, anchor_id))[0])
    neg_sample_path_l = []
    for gallery_id_ in gallery_id:
        neg_sample_path = os.path.join(os.path.join(dataset_root_path, gallery_name, gallery_id_),
                                       os.listdir(os.path.join(dataset_root_path, gallery_name, gallery_id_))[0])
        neg_sample_path_l.append(neg_sample_path)


    # 将 a 列表复制为 b 列表的长度
    anchor_id_ = [copy.deepcopy(anchor_id)]
    while len(anchor_id_) < len(gallery_id):
        anchor_id_.append(anchor_id)
    # matching_result = equal(anchor_id_ == gallery_id)
    matching_result = [x == y for x, y in zip(anchor_id_, gallery_id)]

    new_img = visualization_img_q_g(anchor_path, neg_sample_path_l, matching_result, kwargs.get('score_topk'))


    # anchor_img = cv2.imread(anchor_path)
    # neg_img = cv2.imread(neg_sample_path)
    #
    # height1, width1, _ = anchor_img.shape
    # height2, width2, _ = neg_img.shape
    #
    # tmp = max(height1, width1, height2, width2)
    # height1, width1, height2, width2 = tmp, tmp, tmp, tmp
    # # 创建一个新的图片,宽度为两张图片宽度之和加上中间的5个像素白空隙
    # new_width = width1 + width2 + 5
    # # new_height = max(height1, width1, height2, width2)
    # new_img = np.zeros((height1, new_width, 3), dtype=np.uint8)
    #
    # # 将两张图片横向拼接到新的图片上
    # h_w = max(height1, width1)
    # new_img[:, :width1, :] = cv2.resize(anchor_img, (h_w, h_w))
    # new_img[:, width1 + 5:, :] = cv2.resize(neg_img, (h_w, h_w))

    # 保存拼接后的图片
    save_path = os.path.join(save_path, save_name)
    cv2.imwrite(save_path, np.uint8(new_img))


def swap_first_element(a, value):

    # 找到值的索引
    index = a.index(value)

    # 如果索引为 0 则直接返回列表
    if index == 0:
        return a

    # 交换第一个元素和目标元素
    a[0], a[index] = a[index], a[0]

    return a
def compose_neg_pair_from_similarity(similarity, full_seq_l_other_source, sd_data=False, **kwargs):
    """

    Args:
        similarity: 相似性
        full_seq_l_other_source: 另一个领域的feature

    Returns:

    """
    label = kwargs.get('label')
    sample_number = kwargs.get('sample_number')
    id_names = kwargs.get('id_names')
    epoch = kwargs.get('epoch')
    iteration = kwargs.get('iteration')
    anchor_name = kwargs.get('anchor')
    gallery_name = kwargs.get('gallery_')

    training_dir_path = os.path.dirname(txt_path)
    negative_sample_dir = os.path.join(training_dir_path, 'neagtive_samples')
    cur_negative_sample_dir_path = os.path.join(negative_sample_dir, '{:06d}'.format(epoch))
    if not os.path.exists(cur_negative_sample_dir_path):
        os.makedirs(cur_negative_sample_dir_path)

    neg_pair_l = []
    neg_pair_label_l = []
    bs = similarity.shape[0]
    mssg_ = 'cur_batch_id:'
    for i in label:
        mssg_ += '{:04d}_'.format(int(class_names[np.array(i.detach().cpu())]))
    write_txt(os.path.join(cur_negative_sample_dir_path, 'neg_sample.txt'), mssg_)
    for b in range(bs):
        # 从多项式分布抽取样本
        if sample_number == 1:
            # # 只抽取一个负样本(random)
            neg_idx = torch.multinomial(similarity[b], 1).item()
            neg_pair_l.append(full_seq_l_other_source[neg_idx])

            write_txt(os.path.join(cur_negative_sample_dir_path, 'neg_sample.txt'),
                      'epoch{:03d}\iteration{:02d}: {} -> {} positive id:{} negative pair id:{}'.format(
                          epoch, iteration, anchor_name, gallery_name, id_names[label[b]], id_names[label[neg_idx]]))
            # visualize_anchor_and_negative_sample(anchor_name, gallery_name, id_names[label[b]],
            #                                      id_names[label[neg_idx]], cur_negative_sample_dir_path,
            #                                      'epoch{}\iteration{}\Batch{:03d}'.format(epoch, iteration, b))
            neg_idx_ranking = list(np.array((torch.argsort(similarity[b], descending=True)).detach().cpu()))
            neg_idx_ranking = swap_first_element(neg_idx_ranking, neg_idx)
            g_idx = [int(x) for x in (label[neg_idx_ranking].detach().cpu())]
            # visualize_anchor_and_negative_sample(
            #     anchor_name=anchor_name, gallery_name=gallery_name, anchor_id=id_names[label[b]],
            #     gallery_id=[id_names[x] for x in g_idx], save_path=cur_negative_sample_dir_path,
            #     save_name='epoch{}\iteration{}\Batch{:03d}'.format(epoch, iteration,b),
            #     score_topk=similarity[b][neg_idx_ranking])
        # if label is not None:
        # 	neg_pair_label_l.append(label[neg_idx])

        # neg_idx = (np.array((torch.argsort(similarity[b], descending=True)[0]).detach().cpu()))
        # # write_txt(txt_path, 'negative pair id:{}'.format(neg_idx))
        # neg_pair_l.append(full_seq_l_other_source[neg_idx])
        else:
            # 抽取多个负样本
            # neg_idx = list(np.array(torch.multinomial(similarity[b], sample_number).detach().cpu()))
            neg_idx = list(np.array((torch.argsort(similarity[b], descending=True)[:sample_number]).detach().cpu()))
            # neg_idx_ranking = list(np.array((torch.argsort(similarity[b], descending=True)).detach().cpu()))
            if not sd_data:
                assert b not in neg_idx
            neg_idx_ranking = list(np.array((torch.argsort(similarity[b], descending=True)[:sample_number]).detach().cpu()))
            # neg_idx_ranking = list(np.array(torch.multinomial(similarity[b], sample_number).detach().cpu()))

            mssg = 'epoch{:03d}\iteration{:02d}: {} -> {} positive id:{} negative id:'.format(epoch, iteration, anchor_name, gallery_name, id_names[label[b]])
            for rankid, neg_idx_ in enumerate(neg_idx):
                mssg += '{}_'.format(id_names[label[neg_idx_]])
                # write_txt(os.path.join(cur_negative_sample_dir_path, 'neg_sample.txt'),
                #           'epoch{:03d}\iteration{:02d}: {} -> {} positive id:{} negative pair id:{}'.format(
                #               epoch, iteration, anchor_name, gallery_name, id_names[label[b]], id_names[label[neg_idx_]]))
            write_txt(os.path.join(cur_negative_sample_dir_path, 'neg_sample.txt'),mssg)

            if iteration%500==0:
                pass
                # g_idx = [int(x) for x in (label[neg_idx_ranking].detach().cpu())]
                # visualize_anchor_and_negative_sample(
                #     anchor_name=anchor_name, gallery_name=gallery_name, anchor_id=id_names[label[b]],
                #     gallery_id=[id_names[x] for x in g_idx], save_path=cur_negative_sample_dir_path, save_name='epoch{}\iteration{}\Batch{:03d}'.format(epoch, iteration,b),
                #     score_topk=similarity[b][neg_idx_ranking])

            neg_pair_l.append(full_seq_l_other_source[neg_idx])
    # other_source_neg = torch.stack(neg_pair_l, dim=0)
    if sample_number == 1:
        other_source_neg = torch.stack(neg_pair_l, dim=0)
    else:
        other_source_neg = torch.cat(neg_pair_l, dim=0)
    # write_txt(txt_path, 'next_batch')
    return other_source_neg


def repeat_tensor(input_tensor, repeat_times):
    """
    将输入 tensor 沿着第一个维度重复 repeat_times 次,并按照 [0, 0, 0, 1, 1, 1, ..., (B-1), (B-1), (B-1)] 的次序重复。

    Args:
        input_tensor (torch.Tensor): 输入 tensor,维度为 [B, C, D]
        repeat_times (int): 重复的次数

    Returns:
        torch.Tensor: 重复后的 tensor,维度为 [repeat_times*B, C, D]
    """
    # 获取输入 tensor 的第一个维度的大小 B
    B = input_tensor.size(0)

    # 计算重复的次序
    repeat_indices = torch.tensor([i // repeat_times for i in range(repeat_times * B)])

    # 根据重复的次序重复输入 tensor
    output_tensor = input_tensor[repeat_indices]

    return output_tensor


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        epsilon (float): weight.
    """

    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        _, num_classes = inputs.shape
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss




def get_matching_loss_two_view(seq_feature_group, feat_groups, temper_parameter, model, **kwargs):
    """

    Args:
        seq_feature_group: vit_token_seq [satellite, (drone)bev]
        feat_groups: vit_cls_token [satellite, (drone)bev]
        temper_parameter: scaling factor for similarity
        model: full model(satellite, street, (drone)bev, google)

    Returns:

    """
    opt = kwargs['opt']
    satellite_based_label = kwargs['satellite_based_label']
    drone_based_label = kwargs['drone_based_label']
    sample_number = kwargs.get('sample_number')
    id_names = kwargs.get('class_names')
    # epoch = epoch, iteration = iters
    iteration = kwargs.get('iteration')
    epoch = kwargs.get('epoch')
    # seq_feature_group_neg = [satellite_based_seq_feat_group_neg, drone_based_seq_feat_group_neg],
    # feat_groups_neg = [satellite_based_feat_group_neg, drone_based_feat_group_neg],
    if opt.sd_negative_sample:
        # input for fusion block
        satellite_based_seq_feat_group_neg, drone_based_seq_feat_group_neg = kwargs.get('seq_feature_group_neg')
        satellite_based_satellite_full_seq_neg, satellite_based_drone_or_bev_full_seq_neg = satellite_based_seq_feat_group_neg[0], satellite_based_seq_feat_group_neg[1]
        drone_based_satellite_full_seq_neg, drone_based_drone_or_bev_full_seq_neg = drone_based_seq_feat_group_neg[0], drone_based_seq_feat_group_neg[1]
        # for computing the similarity and choose the negative samples
        satellite_based_feat_neg, drone_based_feat_neg = kwargs.get('feat_groups_neg')
        satellite_based_satellite_cls_neg, satellite_based_drone_or_bev_cls_neg = satellite_based_feat_neg[0], satellite_based_feat_neg[1]
        drone_based_satellite_cls_neg, drone_based_drone_or_bev_cls_neg = drone_based_feat_neg[0], drone_based_feat_neg[1]

    # txt_path = kwargs.get('txt_path')
    satellite_based_sample_l = feat_groups[0]
    satellite_based_satellite_cls = satellite_based_sample_l[0]
    satellite_based_drone_or_bev_cls = satellite_based_sample_l[1]

    drone_based_sample_l = feat_groups[1]
    drone_based_satellite_cls = drone_based_sample_l[0]
    drone_based_drone_or_bev_cls = drone_based_sample_l[1]

    satellite_based_seq_l = seq_feature_group[0]
    satellite_based_satellite_full_seq = satellite_based_seq_l[0]
    satellite_based_drone_or_bev_full_seq = satellite_based_seq_l[1]

    drone_based_seq_l = seq_feature_group[1]
    drone_based_satellite_full_seq = drone_based_seq_l[0]
    drone_based_drone_or_bev_full_seq = drone_based_seq_l[1]

    # forward the positive drone(bev)-satellite
    s_fusion_output_pos, _ = model.model_3_fusion(satellite_based_drone_or_bev_full_seq,
                                                  satellite_based_satellite_full_seq)

    with torch.no_grad():

        similarity_drone_satellite = satellite_based_drone_or_bev_cls @ satellite_based_satellite_cls.t()
        # similarity_drone_satellite = (satellite_based_drone_or_bev_cls @ satellite_based_satellite_cls.t())/model.logit_scale

        weights_d2s = F.softmax(similarity_drone_satellite[:, :], dim=1) + 1e-5
        weights_d2s.fill_diagonal_(0)
    # 从satellite中选出负样本给drone(bev)
    if opt.fusion_cls_loss:
        satellite_based_neg4drone, _ = compose_neg_pair_from_similarity(similarity=weights_d2s,
                                                                        full_seq_l_other_source=satellite_based_satellite_full_seq,
                                                                        label=satellite_based_label)
    else:
        satellite_based_neg4drone = compose_neg_pair_from_similarity(similarity=weights_d2s,
                                                                     full_seq_l_other_source=satellite_based_satellite_full_seq,
                                                                     sample_number=sample_number,
                                                                     label=satellite_based_label,
                                                                     id_names=id_names
                                                                     , epoch=epoch, iteration=iteration, anchor='bev',
                                                                     gallery_='satellite')
    if opt.sd_negative_sample:
        # select negative pair for satellite from sd output
        with torch.no_grad():
            similarity_drone_drone_neg = satellite_based_drone_or_bev_cls @ satellite_based_drone_or_bev_cls_neg.t()
            # similarity_drone_satellite = (satellite_based_drone_or_bev_cls @ satellite_based_satellite_cls.t())/model.logit_scale

            weights_d2d_neg = F.softmax(similarity_drone_drone_neg[:, :], dim=1)# + 1e-5
            # weights_d2s.fill_diagonal_(0)

        # note: negative sample source is from drone_or_bev not satellite
        #       because we use sd generated samples from itself as negative samples
        satellite_based_neg4drone_sd = compose_neg_pair_from_similarity(similarity=weights_d2d_neg, sd_data=True,
                                                                     full_seq_l_other_source=satellite_based_drone_or_bev_full_seq_neg,
                                                                     sample_number=sample_number,
                                                                     label=satellite_based_label,
                                                                     id_names=id_names, epoch=epoch, iteration=iteration, anchor='bev',
                                                                     gallery_='sd_bev')


    # select negative pair for satellite
    # 有可能不同样本选出的是同一个负样本
    with torch.no_grad():
        similarity_satellite_drone = satellite_based_satellite_cls @ satellite_based_drone_or_bev_cls.t()
        # similarity_satellite_drone = (satellite_based_satellite_cls @ satellite_based_drone_or_bev_cls.t()) / model.logit_scale

        weights_s2d = F.softmax(similarity_satellite_drone[:, :], dim=1) + 1e-5
        weights_s2d.fill_diagonal_(0)
    if opt.fusion_cls_loss:
        satellite_based_neg4satellite, satellite_based_neg4satellite_label = compose_neg_pair_from_similarity(
            similarity=weights_s2d,
            full_seq_l_other_source=satellite_based_drone_or_bev_full_seq,
            label=satellite_based_label)
    else:
        # 从drone(bev)中选出负样本给satellite
        satellite_based_neg4satellite = compose_neg_pair_from_similarity(similarity=weights_s2d,
                                                                         full_seq_l_other_source=satellite_based_drone_or_bev_full_seq,
                                                                         sample_number=sample_number,
                                                                         label=satellite_based_label, id_names=id_names
                                                                         , epoch=epoch, iteration=iteration,
                                                                         anchor='satellite', gallery_='bev')
    if opt.sd_negative_sample:
        # select negative samples for satellite from sd produced samples
        with torch.no_grad():
            similarity_satellite_satellite_neg = satellite_based_satellite_cls @ satellite_based_satellite_cls_neg.t()

            weights_s2s_neg = F.softmax(similarity_satellite_satellite_neg[:, :], dim=1)# + 1e-5
            # weights_s2d.fill_diagonal_(0)
        satellite_based_neg4satellite_sd = compose_neg_pair_from_similarity(similarity=weights_s2s_neg,sd_data=True,
                                                                         full_seq_l_other_source=satellite_based_satellite_full_seq_neg,
                                                                         sample_number=sample_number,
                                                                         label=satellite_based_label, id_names=id_names
                                                                         , epoch=epoch, iteration=iteration,
                                                                         anchor='satellite', gallery_='sd_satellite')



    # compose full seq: negative + negative
    # follow BLIP@https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/blip_pretrain.py#L181

    if opt.sd_negative_sample:
        satellite_based_satellite_embeds_all = torch.cat(
            [satellite_based_neg4drone, repeat_tensor(satellite_based_satellite_full_seq, sample_number),
             satellite_based_neg4drone_sd, repeat_tensor(satellite_based_satellite_full_seq, sample_number),
             ],
            dim=0)
        satellite_based_drone_embeds_all = torch.cat(
            [repeat_tensor(satellite_based_drone_or_bev_full_seq, sample_number), (satellite_based_neg4satellite),
             repeat_tensor(satellite_based_drone_or_bev_full_seq, sample_number), (satellite_based_neg4satellite_sd)
             ],
            dim=0)
    else:
        satellite_based_satellite_embeds_all = torch.cat(
            [satellite_based_neg4drone, repeat_tensor(satellite_based_satellite_full_seq, sample_number)],
            dim=0)
        satellite_based_drone_embeds_all = torch.cat(
            [repeat_tensor(satellite_based_drone_or_bev_full_seq, sample_number), (satellite_based_neg4satellite)],
            dim=0)

    # forward the negative(and positive) satellite-drone(bev)
    s_fusion_output_neg, _ = model.model_3_fusion(satellite_based_drone_embeds_all,
                                                  satellite_based_satellite_embeds_all)

    bs = satellite_based_satellite_cls.shape[0]
    satellite_based_embeddings = torch.cat([s_fusion_output_pos, s_fusion_output_neg], dim=0)
    satellite_based_itm_label = torch.cat(
        [torch.ones(bs, dtype=torch.long), torch.zeros(s_fusion_output_neg.shape[0], dtype=torch.long)],
        dim=0).to(satellite_based_satellite_cls.device)

    # if random.random() > 0.5:
    # 	satellite_based_embeddings = torch.cat([s_fusion_output_pos[:, 0, :], s_fusion_output_neg[:, 0, :]], dim=0)
    # 	satellite_based_itm_label = torch.cat(
    # 		[torch.ones(bs, dtype=torch.long), torch.zeros(s_fusion_output_neg.shape[0], dtype=torch.long)],
    # 		dim=0).to(satellite_based_satellite_cls.device)
    # 	if opt.fusion_cls_loss:
    # 		# label for classification in the fusion block
    # 		satellite_based_cls_label = torch.cat([
    # 			satellite_based_label, satellite_based_label, satellite_based_neg4satellite_label
    # 		], dim=0)
    # else:
    # 	satellite_based_embeddings = torch.cat([s_fusion_output_neg[:, 0, :], s_fusion_output_pos[:, 0, :]], dim=0)
    # 	satellite_based_itm_label = torch.cat(
    # 		[torch.zeros(s_fusion_output_neg.shape[0], dtype=torch.long), torch.ones(bs, dtype=torch.long)],
    # 		dim=0).to(satellite_based_satellite_cls.device)
    # 	if opt.fusion_cls_loss:
    # 		# label for classification in the fusion block
    # 		satellite_based_cls_label = torch.cat([
    # 			satellite_based_label, satellite_based_neg4satellite_label, satellite_based_label
    # 		], dim=0)

    satellite_based_output = model.itm_head_3(satellite_based_embeddings)
    if opt.matching_label_smooth:
        label_smooth_ce = partial(CrossEntropyLabelSmooth(0.2))
        itm_loss = label_smooth_ce(satellite_based_output, satellite_based_itm_label)
    else:
        itm_loss = F.cross_entropy(satellite_based_output, satellite_based_itm_label)


    ########################################################################################
    # For drone-based sampling matching
    # forward the positive drone(bev)-satellite
    d_fusion_output_pos, _ = model.model_3_fusion(drone_based_drone_or_bev_full_seq,
                                                  drone_based_satellite_full_seq)
    # select negative pair for drone(bev)
    with torch.no_grad():
        similarity_drone_satellite = drone_based_drone_or_bev_cls @ drone_based_satellite_cls.t()
        # similarity_drone_satellite = (drone_based_drone_or_bev_cls @ drone_based_satellite_cls.t())/model.logit_scale
        weights_d2s = F.softmax(similarity_drone_satellite[:, :], dim=1) + 1e-5
        weights_d2s.fill_diagonal_(0)

    if opt.fusion_cls_loss:
        drone_based_neg4drone, _ = compose_neg_pair_from_similarity(similarity=weights_d2s,
                                                                    full_seq_l_other_source=drone_based_satellite_full_seq,
                                                                    label=drone_based_label, id_names=id_names)
    else:
        drone_based_neg4drone = compose_neg_pair_from_similarity(similarity=weights_d2s,
                                                                 full_seq_l_other_source=drone_based_satellite_full_seq,
                                                                 sample_number=sample_number, label=drone_based_label,
                                                                 id_names=id_names
                                                                 , epoch=epoch, iteration=iteration, anchor='bev',
                                                                 gallery_='satellite')
    if opt.sd_negative_sample:
        # from "sd-generated drone" select negative samples for drone
        with torch.no_grad():
            similarity_drone_drone_sd = drone_based_drone_or_bev_cls @ drone_based_drone_or_bev_cls_neg.t()
            # similarity_drone_satellite = (drone_based_drone_or_bev_cls @ drone_based_satellite_cls.t())/model.logit_scale
            weights_d2d_sd = F.softmax(similarity_drone_drone_sd[:, :], dim=1)# + 1e-5
            # do not fill diagonal because itself can be treated as negative samples
            # weights_d2d_sd.fill_diagonal_(0)
        # note: the source of negative samples for "drone" is sd-generated "drone"
        drone_based_neg4drone_sd = compose_neg_pair_from_similarity(similarity=weights_d2d_sd,sd_data=True,
                                                                 full_seq_l_other_source=drone_based_drone_or_bev_full_seq_neg,
                                                                 sample_number=sample_number, label=drone_based_label,
                                                                 id_names=id_names, epoch=epoch, iteration=iteration, anchor='bev',
                                                                 gallery_='sd_bev')


    # select negative pair for satellite
    with torch.no_grad():
        similarity_satellite_drone = drone_based_satellite_cls @ drone_based_drone_or_bev_cls.t()
        # similarity_satellite_drone = (drone_based_satellite_cls @ drone_based_drone_or_bev_cls.t()) / model.logit_scale
        weights_s2d = F.softmax(similarity_satellite_drone[:, :], dim=1) + 1e-5
        weights_s2d.fill_diagonal_(0)
    if opt.fusion_cls_loss:
        drone_based_neg4satellite, drone_based_neg4satellite_label = compose_neg_pair_from_similarity(
            similarity=weights_s2d,
            full_seq_l_other_source=drone_based_drone_or_bev_full_seq,
            label=drone_based_label)
    else:
        # 从drone(bev)中选出负样本给satellite
        drone_based_neg4satellite = compose_neg_pair_from_similarity(similarity=weights_s2d,
                                                                     full_seq_l_other_source=drone_based_drone_or_bev_full_seq,
                                                                     sample_number=sample_number,
                                                                     label=drone_based_label, id_names=id_names
                                                                     , epoch=epoch, iteration=iteration,
                                                                     anchor='satellite', gallery_='bev')
    if opt.sd_negative_sample:
        # select negative samples for satellite from sd-satellite
        with torch.no_grad():
            similarity_satellite_drone = drone_based_satellite_cls @ drone_based_satellite_cls_neg.t()
            weights_s2s_sd = F.softmax(similarity_satellite_drone[:, :], dim=1)# + 1e-5
            # weights_s2d.fill_diagonal_(0)
        drone_based_neg4satellite_sd = compose_neg_pair_from_similarity(similarity=weights_s2s_sd, sd_data=True,
                                                                     full_seq_l_other_source=drone_based_satellite_full_seq_neg,
                                                                     sample_number=sample_number,
                                                                     label=drone_based_label, id_names=id_names
                                                                     , epoch=epoch, iteration=iteration,
                                                                     anchor='satellite', gallery_='satellite_sd')


    # # compose full seq: negative + negative
    # # follow BLIP@https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/blip_pretrain.py#L181
    # drone_based_satellite_embeds_all = torch.cat(
    #     [drone_based_neg4drone, repeat_tensor(drone_based_satellite_full_seq, sample_number)],
    #     dim=0)
    # drone_based_drone_embeds_all = torch.cat(
    #     [repeat_tensor(drone_based_drone_or_bev_full_seq, sample_number), drone_based_neg4satellite],
    #     dim=0)
    if opt.sd_negative_sample:
        drone_based_satellite_embeds_all = torch.cat(
            [drone_based_neg4drone, repeat_tensor(drone_based_satellite_full_seq, sample_number),
             drone_based_neg4drone_sd, repeat_tensor(drone_based_satellite_full_seq, sample_number),
             ],
            dim=0)
        drone_based_drone_embeds_all = torch.cat(
            [repeat_tensor(drone_based_drone_or_bev_full_seq, sample_number), drone_based_neg4satellite,
             repeat_tensor(drone_based_drone_or_bev_full_seq, sample_number), drone_based_neg4satellite_sd
             ],
            dim=0)
    else:
        drone_based_satellite_embeds_all = torch.cat(
            [drone_based_neg4drone, repeat_tensor(drone_based_satellite_full_seq, sample_number)],
            dim=0)
        drone_based_drone_embeds_all = torch.cat(
            [repeat_tensor(drone_based_drone_or_bev_full_seq, sample_number), drone_based_neg4satellite],
            dim=0)
    # forward the negative(and positive) satellite-drone(bev)
    d_fusion_output_neg, _ = model.model_3_fusion(drone_based_drone_embeds_all,
                                                  drone_based_satellite_embeds_all)

    bs = drone_based_satellite_cls.shape[0]
    drone_based_embeddings = torch.cat([d_fusion_output_pos, d_fusion_output_neg], dim=0)
    drone_based_itm_label = torch.cat(
        [torch.ones(bs, dtype=torch.long), torch.zeros(d_fusion_output_neg.shape[0], dtype=torch.long)],
        dim=0).to(satellite_based_satellite_cls.device)

    # if random.random() > 0.5:
    # 	drone_based_embeddings = torch.cat([d_fusion_output_pos[:, 0, :], d_fusion_output_neg[:, 0, :]], dim=0)
    # 	drone_based_itm_label = torch.cat(
    # 		[torch.ones(bs, dtype=torch.long), torch.zeros(d_fusion_output_neg.shape[0], dtype=torch.long)],
    # 		dim=0).to(satellite_based_satellite_cls.device)
    # 	if opt.fusion_cls_loss:
    # 		# label for classification in the fusion block
    # 		drone_based_cls_label = torch.cat([
    # 			drone_based_label, drone_based_label, drone_based_neg4satellite_label
    # 		], dim=0)
    #
    # else:
    # 	drone_based_embeddings = torch.cat([d_fusion_output_neg[:, 0, :], d_fusion_output_pos[:, 0, :]], dim=0)
    # 	drone_based_itm_label = torch.cat(
    # 		[torch.zeros(d_fusion_output_neg.shape[0], dtype=torch.long), torch.ones(bs, dtype=torch.long)],
    # 		dim=0).to(satellite_based_satellite_cls.device)
    # 	if opt.fusion_cls_loss:
    # 		# label for classification in the fusion block
    # 		drone_based_cls_label = torch.cat([
    # 			drone_based_label, drone_based_neg4satellite_label, drone_based_label
    # 		], dim=0)

    drone_based_output = model.itm_head_3(drone_based_embeddings)
    if opt.matching_label_smooth:
        label_smooth_ce = partial(CrossEntropyLabelSmooth(0.2))
        itm_loss += label_smooth_ce(drone_based_output, drone_based_itm_label)
    else:
        itm_loss += F.cross_entropy(drone_based_output, drone_based_itm_label)


    return itm_loss / 2


def has_duplicates(tensor):
    return len(set(tensor.numpy())) < len(tensor)


def set_eval(model, frozen_layer=['logit_scale', 'model_1', 'model_3', 'classifier']):
    frozen_modules = set()
    # 'logit_scale'
    model.model_1.eval()
    model.model_3.eval()
    if not opt.lpn:
        model.classifier.eval()
    # # model.
    # freeze_l = []
    # all_parameters = dict(model.named_parameters())
    # for param_name, _ in all_parameters.items():
    # 	if param_name == 'itm_head_3.3.bias':
    # 		pass
    # 	for frozen_name in frozen_layer:
    # 		if frozen_name in param_name and 'model_3_fusion' not in param_name and 'itm_head_3' not in param_name:
    # 			# print('{} freeze'.format(param_name))
    # 			freeze_l.append(param_name)
    # 			module_name = param_name.split('.')[0]
    # 			module = getattr(model, module_name)
    # 			frozen_modules.add(module)
    # 			if isinstance(module, nn.Parameter):
    # 				module.requires_grad_(False)
    # 			else:
    # 				for param in module.parameters():
    # 					param.requires_grad_(False)
    # # frozen_state = [module.training for module in frozen_modules]
    # training_params = list(set(dict(model.named_parameters()).keys()) - set(freeze_l))
    # write_txt(txt_path,'============training parameters==============')
    # for train_p in training_params:
    # 	write_txt(txt_path, '{}'.format(train_p))
    return frozen_modules


def train_model(model, model_test, criterion, optimizer, scheduler, record_file_path, tb_logger, model_name,
                num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    # # warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
    # warm_iteration = round(dataset_size['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch


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


    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        write_txt(record_file_path, 'Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        write_txt(record_file_path, ('-' * 10))

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
                if opt.two_stage_training:
                    set_eval(model)
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
            for iters, ((data, data2, data3, data4), (d_data, d_data2, d_data3, d_data4)) in enumerate(
                    zip(dataloader, drone_based_dataloader)):
                if opt.rendered_BEV:
                    (data4, data5, data_neg, data5_neg) = data4
                    (d_data4, d_data5, d_data_neg, d_data5_neg) = d_data4
                cur_iter = (epoch * len(dataloader) + iters)
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
                    if opt.sd_negative_sample:
                        inputs_neg, labels_neg = data_neg
                        inputs5_neg, labels5_neg = data5_neg
                        d_inputs_neg, d_labels_neg = d_data_neg
                        d_inputs5_neg, d_labels5_neg = d_data5_neg
                if cur_iter % 1000 == 0:
                    print('add images')
                    if opt.rendered_BEV:
                        # tb_logger_add_images(tb_logger, [inputs, inputs2, inputs3, inputs4, inputs5], cur_iter)
                        if opt.sd_negative_sample:
                            tb_logger_add_images(tb_logger, [inputs, inputs_neg, inputs5, inputs5_neg], cur_iter)
                        else:
                            tb_logger_add_images(tb_logger, [inputs, inputs5], cur_iter)
                    # save_tensor([inputs,inputs5],'/18141169908/hao/itm/0627v2/model_2024-06-28-05:41:35','satellite_bev.jpg')
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
                        if opt.sd_negative_sample:
                            inputs_neg = Variable(inputs_neg.cuda().detach())
                            inputs5_neg = Variable(inputs5_neg.cuda().detach())
                            d_inputs_neg = Variable(d_inputs_neg.cuda().detach())
                            d_inputs5_neg = Variable(d_inputs5_neg.cuda().detach())
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
                            if opt.two_stage_training:
                                with torch.no_grad():
                                    # replace inputs3(drone) with inputs5(BEV)
                                    outputs, outputs3 = model(inputs, None, inputs5)
                                    # outputs, outputs3 = model(tmp.cuda(), None, tmp.cuda())
                                    d_outputs, d_outputs3 = model(d_inputs, None, d_inputs5)
                                    if opt.sd_negative_sample:
                                        outputs_neg, outputs3_neg = model(inputs_neg, None, inputs5_neg)
                                        d_outputs_neg, d_outputs3_neg = model(d_inputs_neg, None, d_inputs5_neg)


                            else:
                                # replace inputs3(drone) with inputs5(BEV)
                                outputs, outputs3 = model(inputs, None, inputs5)
                                d_outputs, d_outputs3 = model(d_inputs, None, d_inputs5)
                        else:
                            if opt.two_stage_training:
                                with torch.no_grad():
                                    outputs, outputs3 = model(inputs, None, inputs3)
                                    d_outputs, d_outputs3 = model(d_inputs, None, d_inputs3)
                            else:
                                outputs, outputs3 = model(inputs, None, inputs3)
                                d_outputs, d_outputs3 = model(d_inputs, None, d_inputs3)
                    elif opt.views == 3:
                        raise Exception('ERROR')
                        if opt.extra_Google:
                            if opt.rendered_BEV:
                                # replace inputs3(drone) with inputs5(BEV)
                                # satellite, street, bev, google
                                outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs5, inputs4)
                                d_outputs, d_outputs2, d_outputs3, d_outputs4 = model(d_inputs, d_inputs2, d_inputs5,
                                                                                      d_inputs4)
                            else:
                                outputs, outputs2, outputs3, outputs4 = model(inputs, inputs2, inputs3, inputs4)
                                d_outputs, d_outputs2, d_outputs3, d_outputs4 = model(d_inputs, d_inputs2, d_inputs3,
                                                                                      d_inputs4)
                        else:
                            if opt.rendered_BEV:
                                # replace inputs3(drone) with inputs5(BEV)
                                outputs, outputs2, outputs3 = model(inputs, inputs2, inputs5)
                                d_outputs, d_outputs2, d_outputs3 = model(d_inputs, d_inputs2, d_inputs5)
                            else:
                                outputs, outputs2, outputs3 = model(inputs, inputs2, inputs3)
                                d_outputs, d_outputs2, d_outputs3 = model(d_inputs, d_inputs2, d_inputs3)

                # return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere or opt.auxiliary  # + 0.5 * criterion_auxiliary()

                if opt.views == 2:
                    # _, preds = torch.max(outputs.data, 1)
                    # _, preds2 = torch.max(outputs2.data, 1)
                    # loss = criterion(outputs, labels) + criterion(outputs2, labels2)
                    if model.itc and not (model.itm):
                        [outputs, cls_feat], [outputs3, cls_feat3] = outputs, outputs3
                        [d_outputs, d_cls_feat], [d_outputs3, d_cls_feat3] = d_outputs, d_outputs3
                        satellite_based_feat_group = [cls_feat, cls_feat3]
                        drone_based_feat_group = [d_cls_feat, d_cls_feat3]

                        itc_loss = get_contrastive_loss_different_sources(
                            [satellite_based_feat_group, drone_based_feat_group], model.logit_scale)
                    elif model.itc and model.itm:
                        # [outputs, cls_feat, full_seq_1], [outputs3, cls_feat3, full_seq_3] = \
                        # 	outputs, outputs3
                        # [d_outputs, d_cls_feat, d_full_seq_1], [d_outputs3, d_cls_feat3, d_full_seq_3] = \
                        # 	d_outputs, d_outputs3

                        [outputs, cls_feat, full_seq_1, input_unflatten], [outputs3, cls_feat3, full_seq_3, input5_unflatten] = \
                            outputs, outputs3

                        [d_outputs, d_cls_feat, d_full_seq_1, _], [d_outputs3, d_cls_feat3, d_full_seq_3, _] = \
                            d_outputs, d_outputs3


                        satellite_based_feat_group = [cls_feat, cls_feat3]
                        satellite_based_seq_feat_group = [full_seq_1, full_seq_3]
                        drone_based_feat_group = [d_cls_feat, d_cls_feat3]
                        drone_based_seq_feat_group = [d_full_seq_1, d_full_seq_3]


                        if opt.sd_negative_sample:
                            [outputs_neg, cls_feat_neg, full_seq_1_neg, _], [outputs3_neg, cls_feat3_neg, full_seq_3_neg, _] = \
                                outputs_neg, outputs3_neg
                            [d_outputs_neg, d_cls_feat_neg, d_full_seq_1_neg, _], [d_outputs3_neg, d_cls_feat3_neg, d_full_seq_3_neg, _] = \
                                d_outputs_neg, d_outputs3_neg
                            satellite_based_feat_group_neg = [cls_feat_neg, cls_feat3_neg]
                            satellite_based_seq_feat_group_neg = [full_seq_1_neg, full_seq_3_neg]
                            drone_based_feat_group_neg = [d_cls_feat_neg, d_cls_feat3_neg]
                            drone_based_seq_feat_group_neg = [d_full_seq_1_neg, d_full_seq_3_neg]


                        if opt.two_stage_training:
                            itc_loss = None
                        else:
                            itc_loss = get_contrastive_loss_different_sources(
                                [satellite_based_feat_group, drone_based_feat_group], model.logit_scale)
                        # satellite, street, drone(bev), google
                        if opt.fusion_cls_loss:
                            pass
                        else:
                            if cur_iter % 1000 == 0:
                                print('add images')
                                if opt.rendered_BEV:
                                    # input_unflatten = unflatten(satellite_based_seq_feat_group[0])
                                    # input5_unflatten = unflatten(satellite_based_seq_feat_group[1])
                                    if input_unflatten!=None:
                                        flatten_input_input_5 = save_feature([input_unflatten, input5_unflatten],
                                                                             '/18141169908/hao/itm/0627v2/model_2024-06-28-05:41:35',
                                                                             'satellite_bev_after_vit.jpg')
                                        tb_logger.add_image('input_images_after_patch_embedding', flatten_input_input_5,
                                                            cur_iter)

                            if opt.sd_negative_sample:
                                itm_loss = get_matching_loss_two_view(
                                    seq_feature_group=[satellite_based_seq_feat_group, drone_based_seq_feat_group],
                                    feat_groups=[satellite_based_feat_group, drone_based_feat_group],
                                    seq_feature_group_neg=[satellite_based_seq_feat_group_neg, drone_based_seq_feat_group_neg],
                                    feat_groups_neg=[satellite_based_feat_group_neg, drone_based_feat_group_neg],
                                    temper_parameter=None,
                                    model=model,
                                    opt=opt,
                                    satellite_based_label=labels,
                                    drone_based_label=d_labels,
                                    sample_number=opt.neg_sample_number,
                                    txt_path=txt_path, class_names=class_names,
                                    epoch=epoch, iteration=iters
                                )
                            else:
                                itm_loss = get_matching_loss_two_view(
                                    seq_feature_group=[satellite_based_seq_feat_group, drone_based_seq_feat_group],
                                    feat_groups=[satellite_based_feat_group, drone_based_feat_group],
                                    temper_parameter=None,
                                    model=model,
                                    opt=opt,
                                    satellite_based_label=labels,
                                    drone_based_label=d_labels,
                                    sample_number=opt.neg_sample_number,
                                    txt_path=txt_path, class_names=class_names,
                                    epoch=epoch, iteration=iters
                                )

                    if not opt.two_stage_training:
                        _, preds = torch.max(outputs.data, 1)
                        # _, preds2 = torch.max(outputs2.data, 1)
                        _, preds3 = torch.max(outputs3.data, 1)

                        _, d_preds = torch.max(d_outputs.data, 1)
                        # _, d_preds2 = torch.max(d_outputs2.data, 1)
                        _, d_preds3 = torch.max(d_outputs3.data, 1)
                    if opt.rendered_BEV:
                        if opt.two_stage_training:
                            pass
                        else:
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
                        if opt.two_stage_training:
                            pass
                        else:
                            loss = criterion(outputs, labels) + criterion(outputs3, labels3)
                            d_loss = criterion(d_outputs, d_labels) + criterion(d_outputs3, d_labels3)

                if model.itc:
                    if opt.vit_itm:
                        # loss = (d_loss + loss) / 2 + opt.loss_lambda * (itc_loss + itm_loss)
                        # loss = (d_loss + loss) / 2 + opt.loss_lambda * (itc_loss + fusion_cls_loss)
                        if opt.two_stage_training:
                            if opt.fusion_cls_loss:
                                loss = opt.loss_lambda * (fusion_cls_loss + itm_loss)
                            else:
                                loss = opt.loss_lambda * (itm_loss)
                        else:
                            loss = (d_loss + loss) / 2 + opt.loss_lambda * (itc_loss + 0.5 * fusion_cls_loss + itm_loss)
                    else:
                        # follow tingyu to average
                        loss = (d_loss + loss) / 2 + opt.loss_lambda * itc_loss

                else:
                    loss = (d_loss + loss) / 2
                # backward + optimize only if in training phase
                # if epoch < opt.warm_epoch and phase == 'train':
                # 	warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                # 	loss *= warm_up

                if phase == 'train':
                    if fp16:  # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    # if iters ==0:
                    # 	a = dict(model.named_parameters())['model_3_fusion.blocks.1.c_attn.kv.weight'].data
                    # else:
                    # 	b = dict(model.named_parameters())['model_3_fusion.blocks.1.c_attn.kv.weight'].data
                    # 	flag_equal = torch.equal(a,b)
                    # 	lr_l = scheduler.get_lr()
                    # 	lr = lr_l[1]
                    # 	# torch.equal(dict(model.named_parameters())['model_3_fusion.blocks.1.c_attn.kv.weight'].grad,
                    # 	# 			torch.zeros_like(dict(model.named_parameters())[
                    # 	# 								 'model_3_fusion.blocks.1.c_attn.kv.weight'].grad))
                    # 	write_txt(txt_path,'model_3_fusion.blocks.1.c_attn.kv.weight change {}, lr {}, grad {} in optimizer:{}'.format(not(flag_equal),lr,dict(model.named_parameters())['model_3_fusion.blocks.1.c_attn.kv.weight'].grad, id(dict(model.named_parameters())['model_3_fusion.blocks.1.c_attn.kv.weight']) in set(map(id, optimizer.param_groups[1]['params']))))

                    optimizer.step()
                    # TODO: hao: iteration -level update
                    # scheduler.step()
                    lr_l = scheduler.get_lr()
                    for i, lr in enumerate(lr_l):
                        tb_logger.add_scalar('lr/lr_{}'.format(i), lr, cur_iter)
                    ##########
                    if opt.moving_avg < 1.0:
                        update_average(model_test, model, opt.moving_avg)

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                # if opt.vit_itm:
                #     running_loss_itm += itm_loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                if not opt.two_stage_training:
                    running_corrects += float(torch.sum(preds == labels.data))
                    try:
                        running_corrects2 += float(torch.sum(preds2 == labels2.data))
                    except:
                        pass
                if cur_iter % 500 == 0:
                    tb_logger.add_scalar('iter/loss', running_loss / (now_batch_size * (cur_iter + 1)), cur_iter)
                    # tb_logger.add_scalar('iter/loss_itm', running_loss_itm/ (now_batch_size*(cur_iter+1)), cur_iter)
                    # tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                    if opt.vit_itc:
                        if opt.two_stage_training:
                            pass
                        else:
                            tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                    if opt.vit_itm:
                        tb_logger.add_scalar('loss/loss_itm', itm_loss, cur_iter)
                        if opt.fusion_cls_loss:
                            tb_logger.add_scalar('loss/fusion_cls_loss', fusion_cls_loss, cur_iter)
                    tb_logger.add_scalar('iter/Satellite_Acc', running_corrects / (now_batch_size * (cur_iter + 1)),
                                         cur_iter)
                    tb_logger.add_scalar('iter/Street_Acc', running_corrects2 / (now_batch_size * (cur_iter + 1)),
                                         cur_iter)
                    tb_logger.add_scalar('iter/Drone_Acc', running_corrects3 / (now_batch_size * (cur_iter + 1)),
                                         cur_iter)
                if opt.views == 3 or opt.views == 2:
                    if not opt.two_stage_training:
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
                write_txt(record_file_path, (
                    '{} Loss: {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                      epoch_acc3)))
                tb_logger.add_scalar('loss/loss_total', running_loss / dataset_size['satellite'], cur_iter)
                if opt.vit_itc:
                    if opt.two_stage_training:
                        pass
                    else:
                        tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                if opt.vit_itm:
                    tb_logger.add_scalar('loss/loss_itm', itm_loss, cur_iter)
                    if opt.fusion_cls_loss:
                        tb_logger.add_scalar('loss/fusion_cls_loss', fusion_cls_loss, cur_iter)
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
                write_txt(record_file_path,
                          ('{} Loss: {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f}'.format(phase,
                                                                                                                epoch_loss,
                                                                                                                epoch_acc,
                                                                                                                epoch_acc2,
                                                                                                                epoch_acc3)))
                tb_logger.add_scalar('loss/loss_total', running_loss / dataset_size['satellite'], cur_iter)
                if opt.vit_itc:
                    tb_logger.add_scalar('loss/loss_itc', itc_loss, cur_iter)
                if opt.vit_itm:
                    tb_logger.add_scalar('loss/loss_itm', itm_loss, cur_iter)
                tb_logger.add_scalar('Satellite_Acc', running_corrects / dataset_size['satellite'], cur_iter)
                tb_logger.add_scalar('Street_Acc', running_corrects2 / dataset_size['satellite'], cur_iter)
                tb_logger.add_scalar('Drone_Acc', running_corrects3 / dataset_size['satellite'], cur_iter)
            # tb_logger.add_scalar('epoch/loss', running_loss / dataset_size, cur_iter)
            # tb_logger.add_scalar('epoch/Satellite_Acc', running_corrects / dataset_size, cur_iter)
            # tb_logger.add_scalar('epoch/Street_Acc', running_corrects2 / dataset_size, cur_iter)
            # tb_logger.add_scalar('epoch/Drone_Acc', running_corrects3 / dataset_size, cur_iter)

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            # # TODO: epoch-level update lr
            if phase == 'train':
                scheduler.step()


            # last_model_wts = model.state_dict()
            # if epoch % 20 == 19:
            if (epoch+1) % (int(opt.save_epoch)) == 0:
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


def load_vit_contrastive(model, model_path):
    param_dict = torch.load(model_path, map_location='cpu')
    model_1_dict = {}
    model_3_dict = {}
    other_dict = {}
    for k, v in param_dict.items():
        if 'model_1' in k:
            model_1_dict[k] = v
        elif 'model_3' in k:
            model_3_dict[k] = v
        elif 'logit_scale' == k:
            other_dict[k] = v
        elif 'classifier' in k:
            other_dict[k] = v
        else:
            raise Exception('Error: {} is not in model'.format(k))

    # k_3_l = list(model_3_dict.keys())
    # for idx, k_ in enumerate(k_3_l):
    #     k_3_l[idx] = k_.replace('model_3', 'model_1')
    # for k, v in model_1_dict.items():
    #     if k in k_3_l:
    #         continue
    #     else:
    #         model_3_dict[k.replace('model_1', 'model_3')] = v
    model_1_dict.update(model_3_dict)
    model_1_dict.update(other_dict)
    mssg = model.load_state_dict(model_1_dict, strict=False)
    mkeys = mssg.missing_keys
    mkeys.insert(0, 'missing key:')
    mis_load = mssg.unexpected_keys
    mis_load.insert(0, 'mismatching key:')
    mkeys.extend(mis_load)
    return mkeys, mssg.missing_keys


def set_freeze(model, frozen_layer=['logit_scale', 'model_1', 'model_3', 'classifier']):
    frozen_modules = set()
    # model.
    freeze_l = []
    all_parameters = dict(model.named_parameters())
    for param_name, _ in all_parameters.items():
        for frozen_name in frozen_layer:
            if frozen_name in param_name:
                if ('model_3_fusion' in param_name) or ('itm_head_3' in param_name):
                    continue
                # print('{} freeze'.format(param_name))
                freeze_l.append(param_name)
                module_name = param_name.split('.')[0]
                module = getattr(model, module_name)
                frozen_modules.add(module)
                if isinstance(module, nn.Parameter):
                    module.requires_grad_(False)
                else:
                    for param in module.parameters():
                        param.requires_grad_(False)
    # frozen_state = [module.training for module in frozen_modules]
    training_params = []
    for k,v in model.named_parameters():
        if v.requires_grad:
            training_params.append(k)
    # training_params = list(set(dict(model.named_parameters()).keys()) - set(freeze_l))
    write_txt(txt_path, '============training parameters==============')
    assert (set(dict(model.named_parameters()).keys()) - set(freeze_l)) == set(training_params)
    for train_p in training_params:
        write_txt(txt_path, '{}'.format(train_p))

    write_txt(txt_path, '==========================')
    return training_params


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere

if opt.views == 2:
    model = two_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                         share_weight=opt.share, circle=return_feature, ibn=opt.ibn, vit=opt.vit, itc=opt.vit_itc,
                         itm=opt.vit_itm, itm_share=opt.vit_itm_share, fusion_cls_loss=opt.fusion_cls_loss, lpn=opt.lpn,
                         two_stage_training=opt.two_stage_training,backbone_name = opt.backbone_name,)
elif opt.views == 3:
    # model = three_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
    #                        share_weight=opt.share, circle=return_feature)
    model = three_view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                           share_weight=opt.share, circle=return_feature, ibn=opt.ibn, vit=opt.vit, itc=opt.vit_itc,
                           itm=opt.vit_itm, itm_share=opt.vit_itm_share)
mssg_l = []
mssg_l.append('backbone-name:{}:{}'.format(opt.backbone_name,opt.first_stage_weight_path))
if opt.backbone_name=='vit-small':
    pass
elif opt.backbone_name=='swin-tiny':
    model.model_path = opt.first_stage_weight_path
elif opt.backbone_name=='swin-base':
    model.model_path = opt.first_stage_weight_path
elif opt.backbone_name=='vit-base':
    model.model_path = opt.first_stage_weight_path
else:
    raise NotImplementedError
load_params_info, missing_k_l = load_vit_contrastive(model, model.model_path)
mssg_l.extend(load_params_info)
mssg_l.append('============================================')
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
    num_epochs = 120
# num_epochs = 360
# num_epochs = 1

if opt.epoch != 0:
    num_epochs = opt.epoch

mssg_l.append('==================optimizer=======================')
optimizer_grouped_parameters = [
    # base parameters(0)
    {"params": [], "weight_decay": opt.wd, "lr": 0.1 * opt.lr},
    # classification head for the instance loss(1)
    {"params": [], "weight_decay": opt.wd, "lr": opt.lr_instance},
    # {"params": [], "weight_decay": opt.wd, "lr": opt.lr},
]

if opt.vit_itm or opt.vit_itc:
    if (not opt.vit_itm) and (opt.vit_itc):  # lr_itm
        mssg_l.append('3rd optimizer for itc only (logit_scale) ')
        # itc only
        # logit_scale(2)
        optimizer_grouped_parameters.append({"params": [], "weight_decay": opt.wd, "lr": opt.lr})
    else:
        mssg_l.append('3rd optimizer for fusion module w/o weight decay ')
        # fusion module (no decay)(2)
        optimizer_grouped_parameters.append({"params": [], "weight_decay": 0.0, "lr": opt.lr_itm})
        mssg_l.append('4th optimizer for fusion module w weight decay ')
        # fusion module (with decay)(3)
        optimizer_grouped_parameters.append({"params": [], "weight_decay": opt.wd, "lr": opt.lr_itm})
        if opt.two_stage_training:
            pass
        else:
            mssg_l.append('5th optimizer for finetune itc logit_scale ')
            # logit_scale(4)
            optimizer_grouped_parameters.append({"params": [], "weight_decay": opt.wd, "lr": opt.lr})

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
            "norm3.weight", }


currentTime = time.localtime()
model_name = 'model_{}'.format(time.strftime("%Y-%m-%d-%H:%M:%S", currentTime))
# opt.name = model_name
current_file = __file__
absolute_path = os.path.dirname(os.path.abspath(current_file))
dir_name = os.path.join(absolute_path, model_name, name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    txt_path = os.path.join(dir_name, 'training_record.txt')

if opt.two_stage_training:
    l_lr_l = set_freeze(model)
    del (missing_k_l[0])
    if opt.lpn:
        # idx_l = []
        missing_k_l_ = []
        for idx, param_name in enumerate(missing_k_l):
            if 'classifier' not in param_name:
                if 'logit' not in param_name:
                    missing_k_l_.append(param_name)

        missing_k_l = missing_k_l_
    del (missing_k_l[-1])
    assert set(l_lr_l) == set(missing_k_l)
    to_optimizer_l = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if any(nd in n for nd in no_decay):
            to_optimizer_l.append(n)
            # no decay
            mssg_l.append('parameter: {} wd = 0'.format(n))
            optimizer_grouped_parameters[2]['params'].append(p)
        else:
            to_optimizer_l.append(n)
            mssg_l.append('parameter: {} wd = {}'.format(n, opt.wd))
            # with decay
            optimizer_grouped_parameters[3]['params'].append(p)
    assert set(l_lr_l) == set(to_optimizer_l)
    assert (len(optimizer_grouped_parameters[3]['params']) + len(optimizer_grouped_parameters[2]['params'])) == len(
        l_lr_l)

if opt.two_stage_training:
    assert len(optimizer_grouped_parameters) == 4
    assert len(optimizer_grouped_parameters[0]['params']) == 0
    assert len(optimizer_grouped_parameters[1]['params']) == 0
    # del base parameter
    del optimizer_grouped_parameters[0]
    # del classification head
    del optimizer_grouped_parameters[0]
    assert len(optimizer_grouped_parameters) == 2
# set_eval(model)
if opt.optimizer == 'SGD':
    optimizer_ft = optim.SGD(optimizer_grouped_parameters, momentum=0.9, nesterov=True)
elif opt.optimizer == 'AdamW':
    from torch.optim import AdamW

    optimizer_ft = AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.98))
else:
    raise Exception('Not support {} optimizer'.format(opt.optimizer))

# Decay LR by a factor of 0.1 every 40 epochs
if opt.scheduler == 'steplr':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)

elif opt.scheduler == 'linearlr_warmup':

    from torch.optim.lr_scheduler import LambdaLR

    opt.num_training_steps = num_epochs
    # opt.num_warmup_steps = int(0.1 * opt.num_training_steps)
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


    # def lr_lambda(current_step: int):
    #     # 自定义函数
    #     if current_step < opt.num_warmup_steps:
    #         return float(current_step) / float(max(1, opt.num_warmup_steps))
    #     return max(
    #         0.0, float(opt.num_training_steps - current_step)
    #              / float(max(1, opt.num_training_steps - opt.num_warmup_steps))
    #     )

    # def lr_lambda(current_step):
    #     if current_step < opt.num_warmup_steps:
    #         return float(current_step) / float(max(1, opt.num_warmup_steps))
    #     else:
    #         return opt.gamma ** (current_step // opt.step_size)

    exp_lr_scheduler = LambdaLR(optimizer_ft, lr_lambda, last_epoch=-1)

elif opt.scheduler == 'linearlr_warmup_34_epoch':

    from torch.optim.lr_scheduler import LambdaLR

    opt.num_training_steps = num_epochs
    # opt.num_warmup_steps = int(0.1 * opt.num_training_steps)
    # opt.num_warmup_steps = int(34)
    opt.num_warmup_steps = int(opt.num_warmup_steps)
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
            # tt = 0.7
        elif current_step < opt.num_warmup_steps * 7:
            tt = 0.5
        else:
            tt = 0.2

        return tt * max(
            0.0,
            float(opt.num_training_steps - current_step) /
            float(max(1, opt.num_training_steps - opt.num_warmup_steps))
        )


    # def lr_lambda(current_step: int):
    #     # 自定义函数
    #     if current_step < opt.num_warmup_steps:
    #         return float(current_step) / float(max(1, opt.num_warmup_steps))
    #     return max(
    #         0.0, float(opt.num_training_steps - current_step)
    #              / float(max(1, opt.num_training_steps - opt.num_warmup_steps))
    #     )

    # def lr_lambda(current_step):
    #     if current_step < opt.num_warmup_steps:
    #         return float(current_step) / float(max(1, opt.num_warmup_steps))
    #     else:
    #         return opt.gamma ** (current_step // opt.step_size)

    exp_lr_scheduler = LambdaLR(optimizer_ft, lr_lambda, last_epoch=-1)

elif opt.scheduler == 'steplr_warmup':
    from torch.optim.lr_scheduler import LambdaLR

    opt.num_training_steps = num_epochs
    opt.num_warmup_steps = int(0.1 * opt.num_training_steps)
    opt.step_size = opt.gamma * opt.num_training_steps
    mssg_l.append('total epochs:{}'.format(opt.num_training_steps))
    mssg_l.append('warmup epochs:{}'.format(opt.num_warmup_steps))
    mssg_l.append('decay at {} epochs, decay to {} lr'.format(opt.step_size, opt.gamma))


    def lr_lambda(step):
        if step < opt.num_warmup_steps:
            return (step + 1) / (opt.num_warmup_steps + 1)
        else:
            decay_step = int(opt.step_size)
            if step >= decay_step:
                return opt.gamma
            else:
                return 1.0


    exp_lr_scheduler = LambdaLR(optimizer_ft, lr_lambda, last_epoch=-1)

elif opt.scheduler == 'cosine':
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_ft, T_max=num_epochs, eta_min=1e-6)
else:
    raise Exception('Not support {} scheduler'.format(opt.scheduler))

if opt.two_stage_training:
    assert (len(l_lr_l) == (len(optimizer_ft.param_groups[0]['params']) + len(optimizer_ft.param_groups[1]['params'])))
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.

if not opt.resume:
    # if not os.path.isdir(dir_name):
    # 	os.makedirs(dir_name)
    # txt_path = os.path.join(dir_name, 'training_record.txt')
    tb_path = os.path.join(dir_name, 'tb_logger')
    if not os.path.exists(tb_path):
        os.mkdir(tb_path)
    tb_logger = init_tb_logger(tb_path)
    write_txt(txt_path, 'BEV:{}'.format(opt.rendered_BEV))

    # if opt.two_stage_training:
    #     write_txt(txt_path, 'backbone:{}'.format('ResNet50-IBN'))
    if opt.ibn:
        write_txt(txt_path, 'backbone:{}'.format('ResNet50-IBN'))
    elif opt.vit:
        write_txt(txt_path, 'backbone:{}'.format('vit-small'))
    else:
        write_txt(txt_path, 'backbone:{}'.format('ResNet50'))
    write_txt(txt_path, dataset_size)

    write_txt(txt_path, 'Use Image text contrastive loss:{}'.format(opt.vit_itc))
    write_txt(txt_path, 'Use Image text matching loss:{}'.format(opt.vit_itm))
    write_txt(txt_path, 'Use Image text matching loss with a shared head:{}'.format(opt.vit_itm_share))
    write_txt(txt_path,
              'Freeze backbone and contrastive loss. only train fusion module:{}'.format(opt.two_stage_training))
    write_txt(txt_path, 'lambda for loss:{}'.format(opt.loss_lambda))
    for mssg in mssg_l:
        write_txt(txt_path, '{}'.format(mssg))

    write_txt(txt_path, 'optimizer {}'.format(opt.optimizer))
    write_txt(txt_path, 'scheduler {}'.format(opt.scheduler))
    write_txt(txt_path, 'learning rate {}'.format(opt.lr))

    write_txt(txt_path, model)

    # record every run
    copyfile('train_bev_paired_fsra.py', dir_name + '/train_bev_paired_fsra.py')
    # copyfile('./model.py', dir_name + '/model.py')
    copyfile(os.path.join(absolute_path, 'model.py'), dir_name + '/model.py')
    copyfile(os.path.join(absolute_path, 'train.sh'), dir_name + '/train.sh')
    copyfile(os.path.join(absolute_path, 'backbone', 'vit_pytorch.py'), dir_name + '/vit_pytorch.py')
    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

criterion = nn.CrossEntropyLoss()


model = train_model(model, model_test, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=num_epochs, tb_logger=tb_logger, record_file_path=txt_path, model_name=model_name)
