import os
import torch
import yaml
from model import two_view_net, three_view_net
import math
import time
from ptflops import get_model_complexity_info
from typing import Tuple, Union
import random
import string
import gc
from tqdm import tqdm



def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, model_name,dirname, epoch_label):
    if not os.path.isdir('./{}/'.format(model_name)+dirname):
        os.mkdir('./{}/'.format(model_name)+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    # save_path = os.path.join('./model',dirname,save_filename)
    save_path = os.path.join('./{}'.format(model_name),dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
# def load_network(name, opt,model_dir_name):
#     # Load config
#     # dirname = os.path.join('./model',name)
#     dirname = os.path.join('./{}'.format(model_dir_name),name)
#     # dirname = os.path.join('./model_retrain',name)
#     last_model_name = os.path.basename(get_model_list(dirname, 'net'))
#     epoch = last_model_name.split('_')[1]
#     epoch = epoch.split('.')[0]
#     # epoch = '079'
#     if not epoch=='last':
#        epoch = int(epoch)
#     config_path = os.path.join(dirname,'opts.yaml')
#     with open(config_path, 'r') as stream:
#         # config = yaml.load(stream)
#         config = yaml.safe_load(stream)
#
#     opt.name = config['name']
#     opt.data_dir = config['data_dir']
#     opt.train_all = config['train_all']
#     opt.droprate = config['droprate']
#     opt.color_jitter = config['color_jitter']
#     opt.batchsize = config['batchsize']
#     opt.h = config['h']
#     opt.w = config['w']
#     opt.share = config['share']
#     opt.stride = config['stride']
#     if 'pool' in config:
#         opt.pool = config['pool']
#     if 'h' in config:
#         opt.h = config['h']
#         opt.w = config['w']
#     if 'gpu_ids' in config:
#         opt.gpu_ids = config['gpu_ids']
#     opt.erasing_p = config['erasing_p']
#     opt.lr = config['lr']
#     opt.nclasses = config['nclasses']
#     opt.erasing_p = config['erasing_p']
#     opt.use_dense = config['use_dense']
#     opt.fp16 = config['fp16']
#     opt.views = config['views']
#     opt.ibn = config['ibn']
#     opt.vit = config['vit']
#     opt.itc = config['vit_itc']
#     opt.itm = config['vit_itm']
#     opt.itm_share = config['vit_itm_share']
#     # write_txt(txt_path,'BEV:{}'.format(opt.rendered_BEV))
#     #     write_txt(txt_path,'auxiliary loss:{}'.format(opt.auxiliary))
#     opt.rendered_BEV = config['rendered_BEV']
#     opt.auxiliary = config['auxiliary']
#     opt.fusion_cls_loss = config['fusion_cls_loss']
#     opt.lpn = config['lpn']
#     opt.block = 5
#     opt.two_stage_training = config['two_stage_training']
#
#     if opt.use_dense:
#         model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
#     if opt.PCB:
#         model = PCB(opt.nclasses)
#
#     if opt.views == 2:
#         # model = two_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, ibn=opt.ibn,vit=opt.vit,itc=opt.itc)
#         model = two_view_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
#                            share_weight=opt.share, circle=False,ibn=opt.ibn,vit=opt.vit,itc = opt.itc,itm=opt.itm,itm_share=opt.itm_share,fusion_cls_loss=opt.fusion_cls_loss,
#                              lpn=opt.lpn,two_stage_training=opt.two_stage_training)
#     elif opt.views == 3:
#         model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, ibn=opt.ibn,vit=opt.vit,itc=opt.itc)
#
#     if 'use_vgg16' in config:
#         opt.use_vgg16 = config['use_vgg16']
#         if opt.views == 2:
#             model = two_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16)
#         elif opt.views == 3:
#             model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16)
#
#
#     # load model
#     if isinstance(epoch, int):
#         save_filename = 'net_%03d.pth'% epoch
#     else:
#         save_filename = 'net_%s.pth'% epoch
#
#     # save_path = os.path.join('./model',name,save_filename)
#     save_path = os.path.join('./{}'.format(model_dir_name),name,save_filename)
#     # save_path = os.path.join('./model_retrain',name,save_filename)
#     print('Load the model from %s'%save_path)
#     mssg = 'Load the model from %s'%save_path
#     network = model
#     network.load_state_dict(torch.load(save_path))
#     return network, opt, epoch, mssg

def load_network(name, opt,model_dir_name, model_dir_name_epoch):
    # Load config
    # dirname = os.path.join('./model',name)
    dirname = os.path.join('./{}'.format(model_dir_name),name, model_dir_name_epoch)
    # dirname = os.path.join('./model_retrain',name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    # epoch = '079'
    if not epoch=='last':
       epoch = int(epoch)
    config_path = os.path.join(dirname,'opts.yaml')
    with open(config_path, 'r') as stream:
        # config = yaml.load(stream)
        config = yaml.safe_load(stream)

    opt.name = config['name']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    opt.stride = config['stride']
    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.use_dense = config['use_dense']
    opt.fp16 = config['fp16']
    opt.views = config['views']
    opt.ibn = config['ibn']
    opt.vit = config['vit']
    opt.itc = config['vit_itc']
    opt.itm = config['vit_itm']
    opt.itm_share = config['vit_itm_share']
    # write_txt(txt_path,'BEV:{}'.format(opt.rendered_BEV))
    #     write_txt(txt_path,'auxiliary loss:{}'.format(opt.auxiliary))
    opt.rendered_BEV = config['rendered_BEV']
    # opt.auxiliary = config['auxiliary']
    opt.fusion_cls_loss = config['fusion_cls_loss']
    opt.lpn = config['lpn']
    opt.block = 5
    opt.two_stage_training = config['two_stage_training']
    opt.backbone_name = config['backbone_name']

    if opt.use_dense:
        model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    if opt.PCB:
        model = PCB(opt.nclasses)

    if opt.views == 2:
        # model = two_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, ibn=opt.ibn,vit=opt.vit,itc=opt.itc)
        model = two_view_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                           share_weight=opt.share, circle=False,ibn=opt.ibn,vit=opt.vit,itc = opt.itc,itm=opt.itm,itm_share=opt.itm_share,fusion_cls_loss=opt.fusion_cls_loss,
                             lpn=opt.lpn,two_stage_training=opt.two_stage_training,backbone_name=config['backbone_name'])
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, ibn=opt.ibn,vit=opt.vit,itc=opt.itc)

    if 'use_vgg16' in config:
        opt.use_vgg16 = config['use_vgg16']
        if opt.views == 2:
            model = two_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16)
        elif opt.views == 3:
            model = three_view_net(opt.nclasses, opt.droprate, stride = opt.stride, pool = opt.pool, share_weight = opt.share, VGG16 = opt.use_vgg16)


    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth'% epoch
    else:
        save_filename = 'net_%s.pth'% epoch

    # save_path = os.path.join('./model',name,save_filename)
    save_path = os.path.join('./{}'.format(model_dir_name),name,model_dir_name_epoch,save_filename)
    # save_path = os.path.join('./model_retrain',name,save_filename)
    print('Load the model from %s'%save_path)
    mssg = 'Load the model from %s'%save_path
    network = model
    
    state = torch.load(save_path)
    drop_keys = [k for k in state if k.endswith('logit_scale')]  # 可能带 module. 前缀
    for k in drop_keys:
        print(f"[skip] {k}")
        state.pop(k)
    
    network.load_state_dict(state)
    return network, opt, epoch, mssg

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)

# --------------------------- 1. 参数量 --------------------------- #
def count_params_mb(model: torch.nn.Module) -> float:
    """返回模型参数量，单位 MB（按 FP32 = 4 字节）"""
    numel = sum(p.numel() for p in model.parameters())
    return numel * 4 / 1024**2          # 转 MB


# --------------------------- 2. 计算 FLOPs ----------------------- #
def calc_flops_gpu(model: torch.nn.Module,
                   image: torch.Tensor) -> float:
    device = next(model.parameters()).device              # 获取模型所在 GPU
    c, h, w = image.shape[1:]

    def input_constructor(_):
        dummy = torch.randn(1, c, h, w, device=device)
        return dict(x1=None, _=None, x3=dummy)

    macs, _ = get_model_complexity_info(
        model, (c, h, w),
        input_constructor=input_constructor,
        print_per_layer_stat=False, verbose=False,
        as_strings=False,
    )
    return macs * 2 / 1e9

def calc_flops_gpu_itm(model: torch.nn.Module, frame_query_feature_full_seq: torch.Tensor, gallery_feature_top_k: torch.Tensor) -> float:
    """
    计算部分模型的 GFLOPs（model.itm_head_3 和 model.model_3_fusion）
    输入张量是 (B, N, D)，其中 B、N、D 可以根据实际情况调整
    """
    # 获取模型所在设备
    device = next(model.parameters()).device
    B, N, D = frame_query_feature_full_seq.shape  # 假设 frame_query_feature_full_seq 和 gallery_feature_top_k 形状相同

    # 构造 dummy 输入
    def input_constructor(_):
        # 构造虚拟输入，B, N, D 的维度
        dummy1 = torch.randn(1, N, D, device=device)
        dummy2 = torch.randn(1, N, D, device=device)
        return dict(x1=dummy1, x2=dummy2)

    # 计算 model.model_3_fusion 的 FLOPs
    fusion_macs, _ = get_model_complexity_info(
        model.model_3_fusion,         # 只统计 model.model_3_fusion
        (N, D),                      # 输入维度 (N, D)
        input_constructor=input_constructor,
        print_per_layer_stat=False, verbose=False,
        as_strings=False
    )
    feat = model.model_3_fusion(frame_query_feature_full_seq,
                            gallery_feature_top_k)[0]
    B, N, D = feat.shape
    # 构造 dummy 输入
    def input_constructor2(_):
        # 构造虚拟输入，B, N, D 的维度
        dummy = torch.randn(1, N, D, device=device)
        return dict(input=dummy)
    # 计算 model.itm_head_3 的 FLOPs
    head_macs, _ = get_model_complexity_info(
        model.itm_head_3,            # 只统计 itm_head_3
        (N, D),                      # 输入维度 (N, D)
        input_constructor=input_constructor2,
        print_per_layer_stat=False, verbose=False,
        as_strings=False
    )

    # 合并两个模块的 FLOPs
    total_macs = fusion_macs + head_macs
    return total_macs * 2 / 1e9  # 转换为 GFLOPs

# --------------------------- 3. 前向时间 ------------------------- #
def measure_forward_time(model: torch.nn.Module,
                         image: torch.Tensor,
                         repeat: int = 50,
                         warmup: int = 10) -> float:
    """
    多次前向取平均，返回秒。会自动检测设备。
    """
    device = next(model.parameters()).device
    image  = image.to(device)
    model.eval()
    
    # warm‑up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(None, None, image)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(None, None, image)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.perf_counter()
    
    return (t1 - t0) / repeat

def measure_forward_time_itm(model: torch.nn.Module,
                         frame_query_feature_full_seq: torch.Tensor,
                         gallery_feature_top_k: torch.Tensor,
                         repeat: int = 50,
                         warmup: int = 10) -> float:
    """
    计算模型前向传递时间，返回秒。
    自动调整 frame_query_feature_full_seq 和 gallery_feature_top_k 的批次大小为 1。
    """
    device = next(model.parameters()).device
    
    # 确保 frame_query_feature_full_seq 和 gallery_feature_top_k 的 batch_size = 1
    frame_query_feature_full_seq = frame_query_feature_full_seq[0:1,...]
    gallery_feature_top_k = gallery_feature_top_k[0:1,...]

    model.eval()

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.itm_head_3(
                        model.model_3_fusion(
                            frame_query_feature_full_seq,
                            gallery_feature_top_k)[0]
                    )

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model.itm_head_3(
                        model.model_3_fusion(
                            frame_query_feature_full_seq,
                            gallery_feature_top_k)[0]
                    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.perf_counter()

    return (t1 - t0) / repeat

def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def save_batch_features(ff: torch.Tensor, ff_full_seq: torch.Tensor, idx: int, out_dir: str):
    """
    保存每一批的 ff 和 ff_full_seq
    `idx` 作为文件编号，保证每次都保存独立的文件
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 定义保存路径
    ff_path = os.path.join(out_dir, f"{idx:06d}_ff.pt")
    ff_full_seq_path = os.path.join(out_dir, f"{idx:06d}_ff_full_seq.pt")
    
    # 保存张量
    torch.save(ff.cpu(), ff_path)
    torch.save(ff_full_seq.cpu(), ff_full_seq_path)

def save_batch_features_ordered(ff: torch.Tensor, ff_full_seq: torch.Tensor, idx: int, out_dir: str, name_l: list):
    """
    保存每一批的 ff 和 ff_full_seq
    `idx` 作为文件编号，保证每次都保存独立的文件
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(ff.shape[0]):
        # 定义保存路径
        ff_path = os.path.join(out_dir, f"{int(name_l[(idx+i)]):011d}_ff.pt")
        ff_full_seq_path = os.path.join(out_dir, f"{int(name_l[(idx+i)]):011d}_ff_full_seq.pt")
        
        # 保存张量
        torch.save(ff[i].cpu(), ff_path)
        torch.save(ff_full_seq[i].cpu(), ff_full_seq_path)


def save_batch_features_ff(ff_l: list, out_dir: str, to_save_name_l: list, name:str):
    """
    保存每一批的 ff 和 ff_full_seq
    `idx` 作为文件编号，保证每次都保存独立的文件
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for save_name, ff in zip(to_save_name_l, ff_l):
        # 定义保存路径
        ff_path = os.path.join(out_dir, f"{int(save_name):011d}_{name}.pt")
        # ff_path = os.path.join(out_dir, f"{idx:06d}_ff.pt")
        # ff_full_seq_path = os.path.join(out_dir, f"{idx:06d}_ff_full_seq.pt")
        
        # 保存张量
        torch.save(ff.cpu(), ff_path)
    # torch.save(ff_full_seq.cpu(), ff_full_seq_path)


def load_all_features(out_dir: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    读取保存的所有 ff 和 ff_full_seq，拼接成最终结果
    """
    ff_list = []
    ff_full_seq_list = []
    
    files_ff = sorted(os.listdir(out_dir))  # 获取所有文件，并按顺序读取
    for file in files_ff:
        # 如果文件名是 bytes 类型，先进行解码为 str
        if isinstance(file, bytes):
            file = file.decode('utf-8')  # 解码为 utf-8 字符串
        if "_ff.pt" in file:
            ff_path = os.path.join(out_dir, file)
            ff_full_seq_path = os.path.join(out_dir, file.replace("_ff", "_ff_full_seq"))
            
            ff = torch.load(ff_path, map_location=device)
            ff_full_seq = torch.load(ff_full_seq_path, map_location=device)
            
            ff_list.append(ff)
            ff_full_seq_list.append(ff_full_seq)

    # 拼接成最终结果
    features = torch.cat(ff_list, dim=0)
    features_full_seq = torch.cat(ff_full_seq_list, dim=0)

    return features, features_full_seq

def load_all_features_ff(out_dir: str, device: torch.device, name: str):
    """
    读取保存的所有 ff 和 ff_full_seq，拼接成最终结果
    """
    ff_list = []
    # ff_full_seq_list = []
    
    files_ff = sorted(os.listdir(out_dir))  # 获取所有文件，并按顺序读取
    for file in tqdm(files_ff):
        # 如果文件名是 bytes 类型，先进行解码为 str
        if isinstance(file, bytes):
            file = file.decode('utf-8')  # 解码为 utf-8 字符串
        if "_ff.pt" in file:
            ff_path = os.path.join(out_dir, file)
            # ff_full_seq_path = os.path.join(out_dir, file.replace("_ff", "_ff_full_seq"))
            
            ff = torch.load(ff_path, map_location=device)
            # ff_full_seq = torch.load(ff_full_seq_path, map_location=device)
            
            if 'bev' in name:
                # average
                ff = torch.mean(ff, dim=0, keepdim=True)
                pass
            
            ff_list.append(ff)
            # # ff_full_seq_list.append(ff_full_seq)
            # del ff
            # gc.collect()

    if 'bev' in name:
        # 拼接成最终结果
        features = torch.cat(ff_list)
    else:
        features = torch.stack(ff_list,dim=0)
    # features_full_seq = torch.cat(ff_full_seq_list, dim=0)

    # return features, features_full_seq
    return features

def load_all_features_ff_full_seq(out_dir: str, device: torch.device, name: str):
    """
    读取保存的所有 ff 和 ff_full_seq，拼接成最终结果
    """
    ff_list = []
    # ff_full_seq_list = []
    
    files_ff = sorted(os.listdir(out_dir))  # 获取所有文件，并按顺序读取
    for file in tqdm(files_ff):
        # 如果文件名是 bytes 类型，先进行解码为 str
        if isinstance(file, bytes):
            file = file.decode('utf-8')  # 解码为 utf-8 字符串
        if "_ff_full_seq.pt" in file:
            ff_path = os.path.join(out_dir, file)
            # ff_full_seq_path = os.path.join(out_dir, file.replace("_ff", "_ff_full_seq"))
            
            ff = torch.load(ff_path, map_location=device)
            # ff_full_seq = torch.load(ff_full_seq_path, map_location=device)
            
            if 'bev' in name:
                # average
                ff = torch.mean(ff, dim=0, keepdim=True)
                pass
            
            ff_list.append(ff)
            # # ff_full_seq_list.append(ff_full_seq)
            # del ff
            # gc.collect()

    # 拼接成最终结果
    features = torch.cat(ff_list, dim=0)
    # features_full_seq = torch.cat(ff_full_seq_list, dim=0)

    # return features, features_full_seq
    return features

def collect_video_number(gallery_img_path_l):
    """
    collect frame number in a video
    """
    gallery_name_d = {}
    # find out how many images are in one class
    for gallery_img_path in gallery_img_path_l:
        gallery_name = gallery_img_path.split('/')[-2]
        if gallery_name not in gallery_name_d.keys():
            gallery_name_d[gallery_name] = 1
        else:
            gallery_name_d[gallery_name] += 1
    return gallery_name_d

def split_feature(video_frame_d, label_to_save, feature, ff_full_seq, feature_in_last_batch, full_feature_in_last_batch):
    """
    
    """
    
    if feature_in_last_batch == None:
        start_idx = 0
        end_idx = 0
        # feature_in_last_batch = None
        
        # 这里面装的是分好的feature
        feature_split_l = []
        full_seq_split_l = []
        # 上一个batch遗留下的视频帧的数目
        last_batch_frame_number = 0
        to_save_dir_name_l = []
    else:
        to_save_dir_name_l = []
        last_batch_frame_number = feature_in_last_batch.shape[0]
        # 这里的label_to_save有点问题
        feature_split_l = []
        full_seq_split_l = []
        start_idx = 0
        end_idx = 0
        # end_idx = start_idx + (video_frame_d[sorted(video_frame_d.keys())[label_to_save-1]]-feature_in_last_batch.shape[0])
        end_idx = start_idx + (video_frame_d[sorted(video_frame_d.keys())[label_to_save]]-feature_in_last_batch.shape[0])
        
        
        feature_split_l.append(torch.cat([feature_in_last_batch, feature[start_idx:end_idx]],dim=0))
        full_seq_split_l.append(torch.cat([full_feature_in_last_batch, ff_full_seq[start_idx:end_idx]],dim=0))
        # to_save_dir_name_l.append(sorted(video_frame_d.keys())[label_to_save-1])
        to_save_dir_name_l.append(sorted(video_frame_d.keys())[label_to_save])
        
        label_to_save += 1
        
        start_idx = end_idx
    while True:
        end_idx = start_idx + video_frame_d[sorted(video_frame_d.keys())[label_to_save]]
        
        if end_idx > feature.shape[0]:
            # 这个是当前batch的， 但理应属于下个batch的video中的frame
            feature_in_last_batch = feature[start_idx:]
            full_feature_in_last_batch = ff_full_seq[start_idx:]
            break
        if end_idx == feature.shape[0]:
            # 一段视频
            feature_split_l.append(feature[start_idx:end_idx])
            full_seq_split_l.append(ff_full_seq[start_idx:end_idx])
            
            feature_in_last_batch = None
            full_feature_in_last_batch= None
            
            to_save_dir_name_l.append(sorted(video_frame_d.keys())[label_to_save])
            label_to_save += 1
            break
        
        # 一段视频
        feature_split_l.append(feature[start_idx:end_idx])
        full_seq_split_l.append(ff_full_seq[start_idx:end_idx])
        
        
        to_save_dir_name_l.append(sorted(video_frame_d.keys())[label_to_save])
        
        start_idx = end_idx
        # 下一段视频
        label_to_save += 1
    
    batch_number = 0
    for ii, i in enumerate(feature_split_l):
        assert video_frame_d[sorted(video_frame_d.keys())[label_to_save-len(feature_split_l)+ii]] == i.shape[0]
        batch_number += i.shape[0]
    
    if last_batch_frame_number == 0:
        # 判断当前的batch的数据是不是全用上了（被保存 or 留给下个batch feature_in_last_batch）
        assert feature.shape[0] == batch_number + feature_in_last_batch.shape[0]
    else:
        if feature_in_last_batch !=None:
            assert feature.shape[0] == batch_number - last_batch_frame_number + feature_in_last_batch.shape[0]
        else:
            if feature.shape[0] == batch_number - last_batch_frame_number:
                pass
            else:
                assert feature.shape[0] == batch_number - last_batch_frame_number 
    
    assert len(to_save_dir_name_l) == len(feature_split_l)
    
    return feature_in_last_batch,full_feature_in_last_batch, feature_split_l, full_seq_split_l, label_to_save, to_save_dir_name_l