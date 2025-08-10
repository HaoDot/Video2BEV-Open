import os
import torch
import yaml
from model import two_view_net, three_view_net
import math



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
    network.load_state_dict(torch.load(save_path))
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

