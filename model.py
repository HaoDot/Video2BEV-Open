import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from backbone.vit_pytorch import vit_small_patch16_224_reid
# from backbone.vit_pytorch import vit_small_patch16_224_fusion
import numpy as np
######################################################################
class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim = 2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim = 512)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool=='gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x

class ft_net_ibn(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_ibn, self).__init__()
        model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim = 512)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        # x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
        elif self.pool == 'gem':
            x = self.model.gem2(x)

        x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x

# class two_view_net(nn.Module):
#     def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, circle=False,ibn=False,vit=False,itc=False):
#         super(two_view_net, self).__init__()
#         if VGG16:
#             self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
#         elif ibn:
#             self.model_1 = ft_net_ibn(class_num, stride=stride, pool=pool)
#         else:
#             self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
#         if share_weight:
#             self.model_2 = self.model_1
#         else:
#             if VGG16:
#                 self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
#             elif ibn:
#                 self.model_1 = ft_net_ibn(class_num, stride=stride, pool=pool)
#             else:
#                 self.model_2 =  ft_net(class_num, stride = stride, pool = pool)
#
#         self.circle = circle
#
#         self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
#         if pool =='avg+max':
#             self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)
#         if VGG16:
#             self.classifier = ClassBlock(512, class_num, droprate, return_f = circle)
#             if pool =='avg+max':
#                 self.classifier = ClassBlock(1024, class_num, droprate, return_f = circle)
#
#     def forward(self, x1, x2):
#         if x1 is None:
#             y1 = None
#         else:
#             x1 = self.model_1(x1)
#             y1 = self.classifier(x1)
#
#         if x2 is None:
#             y2 = None
#         else:
#             x2 = self.model_2(x2)
#             y2 = self.classifier(x2)
#         return y1, y2
def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )

class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False,
                 VGG16=False, circle=False,ibn=False,vit=False,itc=False, itm=False, itm_share=False, lpn=False):
        super(two_view_net, self).__init__()
        self.vit = vit
        self.itc = itc
        # temperature factor for itc
        import numpy as np
        if self.itc:
            self.logit_scale = (nn.Parameter(1/(torch.ones([]) * np.log(1 / 0.07)).exp()))
        if itm:
            self.itm_share = itm_share
            self.itm = itm
        else:
            self.itm = False
        if self.vit:
            model_path = './pre_trained/vit_small_p16_224-15ec54c9.pth'

        self.lpn = lpn
        if lpn:
            self.pool = 'avg'
            #  4 parts and 1 cls token
            self.block = 5
            for i in range(self.block):
                name = 'classifier' + str(i)
                setattr(self, name, ClassBlock(768, class_num, droprate))
        else:
            # resnet50
            # self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
            # vit-samll
            self.classifier = ClassBlock(768, class_num, droprate, return_f=circle)

        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            # self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        elif ibn:
            self.model_1 = ft_net_ibn(class_num, stride=stride, pool=pool)
            # self.model_2 = ft_net_ibn(class_num, stride=stride, pool=pool)
        elif vit:
            self.model_1 = vit_small_patch16_224_reid(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1,
                                                            drop_rate= 0.0, attn_drop_rate=0.0, itc = itc)
            self.model_1.load_param(model_path)

        else:
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)


        if share_weight:
            self.model_3 = self.model_1
        else:
            raise Exception('only support  share weight')
            if VGG16:
                self.model_3 = ft_net_VGG16(class_num, stride = stride, pool = pool)
            elif ibn:
                self.model_3 = ft_net_ibn(class_num, stride=stride, pool=pool)
            elif vit:
                self.model_3 = vit_small_patch16_224_reid(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1,
                                                            drop_rate= 0.0, attn_drop_rate=0.0, itc = itc)
                self.model_3.load_param(model_path)
            else:
                self.model_3 = ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle



        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)

    def get_part_pool(self, x, pool='avg', no_overlap=True, region_number = 4):
        """

        Args:
            x:
            pool:
            no_overlap:
            region_number:

        Returns: [cls token, part1:part4]

        """
        bs, n, c = x[:,1:,:].shape
        cls = x[:,:1,:]

        tmp_l = []
        for i in range(0, n, 16):
            chunk = x[:,1:,:][:,i:(i+16),:]
            tmp_l.append(chunk)
        # B H W C -> B C H W
        x = torch.stack(tmp_l,dim=1).permute(0,3,1,2)
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1))
        # x:[B,2048,H,W]
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*region_number),W/(2*region_number)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(region_number-c_h)*2, W+(region_number-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            # c_h: center of feature_map
            c_h, c_w = int(H/2), int(W/2)
            # per_h: height of each partition
            # self.block = 4
            per_h, per_w = H/(2*region_number),W/(2*region_number)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(region_number):
            i = i + 1
            if i < region_number:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    # pad side of feature
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        tmp_l = []
        tmp_l.append(cls.squeeze(-2))
        for i in result:
            tmp_l.append(i.squeeze(-1).squeeze(-1))
        return torch.stack(tmp_l, dim=1)

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):

            part[i] = x[:,i,:]
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=1)
        return y

    def lpn_norm_feature(self, ff, block_number=5):
        assert len(ff.shape) == 3
        assert ff.shape[1] == block_number

        fnorm = torch.norm(ff, p=2, dim=2, keepdim=True) * np.sqrt(block_number)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
        return ff

    def forward(self, x1, _, x3, x4 = None): # x4 is extra data
        # dataloaders['satellite'], dataloaders['street'], dataloaders['drone'], dataloaders['google']
        if x1 is None:
            y1 = None
            cls_feats_1 = None
            full_seq_1 = None
        else:
            x1 = self.model_1(x1)

            if self.vit:
                if self.lpn:
                    x1 = self.get_part_pool(x1)
                else:
                    # choose the cls token
                    x1 = x1[:, 0, :]
            else:
                x1 = x1.view(x1.size(0), x1.size(1))
            if self.lpn:
                y1 = self.part_classifier(x1)
            else:
                y1 = self.classifier(x1)
            if self.itc:
                if not self.lpn:
                    x1_f = x1
                    cls_feats_1 = F.normalize(x1_f, dim=-1)
                else:
                    cls_feats_1 = self.lpn_norm_feature(x1)


        if x3 is None:
            y3 = None
            cls_feats_3 = None
            full_seq_3 = None
        else:
            x3 = self.model_3(x3)

            if self.vit:
                if self.lpn:
                    x3 = self.get_part_pool(x3)
                else:
                    # choose the cls token
                    x3 = x3[:, 0, :]
            else:
                x3 = x3.view(x3.size(0), x3.size(1))


            if self.lpn:
                y3 = self.part_classifier(x3)
            else:
                y3 = self.classifier(x3)

            if self.itc:
                if not self.lpn:
                    x3_f = x3
                    cls_feats_3 = F.normalize(x3_f, dim=-1)
                else:
                    cls_feats_3 = self.lpn_norm_feature(x3)
                    # cls_feats_3 = (x3)

        if self.itc:
            return [y1, cls_feats_1], [y3, cls_feats_3]
        else:
            return y1, y3


import math
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False,
                 VGG16=False, circle=False,ibn=False,vit=False,itc=False, itm=False, itm_share=False):
        super(three_view_net, self).__init__()
        self.vit = vit
        self.itc = itc
        # temperature factor for itc
        import numpy as np
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if itm:
            self.itm_share = itm_share
            self.itm= itm
        if self.vit:
            model_path = './backbone/vit_small_p16_224-15ec54c9.pth'
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        elif ibn:
            self.model_1 = ft_net_ibn(class_num, stride=stride, pool=pool)
            self.model_2 = ft_net_ibn(class_num, stride=stride, pool=pool)
        elif vit:
            self.model_1 = vit_small_patch16_224_reid(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1,
                                                            drop_rate= 0.0, attn_drop_rate=0.0, itc = itc)
            self.model_1.load_param(model_path)
            self.model_2 = vit_small_patch16_224_reid(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1,
                                                            drop_rate= 0.0, attn_drop_rate=0.0, itc = itc)
            self.model_2.load_param(model_path)
        else:
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 = ft_net_VGG16(class_num, stride = stride, pool = pool)
            elif ibn:
                self.model_3 = ft_net_ibn(class_num, stride=stride, pool=pool)
            elif vit:
                self.model_3 = vit_small_patch16_224_reid(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1,
                                                            drop_rate= 0.0, attn_drop_rate=0.0, itc = itc)
                self.model_3.load_param(model_path)
            else:
                self.model_3 = ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle

        # resnet50
        # self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        # vit-samll
        self.classifier = ClassBlock(768, class_num, droprate, return_f = circle)
        if self.itm:
            if self.itm_share:
                # TODO:大学习率+cross-attention+itm head (share LPN)
                # self.model_1_fusion = vit_small_patch16_224_fusion(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1, drop_rate= 0.0, attn_drop_rate=0.0)
                # self.model_2_fusion = vit_small_patch16_224_fusion(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1, drop_rate= 0.0, attn_drop_rate=0.0)
                # self.model_3_fusion = self.model_1_fusion
                self.model_3_fusion = vit_small_patch16_224_fusion(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1, drop_rate= 0.0, attn_drop_rate=0.0)
                self.itm_head_3 = nn.Linear(768, 2)
                # initialize
                if isinstance(self.itm_head_3, nn.Linear):
                    trunc_normal_(self.itm_head_3.weight, std=.02)
                    if isinstance(self.itm_head_3, nn.Linear) and self.itm_head_3.bias is not None:
                        nn.init.constant_(self.itm_head_3.bias, 0)
                # self.itm_head_2 = self.itm_head_1
                pass
            else:
                # self.model_1_fusion =
                # self.model_2_fusion =
                # self.model_3_fusion =
                pass
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        # dataloaders['satellite'], dataloaders['street'], dataloaders['drone'], dataloaders['google']
        if x1 is None:
            y1 = None
            cls_feats_1 = None
            full_seq_1 = None
        else:
            x1 = self.model_1(x1)
            full_seq_1 = x1
            if self.vit:
                # choose the cls token
                x1 = x1[:,0]
            else:
                x1 = x1.view(x1.size(0), x1.size(1))
            y1 = self.classifier(x1)
            if self.itc:
                # cls_feats_1 = self.model_1.ITC_head(x1)
                cls_feats_1 = y1 / y1.norm(dim=-1, keepdim=True)
                # cls_feats_1 = cls_feats_1 / cls_feats_1.norm(dim=-1, keepdim=True)


        if x2 is None:
            y2 = None
            cls_feats_2 = None
            full_seq_2 = None
        else:
            x2 = self.model_2(x2)
            full_seq_2 = x2
            if self.vit:
                # choose the cls token
                x2 = x2[:,0]
            else:
                x2 = x2.view(x2.size(0), x2.size(1))
            y2 = self.classifier(x2)
            if self.itc:
                # cls_feats_2 = self.model_2.ITC_head(x2)
                cls_feats_2 = y2 / y2.norm(dim=-1, keepdim=True)
                # cls_feats_2 = cls_feats_2 / cls_feats_2.norm(dim=-1, keepdim=True)

        if x3 is None:
            y3 = None
            cls_feats_3 = None
            full_seq_3 = None
        else:
            x3 = self.model_3(x3)
            full_seq_3 = x3
            if self.vit:
                # choose the cls token
                x3 = x3[:,0]
            else:
                x3 = x3.view(x3.size(0), x3.size(1))
            y3 = self.classifier(x3)
            if self.itc:
                # cls_feats_3 = self.model_3.ITC_head(x3)
                cls_feats_3 = y3 / y3.norm(dim=-1, keepdim=True)
                # cls_feats_3 = cls_feats_3 / cls_feats_3.norm(dim=-1, keepdim=True)

        if x4 is None:
            if self.itc:
                if self.itm:
                    return [y1,cls_feats_1, full_seq_1], [y2,cls_feats_2, full_seq_2], [y3,cls_feats_3,full_seq_3]
                else:
                    return [y1,cls_feats_1], [y2,cls_feats_2], [y3,cls_feats_3]

            else:
                return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            full_seq_4 = x4
            if self.vit:
                # choose the cls token
                x4 = x4[:,0]
            else:
                x4 = x4.view(x4.size(0), x4.size(1))
            y4 = self.classifier(x4)
            if self.itc:
                # cls_feats_4 = self.model_2.ITC_head(x4)
                cls_feats_4 = y4 / y4.norm(dim=-1, keepdim=True)
                # cls_feats_4 = cls_feats_4 / cls_feats_4.norm(dim=-1, keepdim=True)
                if self.itm:
                    return [y1, cls_feats_1, full_seq_1], [y2, cls_feats_2, full_seq_2], [y3, cls_feats_3, full_seq_3]
                else:
                    return [y1,cls_feats_1], [y2,cls_feats_2], [y3,cls_feats_3],[y4,cls_feats_4]
            else:
                return y1, y2, y3, y4
            # return [y1,x1], [y2,x2], [y3,x3], [y4,x4]


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(751, droprate=0.5, VGG16=True)
    #net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output,output = net(input,input)
    print('net output size:')
    print(output.shape)
