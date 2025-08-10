import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import copy

class Dataloader_University(Dataset):
    def __init__(self,root,transforms,names=['satellite','street','drone','google','bev']):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        if 'bev' not in transforms.keys():
            pass
        else:
            self.transforms_bev = transforms['bev']
        # self.transforms_google = transforms['train']
        self.root = root
        self.names = names
        #获取所有图片的相对路径分别放到对应的类别中
        #{satelite:{0839:[0839.jpg],0840:[0840.jpg]}}
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root,name,cls_name))
                img_path_list = [os.path.join(root,name,cls_name,img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        # 获取设置名字与索引之间的镜像
        cls_names = os.listdir(os.path.join(root,'bev'))
        cls_names.sort()
        map_dict={i:cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        # self.index_cls_nums = 2

    #从对应的类别中抽一张出来
    def sample_from_cls(self,name,cls_num):
        img_path = self.dict_path[name][cls_num]
        if len(img_path)==0:
            if name == "satellite" or name == "bev" or name == "drone":
                raise Exception('ERROR')
            else:
                return None
                # raise Exception('ERROR')
        # 等概率取出1个样本
        img_path = np.random.choice(list(img_path),1)[0]
        img = Image.open(img_path).convert('RGB')
        return img


    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite",cls_nums)
        img_s = self.transforms_satellite(img)

        img = self.sample_from_cls("street",cls_nums)
        img_st = self.transforms_drone_street(img)

        img = self.sample_from_cls("drone",cls_nums)
        img_d = self.transforms_drone_street(img)

        img = self.sample_from_cls("google", cls_nums)
        if img == None:
            img = self.sample_from_cls("street", cls_nums)
        img_google = self.transforms_drone_street(img)

        if hasattr(self,'transforms_bev')!=False:
            img = self.sample_from_cls("bev", cls_nums)
            img_bev = self.transforms_bev(img)
            return img_s, img_st, img_d, img_google, img_bev, index
        else:
            return img_s, img_st, img_d, img_google, index


        # try:
        #     img_google = self.transforms_drone_street(img)
        # except:
        #     img = self.sample_from_cls("google", cls_nums)
        #     # img_google = self.transforms_drone_street(img)
        #     raise Exception('error')



    def __len__(self):
        return len(self.cls_names)



class Sampler_University(object):
    r"""Base class for all Samplers.
    用于产生getitem的索引
    https://zhuanlan.zhihu.com/p/76893455
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8,sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0,self.data_len)
        np.random.shuffle(list)
        # 保证在一个类内取'self.sample_num'个样本
        # 下面这个nums是生成getitem的index
        nums = np.repeat(list,self.sample_num,axis=0)
        return iter(nums)
        # # 想每次getitem中的index是两个不同的数，一个给drone/bev 另一个satellite
        # satellite_list = copy.deepcopy(list)
        # np.random.shuffle(list)
        # return iter(np.stack([list,satellite_list],axis=1))

    def __len__(self):
        return (self.data_len)


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    拼成batch时的必经函数,手动设置如何拼的方式
    """
    # *变量 为可变参数
    # img_s,img_st,img_d,ids = zip(*batch)
    if len((batch[0])) == 6:
        img_s,img_st,img_d,img_google,img_bev,ids = zip(*batch)
        ids = torch.tensor(ids,dtype=torch.int64)
        return [torch.stack(img_s, dim=0),ids],\
               [torch.stack(img_st,dim=0),ids], \
               [torch.stack(img_d,dim=0),ids], \
               [[torch.stack(img_google,dim=0),ids], [torch.stack(img_bev,dim=0),ids]]
    else:
        img_s, img_st, img_d, img_google, ids = zip(*batch)
        ids = torch.tensor(ids, dtype=torch.int64)
        return [torch.stack(img_s, dim=0), ids], \
               [torch.stack(img_st, dim=0), ids], \
               [torch.stack(img_d, dim=0), ids], \
               [torch.stack(img_google, dim=0), ids]

if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list ={"satellite": transforms.Compose(transform_train_list),
                            "train":transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="/home/dmmm/University-Release/train",transforms=transform_train_list,names=['satellite','drone'])
    samper = Sampler_University(datasets,8)
    dataloader = DataLoader(datasets,batch_size=8,num_workers=0,sampler=samper,collate_fn=train_collate_fn)
    for data_s,data_d in dataloader:
        print()


