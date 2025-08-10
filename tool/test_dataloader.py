from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
from os.path import join
import numpy as np
from PIL import Image
class Dataloader_University(Dataset):
    def __init__(self,root,names,split='train'):
        super(Dataloader_University).__init__()
        self.root_path = root
        assert self.root_path.split('/')[-1]==split
        self.root_path = join(self.root_path,names)