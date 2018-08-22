
import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import densenet169
from .resnet import resnet50
from .model_utils import *

class DenseNet169(torch.nn.Module):
    def __init__(self, f_dim=512, pretrained=True, **kwargs):
        super(DenseNet169, self).__init__()
        
        self.model = densenet169(pretrained=pretrained, num_classes=f_dim)
        self.norm = kwargs.get('norm', False)
        self.siamese = kwargs.get('siamese', True)
        
    def forward(self, x, mode_flag='train'):

        if self.siamese:
            feat = self.model(x)
        else:
            if mode_flag == 'train':
                x = x.split(x.size(0)//2)
                feat = torch.cat([self.model(x[0], 'skt'), self.model(x[1], 'pho')])
            else:
                feat = self.model(x, mode_flag)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat

class Resnet50(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained=pretrained, num_classes=f_dim)
        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model(x)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat        


