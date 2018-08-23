
import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import densenet169
from .resnet import resnet50
from .model_utils import *
import pretrainedmodels


class DenseNet169_(torch.nn.Module):
    def __init__(self, f_dim=512, pretrained=True, **kwargs):
        super(DenseNet169, self).__init__()
        
        self.model = densenet169(pretrained=pretrained, num_classes=f_dim)
        self.norm = kwargs.get('norm', False)
        self.siamese = kwargs.get('siamese', True)
        
    def forward(self, x, mode_flag='train'):

        if self.siamese:
            feat = self.model.features(x)
        else:
            if mode_flag == 'train':
                x = x.split(x.size(0)//2)
                feat = torch.cat([self.model(x[0], 'skt'), self.model(x[1], 'pho')])
            else:
                feat = self.model(x, mode_flag)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat

class DenseNet169(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(DenseNet169, self).__init__()

        model_name = 'densenet169' # could be fbresnet152 or inceptionresnetv2
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(2048, f_dim, bias=True)

        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model.features(x)
        feat = self.pool(feat).squeeze(3).squeeze(2)
        feat = self.fc(feat)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat 


class Resnet50(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(Resnet50, self).__init__()

        model_name = 'resnet50' # could be fbresnet152 or inceptionresnetv2
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        self.fc = nn.Linear(2048, f_dim, bias=True)

        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model.features(x)
        feat = self.pool(feat).squeeze(3).squeeze(2)
        feat = self.fc(feat)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat 

class Resnext50(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(Resnext50, self).__init__()

        model_name = 'se_resnext50_32x4d' # could be fbresnet152 or inceptionresnetv2
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(2048, f_dim, bias=True)

        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model.features(x)
        feat = self.pool(feat).squeeze(3).squeeze(2)
        feat = self.fc(feat)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat          

class Resnext100(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(Resnext100, self).__init__()

        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(2048, f_dim, bias=True)

        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model.features(x)
        feat = self.pool(feat).squeeze(3).squeeze(2)
        feat = self.fc(feat)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat          

class Pnasnet5(nn.Module):
    def __init__(self, f_dim, pretrained=True, **kwargs):
        super(Pnasnet5, self).__init__()

        model_name = 'pnasnet5large' # could be fbresnet152 or inceptionresnetv2
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(4320, f_dim, bias=True)

        self.norm = kwargs.get('norm', False)

    def forward(self, x, mode_flag='pho'):
        feat = self.model.features(x)
        feat = self.pool(feat).squeeze(3).squeeze(2)
        feat = self.fc(feat)

        if self.norm:
            feat = F.normalize(feat, p=2, dim=1)
        return feat          
