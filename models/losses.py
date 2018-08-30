import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *

class TripletLoss(nn.Module):
    def __init__(self, delta=0.3, dist='sq'):
        super(TripletLoss, self).__init__()

        self.delta = delta
        if dist == 'sqr':
            dist = lambda z1, z2: torch.norm(z1-z2, p=2, dim=1)
        elif dist == 'sq':
            dist = lambda z1, z2: torch.norm(z1-z2, p=2, dim=1) ** 2
        self.dist = dist

    def forward(self, s, pp, pn):

        dp = self.dist(s, pp)
        dn = self.dist(s, pn)
        dist = self.delta+dp-dn
        dist = torch.clamp(dist, min=0.0)

        return torch.mean(dist)




class SphereLoss(nn.Module):
    def __init__(self, config=None, gamma=0, **kwargs):
        super(SphereLoss, self).__init__()

        self.fc = AngleLinear(config.feat_dim, config.c_dim).cuda()

        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = self.fc(input)
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.detach() * 0.0 #size=(B,Classnum)
        index.scatter_(1,target,1)
        index = index.byte()

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=0)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()
        
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss      


class CentreLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, config, **kwargs):
        super(CentreLoss, self).__init__()
        self.num_classes = config.c_dim
        self.feat_dim = config.feat_dim
        self.use_gpu = True

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # one-hot
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class AttributeLoss(nn.Module):
    def __init__(self, config=None, **kwargs):
        super(AttributeLoss, self).__init__()

        self.fc = nn.Linear(config.feat_dim, 100).cuda()
        self.register_buffer('wordvec',kwargs['wordvec'])

    def forward(self, input, target):
        logits = self.fc(input)
        loss = (logits-self.wordvec[target]).abs().mean()

        return loss  

class ClassificationLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super(ClassificationLoss, self).__init__()

        self.fc = nn.Linear(config.feat_dim, config.c_dim).cuda()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logits = self.fc(input)
        loss = self.loss(logits, target)

        return loss
