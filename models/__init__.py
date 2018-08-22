from .losses import SphereLoss, CentreLoss, AttributeLoss, ClassificationLoss
from .networks import DenseNet169, Resnet50
losses = {'sphere':SphereLoss, 'centre':CentreLoss, 'attribute':AttributeLoss, 'softmax':ClassificationLoss}
networks = {'resnet50':Resnet50, 'densenet169':DenseNet169}

__all__ = [networks, losses]
