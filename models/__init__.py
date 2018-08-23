from .losses import SphereLoss, CentreLoss, AttributeLoss, ClassificationLoss
from .networks import DenseNet169, Resnet50, Resnext50, Resnext100, Pnasnet5
losses = {'sphere':SphereLoss, 'centre':CentreLoss, 'attribute':AttributeLoss, 'softmax':ClassificationLoss}
networks = {'resnet50':Resnet50, 'densenet169':DenseNet169, 'resnext50':Resnext50, 'resnext100':Resnext100, 'pnasnet5':Pnasnet5}

__all__ = [networks, losses]
