import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models

def findconv(net,includenorm=True):
        layers = pushconv([],net,includenorm)
        return layers

def pushconv(layers,container,includenorm=True):
    if isinstance(container, models.resnet.ResNet):
        pushconv(layers,container.conv1,includenorm)
        pushconv(layers,container.bn1,includenorm)
        pushconv(layers,container.layer1,includenorm)
        pushconv(layers,container.layer2,includenorm)
        pushconv(layers,container.layer3,includenorm)
        pushconv(layers,container.layer4,includenorm)
        pushconv(layers,container.fc,includenorm)
    elif isinstance(container, models.mobilenet.MobileNetV2):
        pushconv(layers,container.features,includenorm)
        pushconv(layers,container.classifier,includenorm)
    elif isinstance(container, models.AlexNet):
        pushconv(layers,container.features,includenorm)
        pushconv(layers,container.classifier,includenorm)
    elif isinstance(container, torch.nn.Linear):
        layers.append(container)
    elif isinstance(container, torch.nn.Conv2d):
        layers.append(container)
    elif isinstance(container, models.resnet.BasicBlock):
        pushconv(layers,container.conv1,includenorm)
        pushconv(layers,container.bn1,includenorm)
        pushconv(layers,container.conv2,includenorm)
        pushconv(layers,container.bn2,includenorm)
        if isinstance(container.downsample,torch.nn.Sequential):
            pushconv(layers,container.downsample[0],includenorm)
            pushconv(layers,container.downsample[1],includenorm)
    elif isinstance(container, models.resnet.Bottleneck):
        pushconv(layers,container.conv1,includenorm)
        pushconv(layers,container.bn1,includenorm)
        pushconv(layers,container.conv2,includenorm)
        pushconv(layers,container.bn2,includenorm)
        pushconv(layers,container.conv3,includenorm)
        if isinstance(container.downsample,torch.nn.Sequential):
            pushconv(layers,container.downsample[0],includenorm)
        pushconv(layers,container.bn3,includenorm)
        if isinstance(container.downsample,torch.nn.Sequential):
            pushconv(layers,container.downsample[1],includenorm)
        # if isinstance(container.downsample,torch.nn.Sequential):
        #     pushconv(layers,container.downsample[0])
        #     pushconv(layers,container.downsample[1])
    elif isinstance(container, torch.nn.modules.batchnorm.BatchNorm2d) and includenorm:
        layers.append(container)
    elif isinstance(container,torch.nn.Sequential):
        for l in range(0,len(container)):
            pushconv(layers,container[l],includenorm)
    elif isinstance(container, models.mobilenet.ConvBNReLU):
        for l in range(0,len(container.conv)):
            pushconv(layers,container.conv[l],includenorm)
    elif isinstance(container, models.mobilenet.InvertedResidual):
        for l in range(0,len(container.conv)):
            pushconv(layers,container.conv[l],includenorm)

    return layers

def permute(layer,dimen):
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.grad.flatten(1).permute([1,0])
    if isinstance(layer, torch.nn.Linear):
        return layer.weight.grad.flatten(1).permute([1,0])
def zerograd(layer):
    layer.weight.grad.zero_()
