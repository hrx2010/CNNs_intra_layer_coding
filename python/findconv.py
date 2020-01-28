import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models

def findconv(net):
        layers = pushconv([],net)
        return layers

def pushconv(layers,container):
    if isinstance(container, models.resnet.ResNet):
        pushconv(layers,container.conv1)
        pushconv(layers,container.layer1)
        pushconv(layers,container.layer2)
        pushconv(layers,container.layer3)
        pushconv(layers,container.layer4)
        pushconv(layers,container.fc)
    elif isinstance(container, models.mobilenet.MobileNetV2):
        pushconv(layers,container.features)
        pushconv(layers,container.classifier)
    elif isinstance(container, models.AlexNet):
        pushconv(layers,container.features)
        pushconv(layers,container.classifier)
    elif isinstance(container, torch.nn.Linear):
        layers.append(container)
    elif isinstance(container, torch.nn.Conv2d):
        layers.append(container)
    elif isinstance(container, models.resnet.Bottleneck):
        pushconv(layers,container.conv1)
        pushconv(layers,container.conv2)
        pushconv(layers,container.conv3)
        if hasattr(container,'downsample'):
            pushconv(layers,container.downsample)
    elif isinstance(container, torch.nn.modules.batchnorm.BatchNorm2d):
        layers.append(container)
    elif isinstance(container,torch.nn.Sequential):
        for l in range(0,len(container)):
            pushconv(layers,container[l])
    elif isinstance(container, models.mobilenet.ConvBNReLU):
        for l in range(0,len(container.conv)):
            pushconv(layers,container.conv[l])
    elif isinstance(container, models.mobilenet.InvertedResidual):
        for l in range(0,len(container.conv)):
            pushconv(layers,container.conv[l])

    return layers

def permute(layer,dimen):
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.grad.flatten(1).permute([1,0])
    if isinstance(layer, torch.nn.Linear):
        return layer.weight.grad.flatten(1).permute([1,0])
def zerograd(layer):
    layer.weight.grad.zero_()
