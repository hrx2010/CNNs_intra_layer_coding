import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models

def findconv(net):
    if isinstance(net, models.resnet.ResNet):
        layers = []
        for layer in dir(net):
            if isinstance(getattr(net,layer),torch.nn.Conv2d):
                layers.append(getattr(net,layer))
        for layer in dir(net):
            if isinstance(getattr(net,layer),torch.nn.Sequential):
                for i in range(0,len(getattr(net,layer))):
                    for convs in dir(getattr(net,layer)[i]):
                        #print(convs)
                        if isinstance(getattr(getattr(net,layer)[i],convs),torch.nn.Conv2d):
                            layers.append(getattr(getattr(net,layer)[i],convs))
        for layer in dir(net):
            if isinstance(getattr(net,layer),torch.nn.Linear):
                layers.append(getattr(net,layer))
        return layers                        
    else:
        layers = []
        for l in range(0,len(net.features)):
            if isinstance(net.features[l], torch.nn.Conv2d):
                layers.append(net.features[l])
                
        for l in range(0,len(net.classifier)):
            if isinstance(net.classifier[l], torch.nn.Linear):
                layers.append(net.classifier[l])
        return layers            

def permute(layer,dimen):
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.grad.flatten(1).permute([1,0])
    if isinstance(layer, torch.nn.Linear):
        return layer.weight.grad.flatten(1).permute([1,0])
def zerograd(layer):
    layer.weight.grad.zero_()
