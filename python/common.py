import time
import h5py
import torch
import numpy as np
import scipy.io as io
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import resnetpy
import alexnetpy

device = None

batch_size = 25

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
	[transforms.Resize(256,interpolation=1),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(rgb_avg, rgb_std)])

def loadvarstats(archname,trantype,testsize):
    mat = io.loadmat(('%s_%s_stats_%d.mat' % (archname, trantype, testsize)))
    return np.array(mat['cov'])

def loadrdcurves(archname,tranname,trantype,l,part):
    mat = io.loadmat('%s_%s_val_%03d_0100_output_%s_%s' % (archname,tranname,l,trantype,part))
    return mat['%s_Y_sse'%part], mat['%s_delta'%part], mat['%s_coded'%part]
    #mat = io.loadmat('%s_%s_val_1000_%d_%d_output_%s_%s' % (archname,tranname,l+1,l+1,trantype,part))
    #return mat['%s_Y_sse'%part][l,0], mat['%s_delta'%part][l,0], mat['%s_coded'%part][l,0]

def findrdpoints(y_sse,delta,coded,lam):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float('inf')
    ind1 = np.nanargmin(y_sse,1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),y_sse.shape)
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    point = y_sse + lam*coded == (y_sse + lam*coded).min(0)

    return np.select(point, y_sse), np.select(point, delta), np.select(point, coded)


def loaddataset(gpuid,testsize):
    global device
    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
    images = datasets.ImageNet(\
                root='~/Developer/ILSVRC2012_devkit_t12',\
                split='val',transform=transdata)
    images.samples = images.samples[::len(images.samples)//testsize]
    labels = torch.tensor([images.samples[i][1] for i in range(0,len(images))])

    return images, labels.to(device)

def loadnetwork(archname,gpuid,testsize):
    global device
    if archname == 'alexnetpy':
        net = alexnetpy.alexnet(pretrained=True)
    elif archname == 'resnet18py':
        net = resnetpy.resnet18(pretrained=True)
    elif archname == 'resnet34py':
        net = resnetpy.resnet34(pretrained=True)
    elif archname == 'resnet50py':
        net = resnetpy.resnet50(pretrained=True)
    elif archname == 'mobilenetv2py':
        net = models.mobilenet.mobilenet_v2(pretrained=True)
        # load the dataset
    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
    images = datasets.ImageNet(\
                root='~/Developer/ILSVRC2012_devkit_t12',\
                split='val',transform=transdata)
    images.samples = images.samples[::len(images.samples)//testsize]
    labels = torch.tensor([images.samples[i][1] for i in range(0,len(images))])

    return net.to(device), images, labels.to(device)

def gettrans(archname,trantype,tranname,layer,version='v7.3'):
    if version == 'v7.3':
        file = h5py.File('%s_%s_50000_%s.mat' %(archname,tranname,trantype),'r')
        return torch.FloatTensor(file[file['T'][0,layer]]).permute([3,2,1,0]).flatten(2).to(device)
    else:
        mat = io.loadmat('%s_%s_%s_10000.mat' % (archname,trantype,tranname))
        return torch.FloatTensor(mat['T'][0,layer]).flatten(2).to(device)

def getdevice():
	global device
	return device

def getperm(trantype,dir=1):
    if trantype == 'inter':
        return [1,0,2], [0,1]
    elif trantype == 'intra':
        return [2,0,1], [0,1]
    elif trantype == 'exter':
        return [0,1,2], [0,1]
    elif trantype == 'joint':
        return [0,1,2], [1,0]
    elif trantype == 'extra':
        return [1,0,2], [1,0]
    
def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def quantize(weights, delta, b):
    if b > 0:
        minpoint = -(2**(b-1))*delta
        maxpoint = +(2**(b-1))*delta
    else:
        minpoint = 0
        maxpoint = 0

    return (delta*(weights/delta).round()).clamp(minpoint,maxpoint)
    
def min_inds(mat,axis):
    ind1 = np.argmin(mat,axis)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),mat.shape)

    return inds

def gettop1(logp):
    logp = logp.exp()
    logp = logp/logp.sum(1).reshape(-1,1)
    vals, inds = logp.max(1)

    return inds
        
def predict(net,images,batch_size=100):
    global device
    y_hat = torch.zeros(0,device=device)
    loader = torch.utils.data.DataLoader(images,batch_size=batch_size)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
    return y_hat

def findconv(net,includenorm=True):
        layers = pushconv([],net,includenorm)
        return layers

def pushconv(layers,container,includenorm=True):
    if isinstance(container, resnetpy.ResNet):
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
    elif isinstance(container, alexnetpy.AlexNet):
        pushconv(layers,container.features,includenorm)
        pushconv(layers,container.classifier,includenorm)
    elif isinstance(container, torch.nn.Linear):
        layers.append(container)
    elif isinstance(container, torch.nn.Conv2d):
        layers.append(container)
    elif isinstance(container, resnetpy.BasicBlock):
        pushconv(layers,container.conv1,includenorm)
        pushconv(layers,container.bn1,includenorm)
        pushconv(layers,container.conv2,includenorm)
        pushconv(layers,container.bn2,includenorm)
        if isinstance(container.downsample,torch.nn.Sequential):
            pushconv(layers,container.downsample[0],includenorm)
            pushconv(layers,container.downsample[1],includenorm)
    elif isinstance(container, resnetpy.Bottleneck):
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
