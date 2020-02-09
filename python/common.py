import time
import h5py
import torch
import scipy.io
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import alexnetpy

device = None

batch_size = 225

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
	[transforms.Resize(256,interpolation=1),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(rgb_avg, rgb_std)])

def loadnetwork(archname,gpuid,testsize):
    global device
    if archname == 'alexnetpy':
        net = alexnetpy.alexnet(pretrained=True)
    elif archname == 'resnet18py':
        net = models.resnet18(pretrained=True)
    elif archname == 'resnet50py':
        net = models.resnet50(pretrained=True)
    elif archname == 'mobilenetv2py':
        net = models.mobilenet.mobilenet_v2(pretrained=True)
        # load the dataset
    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
    images = datasets.ImageNet(\
                root='~/Developer/ILSVRC2012_devkit_t12',\
                split='val',transform=transdata)
    images.samples = images.samples[::len(images.samples)//testsize]
    labels = torch.tensor([img[1] for img in images])

    return net.to(device), images, labels.to(device)

def loadstrides(archname,tranname):
    if archname == 'alexnetpy':
        if tranname == 'inter':
            return [1,8,8,8,8,8,8,8]

def gettrans(archname,trantype,tranname,layer):
	file = h5py.File('%s_%s_50000_%s.mat' %(archname,tranname,trantype))
	return torch.FloatTensor(file[file['T'][0,layer]]).to(device).permute([3,2,1,0])

def getdevice():
	global device
	return device

def getperm(trantype):
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
    
def quantize(weights, delta, b):
    if b > 0:
        minpoint = -(2**(b-1))*delta
        maxpoint = +(2**(b-1))*delta
    else:
        minpoint = 0
        maxpoint = 0

    return delta*(weights//delta).clamp(minpoint,maxpoint)
    
def gettop1(logp):
    logp = logp.exp()
    logp = logp/logp.sum(1).reshape(-1,1)
    vals, inds = logp.max(1)

    return inds
        
def predict(net,images):
    global device
    y_hat = torch.zeros(0,device=device)
    loader = torch.utils.data.DataLoader(images,batch_size=100)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            #y = y.to(device)
            y_hat = torch.cat((y_hat,net(x)))
    return y_hat

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
    elif isinstance(container, alexnetpy.AlexNet):
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
