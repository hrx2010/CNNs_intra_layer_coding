import time
import h5py
import torch
import scipy as np
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

def loadrdpoints(archname,tranname,trantype,l):
    mat = io.loadmat(sprintf('%s_%s_val_%d_0100_output_%s_%s_kern',\
                             archname,tranname,l,trantype))
    return mat['kern_Y_sse'], mat['kern_delta'], mat['kern_coded']

def findrdcurves(y_sse,delta,coded):
    ind1 = np.argmin(y_sse,1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),y_sse.shape)




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
    labels = torch.tensor([images.samples[i][1] for i in range(0,len(images.samples))])

    return net.to(device), images, labels.to(device)

def gettrans(archname,trantype,tranname,layer):
	file = h5py.File('%s_%s_50000_%s.mat' %(archname,tranname,trantype))
	return torch.FloatTensor(file[file['T'][0,layer]]).to(device).permute([3,2,1,0])

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

    return delta*(weights//delta).clamp(minpoint,maxpoint)
    
def min_inds(mat,axis):
    ind1 = np.argmin(mat,axis)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),mat.shape)

    return inds

def lambda2points(X,Y,Z,lam):
    X = X.flatten(1)
    Y = Y.flatten(1)
    Z = Z.flatten(1)
    points = np.zeros(Z.shape[1])

    for i in range(0,X.shape[1]):
        if np.all(np.isinf(X[:,i])):
            continue
        slopes = np.append(np.diff(Y[:,i])/np.diff(X[:,i]),0.0)
        points[i] = Z[:,i].argwhere(slopes<lam)[0]

    return points


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
        for x, y in loader:
            x = x.to(device)
            #y = y.to(device)
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
