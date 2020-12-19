import time
import h5py
import torch
import torch.optim as optim
import numpy as np
import scipy.io as io
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import transconv
import cifar10_models

import resnetpy
import vggpy
import alexnetpy
import densenetpy
import mobilenetpy

device = None

batch_size = 25

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
	[transforms.Resize(256,interpolation=1),
	 transforms.CenterCrop(224),
	 transforms.ToTensor(),
	 transforms.Normalize(rgb_avg, rgb_std)])

transcifar10 = transforms.Normalize([0.4914, 0.4822, 0.4465],\
                                    [0.2023, 0.1994, 0.2010])

def loadvarstats(archname,trantype,testsize):
    mat = io.loadmat(('%s_%s_stats_%d.mat' % (archname, trantype, testsize)))
    return np.array(mat['cov'])

def loadrdcurves(archname,tranname,trantype,l,part):
    mat = io.loadmat('%s_%s_val_%03d_0100_output_%s_%s' % (archname,tranname,l,trantype,part))
    return mat['%s_Y_sse'%part], mat['%s_delta'%part], mat['%s_coded'%part]
    #mat = io.loadmat('%s_%s_val_1000_%d_%d_output_%s_%s' % (archname,tranname,l+1,l+1,trantype,part))
    #return mat['%s_Y_sse'%part][l,0], mat['%s_delta'%part][l,0], mat['%s_coded'%part][l,0]

def findrdpoints(y_sse,delta,coded,lam_or_bit, is_bit=False):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float('inf')
    ind1 = np.nanargmin(y_sse,1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),y_sse.shape) # bit_depth x blocks
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    if is_bit:
        point = coded == lam_or_bit
    else:
        point = y_sse + lam_or_bit*coded == (y_sse + lam_or_bit*coded).min(0)

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

def loadnetwork(archname,gpuid,testsize,dataset='imagenet'):
    global device
    device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")

    if dataset == 'imagenet':
        if archname == 'alexnetpy':
            net = alexnetpy.alexnet(pretrained=True)
        elif archname == 'vgg16py':
            net = vggpy.vgg16_bn(pretrained=True)
        elif archname == 'resnet18py':
            net = resnetpy.resnet18(pretrained=True)
        elif archname == 'resnet34py':
            net = resnetpy.resnet34(pretrained=True)
        elif archname == 'resnet50py':
            net = resnetpy.resnet50(pretrained=True)
        elif archname == 'densenet121py':
            net = densenetpy.densenet121(pretrained=True)
        elif archname == 'mobilenetv2py':
            net = mobilenetpy.mobilenet_v2(pretrained=True)
        images = datasets.ImageNet(root='~/Developer/ILSVRC2012_devkit_t12',\
                                   split='val',transform=transdata)
        images.samples = images.samples[::len(images.samples)//testsize]
        labels = torch.tensor([images.samples[i][1] for i in range(0,len(images))])
    elif dataset == 'cifar10':
        if archname == 'resnet18py':
            net = cifar10_models.resnet.resnet18(pretrained=True)
        elif archname == 'resnet34py':
            net = cifar10_models.resnet.resnet34(pretrained=True)
        elif archname == 'resnet50py':
            net = cifar10_models.resnet.resnet50(pretrained=True)
        elif archname == 'densenet121py':
            net = difar10_models.densenet.densenet121(pretrained=True)
        images = datasets.CIFAR10('~/Developer/',download=True,transform=transcifar10)
        images.data = images.data[::len(images.data)//testsize]
        images.targets = images.targets[::len(images.targets)//testsize]
        labels = torch.tensor([images.targets])
        # load the dataset
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

def getstride(trantype,tranname,archname,tensorsize):
    rows = int(np.ceil(tensorsize[0]/8))
    cols = int(np.ceil(tensorsize[1]/8))

    if tranname == 'idt':
        return min(rows,rows)
    elif tranname == 'klt':
        return min(rows,cols)
    elif tranname == 'ekt':
        return min(rows,cols)

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

def gettopk(logp,k=1):
    logp = logp.exp()
    logp = logp/logp.sum(1).reshape(-1,1)
    vals, inds = logp.topk(k,dim=1)

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

def replaceconv(net,layers,includenorm=True):
    pushconv([layers],net,includenorm,direction=1)
    return net

def hooklayers(net, classes):
    layers = findlayers(net, classes)
    return [Hook(layer) for layer in layers]

def findconv(net,includenorm=True):
    layers = pushconv([[]],net,includenorm)
    return layers

def findlayers(net, classes):
    layers = []
    for m in net.modules():
        if isinstance(m, classes):
            layers.append(m)
    return layers

def pushattr(layers,container,attr,includenorm,direction):
    if isinstance(getattr(container,attr), torch.nn.Linear) or \
       isinstance(getattr(container,attr), torch.nn.Conv2d) or \
       isinstance(getattr(container,attr), transconv.QAConv2d) or \
       isinstance(getattr(container,attr), torch.nn.modules \
                  .batchnorm.BatchNorm2d) and includenorm:
        if direction == 0:
            layers[0].append(getattr(container,attr))
        else:
            setattr(container,attr,layers[0][0])
            layers[0] = layers[0][1:len(layers[0])]

def pushlist(layers,container,attr,includenorm,direction):
    if isinstance(container[attr], torch.nn.Linear) or \
       isinstance(container[attr], torch.nn.Conv2d) or \
       isinstance(container[attr], transconv.QAConv2d) or \
       isinstance(container[attr], torch.nn.modules \
                  .batchnorm.BatchNorm2d) and includenorm:
        if direction == 0:
            layers[0].append(container[attr])
        else:
            container[attr] = layers[0][0]
            layers[0] = layers[0][1:len(layers[0])]
    else:
        pushconv(layers,container[attr],includenorm,direction)

def pushconv(layers,container,includenorm=True,direction=0):
    if isinstance(container,resnetpy.ResNet):
        pushattr(layers,container,'conv1',includenorm,direction)
        pushattr(layers,container,'bn1',includenorm,direction)
        pushconv(layers,container.layer1,includenorm,direction)
        pushconv(layers,container.layer2,includenorm,direction)
        pushconv(layers,container.layer3,includenorm,direction)
        pushconv(layers,container.layer4,includenorm,direction)
        pushattr(layers,container,'fc',includenorm,direction)
    elif isinstance(container,models.densenet.DenseNet):
        pushconv(layers,container.features,includenorm,direction)
        pushattr(layers,container,'classifier',includenorm,direction)
    elif isinstance(container, alexnetpy.AlexNet):
        pushconv(layers,container.features,includenorm,direction)
        pushconv(layers,container.classifier,includenorm,direction)
    elif isinstance(container, vggpy.VGG):
        pushconv(layers,container.features,includenorm,direction)
        pushconv(layers,container.classifier,includenorm,direction)
    elif isinstance(container, resnetpy.BasicBlock):
        pushattr(layers,container,'conv1',includenorm,direction)
        pushattr(layers,container,'bn1',includenorm,direction)
        pushattr(layers,container,'conv2',includenorm,direction)
        pushattr(layers,container,'bn2',includenorm,direction)
        pushconv(layers,container.downsample,includenorm,direction)
    elif isinstance(container, resnetpy.Bottleneck):
        pushattr(layers,container,'conv1',includenorm,direction)
        pushattr(layers,container,'bn1',includenorm,direction)
        pushattr(layers,container,'conv2',includenorm,direction)
        pushattr(layers,container,'bn2',includenorm,direction)
        pushattr(layers,container,'conv3',includenorm,direction)
        pushattr(layers,container,'bn3',includenorm,direction)
        pushconv(layers,container.downsample,includenorm,direction)
    elif isinstance(container, models.densenet._DenseBlock):
        for l in range(0,25):
            if hasattr(container,'denselayer%d'%l):
                pushconv(layers,getattr(container,'denselayer%d'%l),includenorm,direction)
    elif isinstance(container, models.densenet._DenseLayer):
        pushattr(layers,container,'norm1',includenorm,direction)
        pushattr(layers,container,'conv1',includenorm,direction)
        pushattr(layers,container,'norm2',includenorm,direction)
        pushattr(layers,container,'conv2',includenorm,direction)
    elif isinstance(container, models.densenet._Transition):
        pushattr(layers,container,'norm',includenorm,direction)
        pushattr(layers,container,'conv',includenorm,direction)
    elif isinstance(container,torch.nn.Sequential):
        for attr in range(0,len(container)):
            pushlist(layers,container,attr,includenorm,direction)
    # elif isinstance(container, models.mobilenet.ConvBNReLU):
    #      for l in range(0,len(container.conv)):
    #          pushlist(layers,container.conv,attr,includenorm)
    # elif isinstance(container, models.mobilenet.MobileNetV2):
    #     pushconv(layers,container.features,includenorm,direction)
    #     pushconv(layers,container.classifier,includenorm,direction)
    # elif isinstance(container, models.mobilenet.InvertedResidual):
    #     for l in range(0,len(container.conv)):
    #         pushconv(layers,ptrids,container.conv[l],includenorm)
    return layers[0]

def replacelayer(module, layers, classes):
    module_output = module
    # base case
    if isinstance(module, classes):
        module_output, layers[0] = layers[0][0], layers[0][1:]
    # recursive
    for name, child in module.named_children():
        module_output.add_module(name, replacelayer(child, layers, classes))
    del module
    return module_output

def permute(layer,dimen):
    if isinstance(layer, torch.nn.Conv2d):
        return layer.weight.grad.flatten(1).permute([1,0])
    if isinstance(layer, torch.nn.Linear):
        return layer.weight.grad.flatten(1).permute([1,0])
def zerograd(layer):
    layer.weight.grad.zero_()

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = torch.tensor(input[0].shape[1:])
        self.output = torch.tensor(output[0].shape[1:])
    def close(self):
        self.hook.remove()
