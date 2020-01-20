import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models
import importlib as imp
import findconv 
import scipy.io
import numpy as np

imp.reload(findconv)

arch = 'alexnet'
#dims = scipy.io.loadmat(arch+'_dim.mat')['dim'][0]

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
    [transforms.Resize(256,interpolation=1),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(rgb_avg, rgb_std)])

gpuid = int(sys.argv[1])
testsize = 2500
dataset = torchvision.datasets.ImageNet(root='/media/data2/seany/ILSVRC2012_devkit_t12',split='val',transform=transdata)
dataset.samples = dataset.samples[testsize*gpuid+0:testsize*gpuid+testsize]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
net = torchvision.models.resnet50(pretrained=True)
layers = findconv.findconv(net)

net.to(device)
net.eval()

avg = [torch.zeros(1).to(device)] * len(layers)
cov = [torch.zeros(1).to(device)] * len(layers)

iters = 0
for x, y in dataloader:
    # get the inputs; data is a list of [inputs, labels]
    x = x.to(device)
    y = y.to(device)
    y_hat = torch.exp(net(x))
    y_hat = y_hat/torch.sum(y_hat)
    
    for i in range(0,y_hat.size(1)): #for each class
        for j in range(0,y_hat.size(0)): #for each image
            y_hat[j,i].backward(retain_graph=True) #for each layer
        for l in range(0,len(layers)):
            grad = layers[l].weight.grad.flatten(1).permute([1,0])
            avg[l] = avg[l] + grad.detach()
            cov[l] = cov[l] + grad.mm(grad.transpose(1,0)).detach()
            grad.zero_()
    iters += y_hat.size(0)
    print(iters)

scipy.io.savemat(arch + '_stats_' + str(gpuid) + '.mat',{'avg':avg,'cov':cov})
