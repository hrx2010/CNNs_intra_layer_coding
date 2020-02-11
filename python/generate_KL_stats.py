import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models
import importlib as imp
import common 
import scipy.io
import numpy as np
import alexnetpy
import resnetpy

imp.reload(common)

from common import *

archname = str(sys.argv[1])
trantype = str(sys.argv[2])
testsize = int(sys.argv[3])

#dims = scipy.io.loadmat(arch+'_dim.mat')['dim'][0]

model, dataset, labels = loadnetwork(archname,gpuid,testsize)
model.eval()
Y = predict(model,dataset)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, trantype, 100*mean_Y_top))


layers = findconv(model,includenorm=False)
cov = [torch.zeros(1).to(common.device)] * len(layers)
iperm,itran = getperm(trantype)

for l in range(0,len(layers)):
    dims = layers[l].weight.flatten(2).permute(iperm).flatten(1).permute(itran).size()
    print('%s | generating  %4d x %4d %s statistics for layer %3d using %d images on gpu %d' % (archname, dims[0], dims[0], trantype, l, len(dataset.samples), gpuid))

iters = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
for x, y in dataloader:
    # get the inputs; data is a list of [inputs, labels]
    x = x.to(common.device)
    y = y.to(common.device)
    y_hat = model(x)
    #y_hat = y_hat/torch.sum(y_hat)
    
    sec = time.time()
    for i in range(0,y_hat.size(1)): #for each class
        for j in range(0,y_hat.size(0)): #for each image
            y_hat[j,i].backward(retain_graph=True) #for each layer
        for l in range(0,len(layers)):
            grad = layers[l].weight.grad.flatten(2).permute(iperm).flatten(1).permute(itran)
            cov[l] = cov[l] + grad.mm(grad.transpose(1,0)).detach()
            grad.zero_()
    iters += y_hat.size(0)
    sec = time.time() - sec
    print('iter %05d, %f sec' % (iters, sec))

for l in range(0,len(layers)):
    cov[l] = cov[l].to('cpu').numpy()

io.savemat(('%s_%s_stats_%d.mat' % (archname, trantype, testsize)),{'cov':cov})
