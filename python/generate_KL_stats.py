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

arch = sys.argv[1]#'alexnet'
#dims = scipy.io.loadmat(arch+'_dim.mat')['dim'][0]

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
    [transforms.Resize(256,interpolation=1),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(rgb_avg, rgb_std)])

xform = sys.argv[2]
testsize = int(sys.argv[3])
#gpuid = int(sys.argv[4])
parts = 1#int(sys.argv[5])

device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
if arch == 'alexnetpy':
    net = torchvision.models.alexnet(pretrained=True)
    batchsize = 10
elif arch == 'resnet50py':
    net = torchvision.models.resnet50(pretrained=True)
    batchsize = 10
elif arch == 'mobilenetv2py':
    net = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
    batchsize = 10
elif arch == 'resnet18py':
    net = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
    batchsize = testsize//1000

layers = findconv.findconv(net,includenorm=False)

dataset = torchvision.datasets.ImageNet(root='~/Developer/ILSVRC2012_devkit_t12',split='val',transform=transdata)
dataset.samples = dataset.samples[::len(dataset.samples)//testsize]
dataset.samples = dataset.samples[np.mod(gpuid,parts)::parts]

net.to(device)
net.eval()

avg = [torch.zeros(1).to(device)] * len(layers)
cov = [torch.zeros(1).to(device)] * len(layers)

if   xform == 'inter':
    iperm = [1,0,2]
    itran = [0,1]
elif xform == 'intra':
    iperm = [2,0,1]
    itran = [0,1]
elif xform == 'exter':
    iperm = [0,1,2]
    itran = [0,1]
elif xform == 'joint':
    iperm = [0,1,2]
    itran = [1,0]
elif xform == 'extra':
    iperm = [1,0,2]
    itran = [1,0]

#print('%s | generating %s statistics for %03d layers using %d images' % (arch, len(layers), testsize));

sz = [0] * len(layers)
for l in range(0,len(layers)):
    sz[l] = torch.tensor(layers[l].weight.size())

if arch == 'alexnet':
    sz[5] = torch.tensor([4096,256,6,6])
elif arch == 'resnet50':
    sz[53] = torch.tensor([1000,512,2,2])

for l in range(0,len(layers)):
    dims = layers[l].weight.reshape(sz[l][0],sz[l][1],-1).permute(iperm).flatten(1).permute(itran).size()
    print('%s | generating  %4d x %4d %s statistics for layer %d using %d images on gpu %d' % (arch, dims[0], dims[0], xform, l, len(dataset.samples), gpuid))

iters = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize)
for x, y in dataloader:
    # get the inputs; data is a list of [inputs, labels]
    x = x.to(device)
    y = y.to(device)
    y_hat = net(x)
    #y_hat = y_hat/torch.sum(y_hat)
    
    sec = time.time()
    for i in range(0,y_hat.size(1)): #for each class
        for j in range(0,y_hat.size(0)): #for each image
            y_hat[j,i].backward(retain_graph=True) #for each layer
        for l in range(0,len(layers)):
            grad = layers[l].weight.grad.reshape(sz[l][0],sz[l][1],-1).permute(iperm).flatten(1).permute(itran)
            #avg[l] = avg[l] + grad.detach()
            cov[l] = cov[l] + grad.mm(grad.transpose(1,0)).detach()
            grad.zero_()
        #print(time.time() - sec)
    iters += y_hat.size(0)
    sec = time.time() - sec
    print('iter %05d, %f sec' % (iters, sec))

for l in range(0,len(layers)):
    avg[l] = avg[l].to('cpu').numpy()
    cov[l] = cov[l].to('cpu').numpy()

scipy.io.savemat('/media/data2/seany/' + arch + '_' + xform +  '_stats_' + str(testsize) + '_' + str(gpuid) + '.mat',{'avg':avg,'cov':cov})
