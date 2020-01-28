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

if arch == 'alexnet':
    net = torchvision.models.alexnet(pretrained=True)
elif arch == 'resnet50':
    net = torchvision.models.resnet50(pretrained=True)

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
    [transforms.Resize(256,interpolation=1),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(rgb_avg, rgb_std)])

testsize = int(sys.argv[2])
gpuid = int(sys.argv[3])
device = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
if arch == 'alexnet':
    net = torchvision.models.alexnet(pretrained=True)
    batchsize = 10
elif arch == 'resnet50':
    net = torchvision.models.resnet50(pretrained=True)
    batchsize = 1
elif arch == 'mobilenetv2':
    net = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
    batchsize = 1
layers = findconv.findconv(net)

dataset = torchvision.datasets.ImageNet(root='/media/data2/seany/ILSVRC2012_devkit_t12',split='val',transform=transdata)
dataset.samples = dataset.samples[testsize*gpuid+0:testsize*gpuid+testsize]

net.to(device)
net.eval()

avg = [torch.zeros(1).to(device)] * len(layers)
cov = [torch.zeros(1).to(device)] * len(layers)

print('%s | generating joint statistics for %03d layers using %d images' % (arch, len(layers), testsize));


iters = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize)
for x, y in dataloader:
    # get the inputs; data is a list of [inputs, labels]
    x = x.to(device)
    y = y.to(device)
    y_hat = torch.exp(net(x))
    y_hat = y_hat/torch.sum(y_hat)
    
    for i in range(0,y_hat.size(1)): #for each class
        sec = time.time()
        for j in range(0,y_hat.size(0)): #for each image
            y_hat[j,i].backward(retain_graph=True) #for each layer
        for l in range(0,len(layers)):
            grad = layers[l].weight.grad.flatten(1).permute([1,0])
            avg[l] = avg[l] + grad.detach()
            cov[l] = cov[l] + grad.mm(grad.transpose(1,0)).detach()
            grad.zero_()
        #print(time.time() - sec)
    iters += y_hat.size(0)
    print(iters)

for l in range(0,len(layers)):
    avg[l] = avg[l].to('cpu').numpy()
    cov[l] = cov[l].to('cpu').numpy()

scipy.io.savemat('/media/data2/seany/' + arch + '_' + str(testsize) + '_' + str(gpuid) + '.mat',{'avg':avg,'cov':cov})
