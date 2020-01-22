import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models
import importlib

rgb_avg = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

transdata = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(rgb_avg, rgb_std)])

dataset = torchvision.datasets.ImageNet(root='/media/data2/seany/ILSVRC2012_devkit_t12',split='val',transform=transdata)
#[dataset,restset] = torch.utils.data.random_split(dataset, [1000,49000])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=25)

device = torch.device("cuda:%d"%(0) if torch.cuda.is_available() else "cpu")
net = torchvision.models.vgg16(pretrained=True)
net.to(device)
net.eval()

valid_top1 = 0
tic = time.time()
for inputs, labels in dataloader:
    # get the inputs; data is a list of [inputs, labels]
    inputs = inputs.to(device)
    labels = labels.to(device)
    logpd = net.forward(inputs)
    prob, cats = torch.exp(logpd).topk(1, dim=1)
    valid_top1 += torch.sum(cats == labels.view(*cats.shape)).item()
    #print('[%s %05d] loss: %.3f train_top1: %.3f valid_top1: %.3f' %(archname, iteration, train_loss.item(),
    #train_top1, valid_top1))
valid_top1 /= len(dataloader.dataset)
toc = time.time() - tic
