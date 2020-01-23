import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models

weight = []
biases  = []
net = models.alexnet(pretrained=True)

for l in range(0,len(net.features)):
    if isinstance(net.features[l], torch.nn.Conv2d):
        weight.append(np.array(net.features[l].weight.detach()))
        biases.append(np.array(net.features[l].bias.detach()))

for l in range(0,len(net.classifier)):
    if isinstance(net.classifier[l], torch.nn.Linear):
        weight.append(np.array(net.classifier[l].weight.detach()))
        biases.append(np.array(net.classifier[l].bias.detach()))

weight[5] = weight[5].reshape(4096,256,6,6).swapaxes(2,3).reshape(4096,6*6*256)
io.savemat('alexnetv2.mat',{'weight':weight, 'biases':biases})
														
