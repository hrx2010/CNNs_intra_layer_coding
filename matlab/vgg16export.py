import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models

weight = []
biases  = []
net = models.vgg.vgg16(pretrained=True)

for l in range(0,len(net.features)):
    if isinstance(net.features[l], torch.nn.Conv2d):
        weight.append(np.array(net.features[l].weight.detach()))
        biases.append(np.array(net.features[l].bias.detach()))

for l in range(0,len(net.classifier)):
    if isinstance(net.classifier[l], torch.nn.Linear):
        weight.append(np.array(net.classifier[l].weight.detach()))
        biases.append(np.array(net.classifier[l].bias.detach()))

weight[13] = weight[13].reshape(4096,512,7,7).swapaxes(2,3).reshape(4096,7*7*512)
io.savemat('vgg16.mat',{'weight':weight, 'biases':biases})
														
