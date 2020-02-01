import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models
import findconv

weight = []
biases = []
bnorm_vars = []
bnorm_mean = []

net = models.mobilenet.mobilenet_v2(pretrained=True)
layers = findconv.findconv(net)

for l in range(0,len(layers)):
    if isinstance(layers[l], torch.nn.Conv2d):
        weight.append(np.array(layers[l].weight.detach()))
        biases.append(np.array(layers[l].bias.detach()))
        bnorm_vars.append(np.array(0))
        bnorm_mean.append(np.array(0))
    if isinstance(layers[l], torch.nn.Linear):
        weight.append(np.array(layers[l].weight.detach()))
        biases.append(np.array(layers[l].bias.detach()))
        bnorm_vars.append(np.array(0))
        bnorm_mean.append(np.array(0))
    if isinstance(layers[l], torch.nn.modules.batchnorm.BatchNorm2d):
        weight.append(np.array(layers[l].weight.detach()))
        biases.append(np.array(layers[l].bias.detach()))
        bnorm_vars.append(np.array(layers[l].running_var.detach()))
        bnorm_mean.append(np.array(layers[l].running_mean.detach()))

io.savemat('mobilenetv2.mat',{'weight':weight, 'biases':biases, 'bnorm_vars':bnorm_vars, 'bnorm_mean':bnorm_mean})
														
