import numpy as np
import scipy.io as io
import torch.nn
import torchvision.models as models
import findconv

weight = []
biases = []
stride = []
padding = []
bnorm_vars = []
bnorm_mean = []

net = models.resnet.resnet18(pretrained=True)
layers = findconv.findconv(net)

for l in range(0,len(layers)):
    if isinstance(layers[l], torch.nn.Conv2d):
        weight.append(np.array(layers[l].weight.detach()))
        stride.append(np.array(layers[l].stride))
        padding.append(np.array(layers[l].padding))
        if isinstance(layers[l].bias,torch.nn.parameter.Parameter):
            biases.append(np.array(layers[l].bias.detach()))
        else:
            biases.append(np.zeros(layers[l].out_channels))
        bnorm_vars.append(np.zeros(layers[l].out_channels))
        bnorm_mean.append(np.zeros(layers[l].out_channels))
    if isinstance(layers[l], torch.nn.Linear):
        weight.append(np.array(layers[l].weight.detach()))
        stride.append(np.array([0,0]))
        padding.append(np.array([0,0]))
        if isinstance(layers[l].bias,torch.nn.parameter.Parameter):
            biases.append(np.array(layers[l].bias.detach()))
        else:
            biases.append(np.zeros(layers[l].out_features))
        bnorm_vars.append(np.zeros(layers[l].out_features))
        bnorm_mean.append(np.zeros(layers[l].out_features))
    if isinstance(layers[l], torch.nn.modules.batchnorm.BatchNorm2d):
        weight.append(np.array(layers[l].weight.detach()))
        stride.append(np.array([0,0]))
        padding.append(np.array([0,0]))
        if isinstance(layers[l].bias,torch.nn.parameter.Parameter):
            biases.append(np.array(layers[l].bias.detach()))
        else:
            biases.append(np.zeros(layers[l].out_channels))
        bnorm_vars.append(np.array(layers[l].running_var.detach()))
        bnorm_mean.append(np.array(layers[l].running_mean.detach()))

io.savemat('resnet18.mat',{'weight':weight, 'biases':biases, 'stride':stride, 'padding':padding, 'bnorm_vars':bnorm_vars, 'bnorm_mean':bnorm_mean})
														
