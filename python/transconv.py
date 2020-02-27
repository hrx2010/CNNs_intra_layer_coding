import torch
import torch.nn as nn
from common import *

class TransConv2d(nn.Module):
    def __init__(self, base, kern, bias, stride, padding, trantype, skip, \
                 kern_coded, kern_delta, base_coded, base_delta, codekern, codebase):
        super(TransConv2d, self).__init__()
        self.stride = skip
        if trantype == 'inter':
            self.conv1 = nn.Conv2d(base.shape[1],base.shape[0],kernel_size=1,bias=False)
            self.conv2 = nn.Conv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],\
                                   stride=stride,padding=padding)
            with torch.no_grad():
                self.conv1.weight[:] = base.reshape(self.conv1.weight.shape)
                self.conv2.weight[:] = kern.reshape(self.conv2.weight.shape)
                self.conv2.bias = bias
                self.conv1_delta = base_delta
                self.conv2_delta = kern_delta
                self.conv1_coded = base_coded
                self.conv2_coded = kern_coded
                self.codeconv1 = codebase
                self.codeconv2 = codekern
        elif trantype == 'exter':
            self.conv1 = nn.Conv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],\
                                   stride=stride,padding=padding,bias=False)
            self.conv2 = nn.Conv2d(base.shape[1],base.shape[0],kernel_size=1)
            with torch.no_grad():
                self.conv1.weight[:] = kern.reshape(self.conv1.weight.shape)
                self.conv2.weight[:] = base.reshape(self.conv2.weight.shape)
                self.conv2.bias[:] = bias
                self.conv1_delta = kern_delta
                self.conv2_delta = base_delta
                self.conv1_coded = kern_coded
                self.conv2_coded = base_coded
                self.codeconv1 = codekern
                self.codeconv2 = codebase 

    def forward(self, x):
        with torch.no_grad():
            for i in range(0,self.conv1.weight.shape[0],self.stride):
                rs = range(i,min(i+self.stride,self.conv1.weight.shape[0]))
                scale = (self.conv1.weight[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -24.0:
                    self.conv1.weight[rs,:] = 0.0
                if self.codeconv1:
                    self.conv1.weight[rs,:] = quantize(self.conv1.weight[rs,:],2**self.conv1_delta[i],\
                                                       self.conv1_coded[i]/self.conv1.weight[rs,:].numel())
                scale = (self.conv2.weight[:,rs].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -24.0:
                    self.conv2.weight[:,rs] = 0.0
                if self.codeconv2:
                    self.conv2.weight[:,rs] = quantize(self.conv2.weight[:,rs],2**self.conv2_delta[i],\
                                                       self.conv2_coded[i]/self.conv2.weight[:,rs].numel())
        x = self.conv1(x)
        x = self.conv2(x)
        return x
