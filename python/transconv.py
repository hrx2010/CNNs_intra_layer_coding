import torch
import torch.nn as nn
import common
from header import *

def convert_tconv(layer, base, kern, trantype, block, \
                  kern_coded, kern_delta, base_coded, base_delta, acti_coded, acti_delta, \
                  codekern, codebase, codeacti):

    if trantype == 'inter':
        conv1 = QWConv2d(layer.in_channels,layer.in_channels,kernel_size=1,bias=None,weights=base,\
                         delta=base_delta,coded=base_coded,block=block,is_quantized=codebase) if layer.groups == 1 \
                         else nn.Identity()
        conv2 = QWConv2d(layer.in_channels,layer.out_channels,kernel_size=layer.kernel_size,bias=layer.bias,weights=kern,perm=True,\
                         stride=layer.stride,padding=layer.padding,groups=layer.groups,delta=kern_delta,coded=kern_coded,\
                         block=block,is_quantized=codekern)
    elif trantype == 'exter':
        conv1 = QWConv2d(layer.in_channels,layer.out_channels,kernel_size=layer.kernel_size,bias=None,weights=kern,\
                         stride=layer.stride,padding=layer.padding,groups=layers.groups,delta=kern_delta,coded=kern_coded,\
                         block=block,is_quantized=codekern)
        conv2 = QWConv2d(layer.out_channels,layer.out_channels,kernel_size=1,bias=layer.bias,weights=base,perm=True,\
                         delta=base_delta,coded=base_coded,block=block,is_quantized=codebase) if layer.groups == 1 \
                         else nn.Identity()
    return QAConv2d(nn.Sequential(conv1, conv2), acti_delta, acti_coded, codeacti)

class QAConv2d(nn.Module):
    class Quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, delta, coded):
            return common.quantize(input, 2**delta[0], coded[0]/input[0,:].numel(), centered=True)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None

    def __init__(self, layer, delta, coded, is_quantized=False):
        super(QAConv2d, self).__init__()
        self.quant = self.Quantize.apply
        self.is_quantized = is_quantized
        self.layer = layer
        self.delta = delta
        self.coded = coded
        self.numel = 1

    def forward(self, input):
        self.numel = input[0].numel()
        if self.is_quantized:
            input = self.quant(input, self.delta, self.coded) 
        return self.layer(input)
    
    def extra_repr(self):
        dic = {'depth':[self.coded[0]/self.numel], 'delta':[self.delta[0]], 'quantized':self.is_quantized}
        s = ('bit_depth={depth}, step_size={delta}, quantized={quantized}')
        return self.layer.__repr__() + ', ' + s.format(**dic)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

class QWConv2d(nn.Conv2d):
    class Quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, quant, delta, coded, block, perm):
            quant = quant.clone().permute([1,0,2,3]) if perm else quant.clone()
            for i in range(0,quant.shape[0],block):
                # if delta[i] == Inf:
                #     delta[i] = coded[i] = 0
                rs = range(i,min(i+block,quant.shape[0]))
                quant[rs,:] = common.quantize(quant[rs,:],2**delta[i],coded[i]/quant[rs,:].numel())
            quant = quant.permute([1,0,2,3]) if perm else quant

            return quant

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None, None, None


    def __init__(self, in_channels, out_channels, kernel_size, weights, delta, coded, block,\
                 is_quantized, stride=1, padding=0, groups=1, bias=False, perm=False):
        super(QWConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
                                       bias=True if bias != None else False, groups=groups)
        self.quant = self.Quantize.apply
        self.delta = delta
        self.coded = coded
        self.block = block
        self.perm = perm
        self.is_quantized = is_quantized

        with torch.no_grad():
            self.weight[:] = weights.reshape(self.weight.shape)
            if bias != None:
                self.bias[:] = bias.reshape(self.bias.shape)

        for param in self.parameters():
            param.requires_grad = is_quantized

    def get_bands(self, rs):
        return self.weight[:,rs] if self.perm else self.weight[rs,:]

    def num_bands(self):
        return self.weight.shape[1] if self.perm else self.weight.shape[0]

    def get_bandwidth(self):
        shape = torch.tensor(self.weight.shape)
        return int(torch.ceil(shape[1]/8)) if self.perm else int(torch.ceil(shape[0]/8))

    def extra_repr(self):
        numel = (self.weight.numel() * self.block / (self.weight.shape[1] if self.perm else self.weight.shape[0]))
        depth = [self.coded[i] / numel for i in range(0,len(self.coded),self.block)][0:8]
        delta = [self.delta[i] for i in range(0,len(self.delta),self.block)][0:8]
        s = ('bit_depth={depth}, step_size={delta}, transpose={perm}, quantized={quantized}')
        dic = {'depth':depth, 'delta':delta, 'perm':self.perm, 'quantized':self.is_quantized}
        return super().extra_repr() + ', ' + s.format(**dic)

    def forward(self, input):
        weight = self.quant(self.weight, self.delta, self.coded, self.block, self.perm) if self.is_quantized else self.weight
        return torch.nn.functional.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
