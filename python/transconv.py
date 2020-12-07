import torch
import torch.nn as nn
import common
from header import *

class TransConv2d(nn.Module):
    def __init__(self, base, kern, bias, stride, padding, trantype, block, \
                 kern_coded, kern_delta, base_coded, base_delta, acti_coded, acti_delta, \
                 codekern, codebase, codeacti):
        super(TransConv2d, self).__init__()

        if trantype == 'inter':
            conv1 = QWConv2d(base.shape[1],base.shape[0],kernel_size=1,bias=None,weights=base,\
                             delta=base_delta,coded=base_coded,block=block,is_quantized=codebase)
            conv2 = QWConv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],bias=bias,weights=kern,perm=True,\
                             stride=stride,padding=padding,delta=kern_delta,coded=kern_coded,\
                             block=block,is_quantized=codekern)
        elif trantype == 'exter':
            conv1 = QWConv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],bias=None,weights=kern,\
                             stride=stride,padding=padding,delta=kern_delta,coded=kern_coded,\
                             block=block,is_quantized=codekern)
            conv2 = QWConv2d(base.shape[1],base.shape[0],kernel_size=1,bias=bias,weights=base,perm=True,\
                             delta=base_delta,coded=base_coded,block=block,is_quantized=codebase)

        self.quant = QAConv2d(nn.Sequential(conv1, conv2), acti_delta, acti_coded, codeacti)

    def forward(self, x):
        return self.quant(x)

    # def quantize(self):
    #     self.quant.layer[0].quantize()
    #     self.quant.layer[1].quantize()

class QAConv2d(nn.Module):
    class Quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, delta, depth):
            return common.quantize(input, 2**delta[0], depth[0])

        def backward(ctx, grad_output):
            return grad_output, None, None

    def __init__(self, layer, delta, depth, quantized=False):
        super(QAConv2d, self).__init__()
        self.quant = self.Quantize.apply
        self.quantized = quantized
        self.layer = layer
        self.delta = delta
        self.depth = depth

    def forward(self, input):
        if self.quantized:
            input = self.quant(input, self.delta, self.depth) 
        return self.layer(input)
    
    def extra_repr(self):
        s = ('bit_depth={depth}, step_size={delta}, quantized={quantized}')
        return self.layer.__repr__() + ', ' + s.format(**self.__dict__)

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
        def forward(ctx, quant, delta, coded, block, perm, inplace=False):
            quant = quant.permute([1,0,2,3]) if perm else quant
            for i in range(0,quant.shape[0],block):
                if delta[i] == Inf:
                    continue
                rs = range(i,min(i+block,quant.shape[0]))
                quant[rs,:] = common.quantize(quant[rs,:],2**delta[i],coded[i]/quant[rs,:].numel())
            quant = quant.permute([1,0,2,3]) if perm else quant

            return quant

    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


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

    def forward(self, input):
        weight = self.quant(self.weight, self.delta, self.coded, self.block, self.perm) if self.is_quantized else self.weight
        return torch.nn.functional.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def quantize(self):
        if self.is_quantized:
            self.quant(self.weight,self.delta,self.coded,self.block,self.perm,True)


