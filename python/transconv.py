import torch
import torch.nn as nn
import common
from header import *

class TransConv2d(nn.Module):
    def __init__(self, base, kern, bias, stride, padding, trantype, block, \
                 kern_coded, kern_delta, base_coded, base_delta, codekern, codebase):
        super(TransConv2d, self).__init__()

        if trantype == 'inter':
            self.conv1 = QuantConv2d(base.shape[1],base.shape[0],kernel_size=1,bias=False,\
                                     delta=base_delta,coded=base_coded,block=block,is_coded=codebase)
            self.conv2 = QuantConv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],perm=True,\
                                     stride=stride,padding=padding,delta=kern_delta,coded=kern_coded,\
                                     block=block,is_coded=codekern)
            with torch.no_grad():
                self.conv1.weight[:] = base.reshape(self.conv1.weight.shape)
                self.conv2.weight[:] = kern.reshape(self.conv2.weight.shape)
                self.conv2.bias = bias
        elif trantype == 'exter':
            self.conv1 = QuantConv2d(kern.shape[1],kern.shape[0],kernel_size=kern.shape[2],bias=False,\
                                     stride=stride,padding=padding,delta=kern_delta,coded=kern_coded,\
                                     block=block,is_coded=codekern)
            self.conv2 = QuantConv2d(base.shape[1],base.shape[0],kernel_size=1,perm=True,\
                                     delta=base_delta,coded=base_coded,block=block,is_coded=codebase)
            with torch.no_grad():
                self.conv1.weight[:] = kern.reshape(self.conv1.weight.shape)
                self.conv2.weight[:] = base.reshape(self.conv2.weight.shape)
                self.conv2.bias = bias

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)

        return x

    def quantize(self):
        self.conv1.quantize()
        self.conv2.quantize()


class QConv2d(nn.Module):
    class Quantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, delta, depth):
            return common.quantize(input, 2**delta, depth)

        def backward(ctx, grad_output):
            return grad_output, None, None

    def __init__(self, layer, delta, depth, lossy=False):
        super(QConv2d, self).__init__()
        self.quant = self.Quantize.apply
        self.layer = layer
        self.delta = delta
        self.depth = depth
        self.lossy = lossy

    def extra_repr(self):
        s = ('delta={delta}, depth={depth}, lossy={lossy}')
        return self.layer.extra_repr() + ', ' + s.format(**self.__dict__)

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

    def forward(self, input):
        input = self.quant(input, self.delta, self.depth) if self.lossy else input
        return self.layer(input)
    
class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, delta, coded, block,\
                 is_coded, stride=1, padding=0, bias=False, perm=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size,\
                                          stride=stride,padding=padding,bias=bias)
        self.delta = delta
        self.coded = coded
        self.block = block
        self.perm = perm
        self.is_coded = is_coded
        self.quant = Quantize.apply

        if not self.is_coded:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        if self.is_coded:
            return self.conv2d_forward(input, self.quant(self.weight,self.delta,self.coded,\
                                                         self.block,self.perm))
        else:
            return self.conv2d_forward(input, self.weight)

    def quantize(self):
        if self.is_coded:
            self.quant(self.weight,self.delta,self.coded,self.block,self.perm,True)


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, delta, coded, block, perm, inplace=False):
        quant = weight.clone()
        if perm:
            quant = quant.permute([1,0,2,3])
        for i in range(0,quant.shape[0],block):
            rs = range(i,min(i+block,quant.shape[0]))
            scale = (quant[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
            if coded[i] == Inf:
                quant[rs,:] = 0
                continue
            quant[rs,:] = common.quantize(quant[rs,:],2**delta[i],coded[i]/quant[rs,:].numel())
        if perm:
            quant = quant.permute([1,0,2,3])

        if inplace:
            weight[:] = quant
        return quant

    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None
