import torch.nn as nn
import transconv
import common
from common import *
from header import *

def transform(network,trantype,tranname,archname,rdlambda,codekern,codebase,codeacti):
    layers = findlayers(network,(nn.Conv2d))
    perm, flip = getperm(trantype)

    with torch.no_grad():
        for l in range(0,len(layers)):
            basis_vectors = gettrans(archname,trantype,tranname,l,'').flatten(2)
            layer_weights = layers[l].weight
            layer_weights = layer_weights.flatten(2).permute(perm)
            dimen_weights = layer_weights.size()
            layer_weights = layer_weights.flatten(1).permute(flip)
            trans_weights = basis_vectors[:,:,0].mm(layer_weights)
            ##load files here
            kern_delta = kern_coded = []
            if codekern:
                kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,tranname,trantype,l, 'kern')
                kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, 2**rdlambda)
            base_delta = base_coded = []
            if codebase:
                base_Y_sse, base_delta, base_coded = loadrdcurves(archname,tranname,trantype,l, 'base')
                base_Y_sse, base_delta, base_coded = findrdpoints(base_Y_sse,base_delta,base_coded, 2**rdlambda)
            acti_delta = acti_coded = []
            if codeacti:
                acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname,tranname,trantype,l, 'acti')
                acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**rdlambda)

            block = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))
            basis_vectors = basis_vectors[:,:,1].permute(inv(perm[0:2]))
            trans_weights = trans_weights.permute(inv(flip)).reshape(dimen_weights).permute(inv(perm)).reshape(layers[l].weight.size())
            layers[l] = transconv.convert_tconv(layers[l],basis_vectors,trans_weights,\
                                                trantype,block,kern_coded,kern_delta,base_coded,base_delta,acti_coded,acti_delta,\
                                                codekern,codebase,codeacti)
        network = replacelayer(network,[layers], (nn.Conv2d))

    return network.to(common.device)

def quantize_slope_2d(neural, archname, slope, codekern, codeacti, a_dimens=None):
    w_layers = findlayers(neural,transconv.QWConv2d)
    a_layers = findlayers(neural,transconv.QAConv2d)

    pred_sum_Y_sse = hist_sum_coded = hist_sum_denom = 0
    for l in range(0,max(len(w_layers),len(a_layers))):
        if codekern:
            hist_sum_denom = hist_sum_denom + w_layers[l].weight.numel()
            kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,'idt','inter',l, 'kern')
            kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, 2**slope)
            stride = w_layers[l].get_bandwidth()
            for i in range(0,w_layers[l].num_bands(),stride):
                w_layers[l].is_quantized = True
                w_layers[l].delta[i] = kern_delta[i]
                w_layers[l].coded[i] = kern_coded[i]
                pred_sum_Y_sse = pred_sum_Y_sse + kern_Y_sse[i]
                hist_sum_coded = hist_sum_coded + kern_coded[i]

        if codeacti:
            hist_sum_denom = hist_sum_denom + a_dimens[l]
            acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname,'idt','inter',l, 'acti')
            acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope)

            for i in range(0,1):#a_layers[l].num_bands(),stride):
                a_layers[l].is_quantized = True
                a_layers[l].coded[i] = acti_coded[i]
                a_layers[l].delta[i] = acti_delta[i]
                pred_sum_Y_sse = pred_sum_Y_sse + acti_Y_sse[i]
                hist_sum_coded = hist_sum_coded + acti_coded[i]

    return pred_sum_Y_sse, hist_sum_coded, hist_sum_denom


def convert_qconv(network):
    layers = findlayers(network, (nn.Conv2d))

    with torch.no_grad():
        for l in range(0,len(layers)):
            layers[l] = transconv.QAConv2d(layers[l], [0], [0])
        
        network = replacelayer(network, [layers], (nn.Conv2d))
    return network.to(common.device)


def quantize_2d(network):
    network.features[0][0] = convert_qwconv(network.features[0][0], False)
    network.features[1].conv[0][0] = convert_qwconv(network.features[1].conv[0][0], False)
    network.features[1].conv[1] = convert_qwconv(network.features[1].conv[1], True)

    for i in range(2,len(network.features) - 1):
        for j in range(2):
            network.features[i].conv[j][0] = convert_qwconv(network.features[i].conv[j][0], False)
        network.features[i].conv[2] = convert_qwconv(network.features[i].conv[2], True)

    network.features[-1][0] = convert_qwconv(network.features[-1][0], False)
    network.classifier = convert_qwconv(network.classifier, False)

def convert_qwconv(conv, perm):
    size = conv.in_channels if perm else conv.out_channels
    conv = transconv.QWConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.weight,\
                                                [Inf]*size, [Inf]*size, size//8, False,\
                                                conv.stride, conv.padding, conv.groups, conv.bias, perm)
    return transconv.QAConv2d(conv, delta=[Inf], coded=[Inf], is_quantized=False)
