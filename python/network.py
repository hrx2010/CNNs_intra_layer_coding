import torch.nn as nn
import transconv
import common
from common import *

def transform(network,trantype,tranname,archname,bitdepth,rdlambda,codekern,codebase,codeacti):
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

def convert_qconv(network):
    layers = findlayers(network, (nn.Conv2d))

    with torch.no_grad():
        for l in range(0,len(layers)):
            layers[l] = transconv.QAConv2d(layers[l], [0], [0])
        
        network = replacelayer(network, [layers], (nn.Conv2d))
    return network.to(common.device)
