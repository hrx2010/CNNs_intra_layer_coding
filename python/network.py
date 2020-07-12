import transconv
import common
import importlib
from common import *

importlib.reload(transconv)

def trans1d(layers,l,trantype,tranname,archname):
    with torch.no_grad():
        layer_weights = layers[l].weight

        basis_permute = getperm(trantype)[0]
        basis_vectors = gettrans(archname,trantype,tranname,l,'').flatten(2)
        layer_weights = layer_weights.flatten(2).permute(basis_permute)
        dimen_weights = layer_weights.size()
        layer_weights = layer_weights.flatten(1)
        trans_weights = basis_vectors[:,:,0].mm(layer_weights)
        basis_vectors = basis_vectors[:,:,1].permute(inv(basis_permute[0:2]))
        trans_weights = trans_weights.reshape(dimen_weights).permute(inv(basis_permute))\
                                     .reshape(layers[l].weight.size())
        return transconv.Trans1dConv2d(basis_vectors,trans_weights,layers[l].bias,layers[l].stride,\
                                       layers[l].padding,trantype)

def trans2d(layers,l,tranname,archname):
    with torch.no_grad():
        layer_weights = layers[l].weight

        inter_permute = getperm('inter')[0]
        inter_vectors = gettrans(archname,'inter',tranname,l,'').flatten(2)
        layer_weights = layer_weights.flatten(2).permute(inter_permute)
        dimen_weights = layer_weights.size()
        layer_weights = layer_weights.flatten(1)
        layer_weights = inter_vectors[:,:,0].mm(layer_weights)
        inter_vectors = inter_vectors[:,:,1].permute(inv(inter_permute[0:2]))
        layer_weights = layer_weights.reshape(dimen_weights).permute(inv(inter_permute))\
                                     .reshape(layers[l].weight.size())
        
        exter_permute = getperm('exter')[0]
        exter_vectors = gettrans(archname,'exter',tranname,l,'').flatten(2)
        layer_weights = layer_weights.flatten(2).permute(exter_permute)
        dimen_weights = layer_weights.size()
        layer_weights = layer_weights.flatten(1)
        layer_weights = exter_vectors[:,:,0].mm(layer_weights)
        exter_vectors = exter_vectors[:,:,1].permute(inv(exter_permute[0:2]))
        layer_weights = layer_weights.reshape(dimen_weights).permute(inv(exter_permute))\
                                     .reshape(layers[l].weight.size())
        
        return transconv.Trans2dConv2d(inter_vectors,layer_weights,exter_vectors,\
                                       layers[l].bias,layers[l].stride,layers[l].padding)

def transform2d(network,tranname,archname):
    layers = findconv(network,False)
    for l in range(0,len(layers)):
        if layers[l].weight.shape[2] == 1:
            layers[l] = trans1d(layers,l,'inter',tranname,archname)
        else:
            layers[l] = trans2d(layers,l,tranname,archname)

    network = replaceconv(network,layers,includenorm=False)

    return network.to(common.device)

def transform1d(network,tranname,archname):
    layers = findconv(network,False)
    for l in range(0,len(layers)):
        layers[l] = trans1d(layers,l,tranname,archname)

    network = replaceconv(network,layers,includenorm=False)

    return network.to(common.device)

def transform(network,trantype,tranname,archname,rdlambda,codekern,codebase):
    layers = findconv(network,False)
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
            
            stride = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))
            basis_vectors = basis_vectors[:,:,1].permute(inv(perm[0:2]))
            trans_weights = trans_weights.permute(inv(flip)).reshape(dimen_weights).permute(inv(perm))\
                                                                                   .reshape(layers[l].weight.size())
            layers[l] = transconv.QuantTransConv2d(basis_vectors,trans_weights,layers[l].bias,layers[l].stride,layers[l].padding,\
                                              trantype,stride,kern_coded,kern_delta,base_coded,base_delta,codekern,codebase)
        network = replaceconv(network,layers,includenorm=False)

    return network.to(common.device)

def quantize(network):
    layers = findconv(network,False)
    with torch.no_grad():
        for l in range(0,len(layers)):
            layers[l].quantize()

    return network.to(common.device)

