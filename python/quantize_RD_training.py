import common
import header
import transconv
import importlib
importlib.reload(common)
importlib.reload(header)
importlib.reload(transconv)

from common import *
from header import *
from transconv import *

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])
rdlambda = float(sys.argv[5])
codebase = True if len(sys.argv) < 7 else int(sys.argv[6])
codekern = True if len(sys.argv) < 8 else int(sys.argv[7])
gpuid   = gpuid if len(sys.argv) < 9 else int(sys.argv[8])

tarnet, images, labels = loadnetwork(archname,gpuid,testsize)
tarnet.eval()
Y = predict(tarnet,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

layers = findconv(tarnet,False)
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
        trans_weights = trans_weights.permute(inv(flip)).reshape(dimen_weights).permute(inv(perm)).reshape(layers[l].weight.size())
        layers[l] = transconv.TransConv2d(basis_vectors,trans_weights,layers[l].bias,layers[l].stride,layers[l].padding,\
                                          trantype,stride,kern_coded,kern_delta,base_coded,base_delta,codekern,codebase)

    tarnet = replaceconv(tarnet,layers,includenorm=False).to(common.device)

Y_hats = predict(tarnet,images)
Y_cats = gettop1(Y_hats)
hist_sum_Y_sse = ((Y_hats - Y)**2).mean()
hist_sum_Y_top = (Y_cats == labels).double().mean()

print('%s %s | slope: %+5.1f, top1: %5.2f' % (archname, tranname, rdlambda, 100*hist_sum_Y_top))
