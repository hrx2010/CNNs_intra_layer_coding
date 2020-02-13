import common
import header
import importlib
importlib.reload(common)
importlib.reload(header)

from common import *
from header import *

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])

maxsteps = 96
maxrates = 17

neural, images, labels = loadnetwork(archname,gpuid,testsize)

neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

perm, flip = getperm(trantype)
layers = findconv(neural,False)

for j in range(0,maxsteps):
    with torch.no_grad():
        slope = -32 + 0.5*j
        sec = time.time()
        for l in range(0,len(layers)):
            basis_vectors = gettrans(archname,trantype,tranname,l)
            layer_weights = layers[l].weight.clone()
            layer_weights = layer_weights.flatten(2).permute(perm)
            dimen_weights = layer_weights.size()
            layer_weights = layer_weights.flatten(1).permute(flip)
            trans_weights = basis_vectors.flatten(2)[:,:,0].mm(layer_weights)
            ##load files here
            kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,tranname,trantype,l)
            kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, slope)
            s = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))
            for i in range(0,trans_weights.size(1),s):
                rs = range(i,min(i+s,trans_weights.size(1)))
                scale = (trans_weights[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -24:
                    continue
                quant_weights[rs] = quantize(quant_weights[rs],2**kern_delta,kern_coded//(s*trans_weights.size[1]))
            quant_weights = basis_vectors.flatten(2)[:,rs,1].mm(quant_weights)
            layerᆫs[l].weight = quant_weights.permute(inv(flip)).\
                               reshape(dimen_weights).permute(inv(perm)).\
                               reshape(layers[l].weight.size())
            pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + kern_Y_sse.sum()
            hist_sum_coded[j] = hist_sum_coded[j] + kern_coded.sum()
        Y_hats = predict(neural,images)
        Y_cats = gettop1(Y_hats)
        sec = time.time() - sec
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        hist_sum_Y_top[j] = (Y_cats == labels).double().mean()

    print('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e' %\
          archname, tranname, slope, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
          hist_sum_W_sse[j], 100*hist_sum_Y_top[j], hist_sum_coded[j])
    if hist_sum_coded[j] == 0 or \
       hist_sum_Y_top[j] <= 0.002
        break;
    end
        
