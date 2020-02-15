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

maxsteps = 48
maxrates = 17

srcnet, images, labels = loadnetwork(archname,gpuid,testsize)
tarnet, images, labels = loadnetwork(archname,gpuid,testsize)
tarnet.eval()

srclayers = findconv(srcnet,False)
tarlayers = findconv(tarnet,False)

perm, flip = getperm(trantype)

Y = predict(tarnet,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

hist_sum_W_sse = [Inf] * maxsteps
hist_sum_Y_sse = [Inf] * maxsteps
pred_sum_Y_sse = [Inf] * maxsteps
hist_sum_coded = [Inf] * maxsteps
hist_sum_denom = [Inf] * maxsteps
hist_sum_Y_top = [Inf] * maxsteps

for j in range(0,maxsteps):
    hist_sum_W_sse[j] = hist_sum_Y_sse[j] = pred_sum_Y_sse[j] = 0.0
    hist_sum_coded[j] = hist_sum_Y_top[j] = hist_sum_denom[j] = 0.0
    with torch.no_grad():
        slope = -36 + 0.5*j
        sec = time.time()
        for l in range(5,len(srclayers)):
            basis_vectors = gettrans(archname,trantype,tranname,l).flatten(2)
            layer_weights = srclayers[l].weight.clone()
            layer_weights = layer_weights.flatten(2).permute(perm)
            dimen_weights = layer_weights.size()
            layer_weights = layer_weights.flatten(1).permute(flip)
            trans_weights = basis_vectors[:,:,0].mm(layer_weights)
            ##load files here
            kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,tranname,trantype,l, 'kern')
            kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, 2**slope)
            base_Y_sse, base_delta, base_coded = loadrdcurves(archname,tranname,trantype,l, 'base')
            base_Y_sse, base_delta, base_coded = findrdpoints(base_Y_sse,base_delta,base_coded, 2**slope)

            stride = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))
            for i in range(0,trans_weights.shape[0],stride):
                rs = range(i,min(i+stride,trans_weights.shape[0]))
                scale = (trans_weights[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -24:
                    continue
                trans_weights[rs,:] = quantize(trans_weights[rs,:],2**kern_delta[i],\
                                               kern_coded[i]/(len(rs)*trans_weights.shape[1]))
                basis_vectors[:,rs] = quantize(basis_vectors[:,rs],2**base_delta[i],\
                                               base_coded[i]/(len(rs)*basis_vectors.shape[0]))
                pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + kern_Y_sse[i] + base_Y_sse[i]
                hist_sum_coded[j] = hist_sum_coded[j] + kern_coded[i] + base_coded[i]

            layer_weights = basis_vectors[:,:,1].mm(trans_weights)
            layer_weights = layer_weights.permute(inv(flip)).reshape(dimen_weights).permute(inv(perm)).\
                            reshape(srclayers[l].weight.size())
            hist_sum_W_sse[j] = hist_sum_W_sse[j] + ((srclayers[l].weight - layer_weights)**2).sum()
            hist_sum_denom[j] = hist_sum_denom[j] + layer_weights.numel()
            tarlayers[l].weight[:] = layer_weights
        Y_hats = predict(tarnet,images)
        Y_cats = gettop1(Y_hats)
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        hist_sum_Y_top[j] = (Y_cats == labels).double().mean()
        hist_sum_W_sse[j] = hist_sum_W_sse[j]/hist_sum_denom[j]
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        sec = time.time() - sec

        print('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e' %\
              (archname, tranname, slope, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
               hist_sum_W_sse[j], 100*hist_sum_Y_top[j], hist_sum_coded[j]))
        if hist_sum_coded[j] == 0.0 or \
           hist_sum_Y_top[j] <= 0.002:
            break
