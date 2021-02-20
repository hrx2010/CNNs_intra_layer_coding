import common
import header
import importlib
importlib.reload(common)
importlib.reload(header)

from common import *
from header import *
from network import convert_qconv

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])
codebase = True if len(sys.argv) < 6 else bool(int(sys.argv[5]))
codekern = True if len(sys.argv) < 7 else bool(int(sys.argv[6]))
codeacti = True if len(sys.argv) < 8 else bool(int(sys.argv[7]))

maxsteps = 48
maxrates = 17

srcnet, _, _, images, labels = loadnetwork(archname,testsize)
tarnet, _, _, images, labels = loadnetwork(archname,testsize)
tarnet = convert_qconv(tarnet)

srclayers = findlayers(srcnet,nn.Conv2d)
tarlayers = findlayers(tarnet,transconv.QAConv2d)
tardimens = hooklayers(findlayers(tarnet,transconv.QAConv2d))

perm, flip = getperm(trantype)

tarnet.eval()
Y = predict(tarnet,images)
Y_cat1 = gettopk(Y,1)
Y_cat5 = gettopk(Y,5)
mean_Y_tp1 = (Y_cat1 == labels[:,None]).double().sum(1).mean()
mean_Y_tp5 = (Y_cat5 == labels[:,None]).double().sum(1).mean()
dimens = [tardimens[i].input[0].numel() for i in range(0,len(tardimens))]

print('%s %s | topk: %5.2f (%5.2f)' % (archname, tranname, 100*mean_Y_tp1, 100*mean_Y_tp5))

hist_sum_W_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
pred_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_tp1 = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_tp5 = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_non0s = torch.ones(maxsteps,len(srclayers),device=getdevice()) * Inf

for j in range(0,maxsteps):
    hist_sum_W_sse[j] = hist_sum_Y_sse[j] = pred_sum_Y_sse[j] = 0.0
    hist_sum_coded[j] = hist_sum_Y_tp1[j] = hist_sum_Y_tp5[j] = hist_sum_denom[j] = 0.0
    with torch.no_grad():
        slope = -30 + 0.5*j
        sec = time.time()
        for l in range(0,len(srclayers)):
            basis_vectors = gettrans(archname,trantype,tranname,l,'').flatten(2)
            layer_weights = srclayers[l].weight.clone()
            layer_weights = layer_weights.flatten(2).permute(perm)
            dimen_weights = layer_weights.size()
            layer_weights = layer_weights.flatten(1).permute(flip)
            trans_weights = basis_vectors[:,:,0].mm(layer_weights)
            ##load files here
            if codekern and srclayers[l].groups >= 1:
                kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,tranname,trantype,l, 'kern')
                kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, 2**slope)
            if codebase and srclayers[l].groups == 1:
                base_Y_sse, base_delta, base_coded = loadrdcurves(archname,tranname,trantype,l, 'base')
                base_Y_sse, base_delta, base_coded = findrdpoints(base_Y_sse,base_delta,base_coded, 2**slope)
            if codeacti and srclayers[l].groups >= 1:
                acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname,tranname,trantype,l, 'acti')
                acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope)

            stride = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))
            for i in range(0,trans_weights.shape[0],stride):
                rs = range(i,min(i+stride,trans_weights.shape[0]))
                scale = (trans_weights[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -20:
                    continue
                if codekern and srclayers[l].groups >= 1:
                    trans_weights[rs,:] = quantize(trans_weights[rs,:],2**kern_delta[i],\
                                                   kern_coded[i]/(len(rs)*trans_weights.shape[1]))
                    pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + kern_Y_sse[i]
                    hist_sum_coded[j] = hist_sum_coded[j] + kern_coded[i]
                if codebase and srclayers[l].groups == 1:
                    basis_vectors[:,rs] = quantize(basis_vectors[:,rs],2**base_delta[i],\
                                                   base_coded[i]/(len(rs)*basis_vectors.shape[0]))
                    pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + base_Y_sse[i]
                    hist_sum_coded[j] = hist_sum_coded[j] + base_coded[i]
            if codeacti and srclayers[l].groups >= 1:
                pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + acti_Y_sse[0]
                hist_sum_coded[j] = hist_sum_coded[j] + acti_coded[0]
                hist_sum_denom[j] = hist_sum_denom[j] + dimens[l]
                tarlayers[l].quantized, tarlayers[l].coded, tarlayers[l].delta = True, acti_coded, acti_delta
            layer_weights = basis_vectors[:,:,1].mm(trans_weights)
            layer_weights = layer_weights.permute(inv(flip)).reshape(dimen_weights).permute(inv(perm)).\
                            reshape(srclayers[l].weight.size())
            hist_sum_non0s[j,l] = (trans_weights != 0).any(1).sum()
            hist_sum_W_sse[j] = hist_sum_W_sse[j] + ((srclayers[l].weight - layer_weights)**2).sum()
            hist_sum_denom[j] = hist_sum_denom[j] + layer_weights.numel()
            tarlayers[l].layer.weight[:] = layer_weights
        Y_hats = predict(tarnet,images)
        Y_cat1 = gettopk(Y_hats,1)
        Y_cat5 = gettopk(Y_hats,5)
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        hist_sum_Y_tp1[j] = (Y_cat1 == labels[:,None]).double().sum(1).mean()
        hist_sum_Y_tp5[j] = (Y_cat5 == labels[:,None]).double().sum(1).mean()
        hist_sum_W_sse[j] = hist_sum_W_sse[j]/hist_sum_denom[j]
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        sec = time.time() - sec

        print('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, topk: %5.2f (%5.2f), rate: %5.2e' %\
              (archname, tranname, slope, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
               hist_sum_W_sse[j], 100*hist_sum_Y_tp1[j], 100*hist_sum_Y_tp5[j],  hist_sum_coded[j]))
        if hist_sum_coded[j] == 0.0 or \
           hist_sum_Y_tp1[j] <= 0.002:
            break

io.savemat(('%s_%s_sum_%d_output_%s.mat' % (archname,tranname,testsize,trantype)),\
           {'hist_sum_Y_sse':hist_sum_Y_sse.cpu().numpy(),'hist_sum_Y_tp1':hist_sum_Y_tp1.cpu().numpy(),\
            'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),\
            'hist_sum_W_sse':hist_sum_W_sse.cpu().numpy(),'hist_sum_denom':hist_sum_denom.cpu().numpy(),\
            'hist_sum_non0s':hist_sum_non0s.cpu().numpy(),'hist_sum_Y_tp5':hist_sum_Y_tp5.cpu().numpy()});

