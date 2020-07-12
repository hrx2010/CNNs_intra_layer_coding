import common
import header
import importlib
importlib.reload(network)
importlib.reload(common)
importlib.reload(header)

from common import *
from header import *

tranname = str(sys.argv[1])
archname = str(sys.argv[2])
testsize = int(sys.argv[3])
gpuid   = gpuid if len(sys.argv) < 5 else int(sys.argv[4])

maxsteps = 48
maxparts = 8

srcnet, images, labels = loadnetwork(archname,gpuid,testsize)
srcnet = network.transform2d(srcnet,tranname,archname)
srclayers = findconv(srcnet,False)

tarnet, images, labels = loadnetwork(archname,gpuid,testsize)
tarnet = network.transform2d(tarnet,tranname,archname)
tarlayers = findconv(tarnet,False)
tarnet.eval()

Y = predict(tarnet,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

hist_sum_W_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
pred_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_top = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_non0s = torch.ones(maxsteps,len(srclayers),device=getdevice()) * Inf

for k in range(0,maxsteps):
    with torch.no_grad():
        hist_sum_W_sse[k] = hist_sum_Y_sse[k] = pred_sum_Y_sse[k] = 0.0
        hist_sum_coded[k] = hist_sum_Y_top[k] = hist_sum_denom[k] = 0.0
        slope = -23 + 0.5*k
        sec = time.time()
        for l in range(0,len(tarlayers)):
            m, n, p, _ = list(tarlayers[l].conv2.weight.shape)
                
            s = int(np.ceil(m/maxparts)) if layers[l].conv3 == None else \
                min(int(np.ceil(m/maxparts)),int(np.ceil(n*p*p/maxparts)))

            t = int(np.ceil(n/maxparts)) if layers[l].conv1 == None else \
                min(int(np.ceil(n/maxparts)),int(np.ceil(m*p*p/maxparts)))

            if tarlayers[l].conv1 != None:
                base1_Y_sse, base1_delta, base1_coded = loadrdcurves(archname,tranname,'2d',l, 'base1')
                base1_Y_sse, base1_delta, base1_coded = findrdpoints(base1_Y_sse,base1_delta,base1_coded, 2**slope)
                tarlayer_weight = tarlayers[l].conv1.weight
                tarlayer_weight[:] = srclayers[l].conv1.weight[:]
                for j in range(0,n,t):
                    ca,cz = j, min(j+t,n)
                    scale = (srclayers[l].conv1.weight[:,ca:cz].reshape(-1)**2).mean().sqrt().log2().floor()
                    if scale < -20:
                        continue
                    tarlayer_weight[ca:cz,:] = quantize(tarlayer_weight[ca:cz,:],2**base1_delta[j//t],\
                                                        base1_coded[j//t]/((cz-ca)*n*1*1))
                    pred_sum_Y_sse[k] = pred_sum_Y_sse[k] + base1_Y_sse[j//t]
                    hist_sum_W_sse[k] = hist_sum_W_sse[k] + ((tarlayer_weight - srclayers[l].conv1.weight[:])**2).sum()
                    hist_sum_coded[k] = hist_sum_coded[k] + base1_coded[j//t]

            if tarlayers[l].conv3 != None:
                base3_Y_sse, base3_delta, base3_coded = loadrdcurves(archname,tranname,'2d',l, 'base3')
                base3_Y_sse, base3_delta, base3_coded = findrdpoints(base3_Y_sse,base3_delta,base3_coded, 2**slope)
                tarlayer_weight = tarlayers[l].conv3.weight
                tarlayer_weight[:] = srclayers[l].conv3.weight[:]
                for i in range(0,m,s):
                    ra,rz = i, min(i+s,m)
                    scale = (srclayers[l].conv3.weight[ra:rz,:].reshape(-1)**2).mean().sqrt().log2().floor()
                    if scale < -20:
                        continue
                    tarlayer_weight[:,ra:rz] = quantize(tarlayer_weight[:,ra:rz],2**base3_delta[i//s],\
                                                        base3_coded[i//s]/(m*(rz-ra)*1*1))
                    pred_sum_Y_sse[k] = pred_sum_Y_sse[k] + base3_Y_sse[i//s]
                    hist_sum_W_sse[k] = hist_sum_W_sse[k] + ((tarlayer_weight - srclayers[l].conv3.weight[:])**2).sum()
                    hist_sum_coded[k] = hist_sum_coded[k] + base3_coded[i//s]
            
            if tarlayers[l].conv2 != None:
                kern2_Y_sse, kern2_delta, kern2_coded = loadrdcurves(archname,tranname,'2d',l, 'kern')
                kern2_Y_sse, kern2_delta, kern2_coded = findrdpoints2(kern2_Y_sse,kern2_delta,kern2_coded, 2**slope)
                tarlayer_weight = tarlayers[l].conv2.weight
                tarlayer_weight[:] = srclayers[l].conv2.weight[:]
                for i in range(0,m,s):
                    for j in range(0,n,t):
                        ra,rz = i, min(i+s,m)
                        ca,cz = j, min(j+t,n)
                        scale = (srclayers[l].conv2.weight[ra:rz,ca:cz].reshape(-1)**2).mean().sqrt().log2().floor()
                        if scale < -20:
                            continue
                        tarlayer_weight[ra:rz,ca:cz] = quantize(tarlayer_weight[ra:rz,ca:cz],2**kern2_delta[i//s,j//t],\
                                                                kern2_coded[i//s,j//t]/((rz-ra)*(cz-ca)*p*p))
                        pred_sum_Y_sse[k] = pred_sum_Y_sse[k] + kern2_Y_sse[i//s,j//t]
                        hist_sum_W_sse[k] = hist_sum_W_sse[k] + ((tarlayer_weight - srclayers[l].conv2.weight[:])**2).sum()
                        hist_sum_coded[k] = hist_sum_coded[k] + kern2_coded[i//s,j//t]

            hist_sum_non0s[k,l] = (tarlayer_weight != 0).any(3).any(2).sum()
            hist_sum_denom[k] = hist_sum_denom[k] + tarlayer_weight.numel()

        Y_hats = predict(tarnet,images)
        Y_cats = gettop1(Y_hats)
        hist_sum_Y_sse[k] = ((Y_hats - Y)**2).mean()
        hist_sum_Y_top[k] = (Y_cats == labels).double().mean()
        hist_sum_W_sse[k] = hist_sum_W_sse[k]/hist_sum_denom[k]
        hist_sum_coded[k] = hist_sum_coded[k]/hist_sum_denom[k]
        sec = time.time() - sec

        print('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %5.2f, rate: %5.2e' %\
              (archname, tranname, slope, hist_sum_Y_sse[k], pred_sum_Y_sse[k], \
               hist_sum_W_sse[k], 100*hist_sum_Y_top[k], hist_sum_coded[k]))
        if hist_sum_coded[k] == 0.0 or \
           hist_sum_Y_top[k] <= 0.002:
            break

# io.savemat(('%s_%s_sum_%d_output_%s.mat' % (archname,tranname,testsize,trantype)),\
#            {'hist_sum_Y_sse':hist_sum_Y_sse.cpu().numpy(),'hist_sum_Y_top':hist_sum_Y_top.cpu().numpy(),\
#             'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),\
#             'hist_sum_W_sse':hist_sum_W_sse.cpu().numpy(),'hist_sum_denom':hist_sum_denom.cpu().numpy(),\
#             'hist_sum_non0s':hist_sum_non0s.cpu().numpy()});

