import sys
import scipy
import scipy.io as io
import scipy.linalg as linalg
import common
import importlib

importlib.reload(common)
from common import *

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

from model import *
from model.edsr import EDSR
from model.common import ResBlock
from model.common import Upsampler

from tqdm import tqdm

import common
import header
import importlib
importlib.reload(common)
importlib.reload(header)

from common import *
from header import *

trantype = str(args.trantype)
tranname = str(args.tranname)
archname = str(args.archname)
testsize = int(args.testsize)
codebase = True if int(args.codebase) == 1 else 0
codekern = True if int(args.codekern) == 1 else 0
gpuid   = int(args.gpuid)

maxsteps = 48
maxrates = 17

srcnet, images, labels, configs_srcnet = loadnetwork2(archname,gpuid,testsize,args)
tarnet, images, labels, configs_tarnet = loadnetwork2(archname,gpuid,testsize,args)
tarnet.eval()

srclayers = findconv2(archname,srcnet,False)
tarlayers = findconv2(archname,tarnet,False)

perm, flip = getperm(trantype)
Y = predict2(archname, tarnet, images, configs_tarnet)
if archname != 'edsr':
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
hist_sum_Y_psnr = torch.ones(maxsteps,device=getdevice()) * Inf

for j in range(0,maxsteps):
    hist_sum_W_sse[j] = hist_sum_Y_sse[j] = pred_sum_Y_sse[j] = 0.0
    hist_sum_coded[j] = hist_sum_Y_top[j] = hist_sum_denom[j] = 0.0
    with torch.no_grad():
        slope = -34 + 0.5*j
        sec = time.time()
        for l in range(0,len(srclayers)):
            basis_vectors = gettrans(archname,trantype,tranname,l,'').flatten(2)
            layer_weights = srclayers[l].weight.clone()
            layer_weights = layer_weights.flatten(2).permute(perm)
            dimen_weights = layer_weights.size()
            layer_weights = layer_weights.flatten(1).permute(flip)
            trans_weights = basis_vectors[:,:,0].mm(layer_weights)
            ##load files here
            if codekern:
                kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,tranname,trantype,l, 'kern')
                kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, 2**slope)
            if codebase:
                base_Y_sse, base_delta, base_coded = loadrdcurves(archname,tranname,trantype,l, 'base')
                base_Y_sse, base_delta, base_coded = findrdpoints(base_Y_sse,base_delta,base_coded, 2**slope)

            stride = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))
            for i in range(0,trans_weights.shape[0],stride):
                rs = range(i,min(i+stride,trans_weights.shape[0]))
                scale = (trans_weights[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -20:
                    trans_weights[rs,:] = 0
                    basis_vectors[:,rs] = 0
                    continue
                if codekern:
                    trans_weights[rs,:] = quantize(trans_weights[rs,:],2**kern_delta[i],\
                                                   kern_coded[i]/(len(rs)*trans_weights.shape[1]))
                    pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + kern_Y_sse[i]
                    hist_sum_coded[j] = hist_sum_coded[j] + kern_coded[i]
                if codebase:
                    basis_vectors[:,rs] = quantize(basis_vectors[:,rs],2**base_delta[i],\
                                                   base_coded[i]/(len(rs)*basis_vectors.shape[0]))
                    pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + base_Y_sse[i]
                    hist_sum_coded[j] = hist_sum_coded[j] + base_coded[i]
            layer_weights = basis_vectors[:,:,1].mm(trans_weights)
            layer_weights = layer_weights.permute(inv(flip)).reshape(dimen_weights).permute(inv(perm)).\
                            reshape(srclayers[l].weight.size())
            hist_sum_non0s[j,l] = (trans_weights != 0).any(1).sum()
            hist_sum_W_sse[j] = hist_sum_W_sse[j] + ((srclayers[l].weight - layer_weights)**2).sum()
            hist_sum_denom[j] = hist_sum_denom[j] + layer_weights.numel()
            tarlayers[l].weight[:] = layer_weights

        Y_hats = predict2(archname, tarnet, images, configs_tarnet)
        #Y_hats = predict(tarnet,images)
        if archname != 'edsr':
            Y_cats = gettop1(Y_hats)
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        if archname != 'edsr':
            hist_sum_Y_top[j] = (Y_cats == labels).double().mean()
        hist_sum_W_sse[j] = hist_sum_W_sse[j]/hist_sum_denom[j]
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        sec = time.time() - sec
        if archname == 'edsr':
            hist_sum_Y_psnr[j] = predict_psnr(configs_tarnet)

        print('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %5.2f, psnr: %5.2f, rate: %5.2e' %\
              (archname, tranname, slope, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
               hist_sum_W_sse[j], 100*hist_sum_Y_top[j], hist_sum_Y_psnr[j], hist_sum_coded[j]))

        if archname != 'edsr':
            if hist_sum_coded[j] == 0.0 or \
            hist_sum_Y_top[j] <= 0.002:
                break

io.savemat(('%s_%s_sum_%d_output_%s.mat' % (archname,tranname,testsize,trantype)),\
           {'hist_sum_Y_sse':hist_sum_Y_sse.cpu().numpy(),'hist_sum_Y_top':hist_sum_Y_top.cpu().numpy(),\
            'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),\
            'hist_sum_W_sse':hist_sum_W_sse.cpu().numpy(),'hist_sum_denom':hist_sum_denom.cpu().numpy(),\
            'hist_sum_non0s':hist_sum_non0s.cpu().numpy(), 'hist_sum_Y_psnr':hist_sum_Y_psnr.cpu().numpy()});
