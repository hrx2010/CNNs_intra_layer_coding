import torch.nn as nn
import common
import header
import importlib
import network
importlib.reload(common)
importlib.reload(header)

from common import *
from header import *

archname = str(sys.argv[1])
testsize = int(sys.argv[2])

maxsteps = 32
maxrates = 17

neural, _, _, images, labels = loadnetwork(archname,testsize)
network.quantize_2d(neural)
neural = neural.to(common.device)

neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s | top1: %5.2f' % (archname, 100*mean_Y_top))

layers = findlayers(neural,(transconv.QWConv2d))

for l in range(0,len(layers)):
    with torch.no_grad():
        kern_delta = torch.ones(maxrates,maxsteps,layers[l].num_bands(),device=getdevice()) * Inf
        kern_coded = torch.ones(maxrates,maxsteps,layers[l].num_bands(),device=getdevice()) * Inf
        kern_W_sse = torch.ones(maxrates,maxsteps,layers[l].num_bands(),device=getdevice()) * Inf
        kern_Y_sse = torch.ones(maxrates,maxsteps,layers[l].num_bands(),device=getdevice()) * Inf
        kern_Y_top = torch.ones(maxrates,maxsteps,layers[l].num_bands(),device=getdevice()) * Inf
        s = layers[l].get_bandwidth()

        for i in range(0,layers[l].num_bands(),s):
            rs = range(i,min(i+s,layers[l].num_bands()))
            bands = layers[l].get_bands(rs)
            scale = float((bands ** 2).mean().sqrt().log2().floor())

            coded = Inf
            start = scale - 2
            for b in range(0,maxrates):
                last_Y_sse = Inf
                last_W_sse = Inf
                for j in range(0,maxsteps):
                    sec = time.time()
                    delta = start + 0.25*j
                    coded = bands.numel()

                    layers[l].is_quantized = True
                    layers[l].delta[i] = delta
                    layers[l].coded[i] = b * coded

                    Y_hats = predict(neural,images)
                    Y_cats = gettop1(Y_hats)
                    sec = time.time() - sec
                    kern_W_sse[b,j,i] = ((bands - common.quantize(bands,2**delta, b)) ** 2).mean()
                    kern_Y_sse[b,j,i] = ((Y_hats - Y)**2).mean()
                    kern_Y_top[b,j,i] = (Y_cats == labels).double().mean()
                    kern_delta[b,j,i] = delta
                    kern_coded[b,j,i] = b * coded
                    mean_Y_sse = kern_Y_sse[b,j,i]
                    mean_Y_top = kern_Y_top[b,j,i]
                    mean_W_sse = kern_W_sse[b,j,i]
                    mean_coded = kern_coded[b,j,i]

                    layers[l].delta[i] = Inf
                    layers[l].coded[i] = Inf
                    layers[l].is_quantized = False

                    if b == 0:
                        break

                    if mean_Y_sse > last_Y_sse and \
                       mean_W_sse > last_W_sse:
                        break

                    last_W_sse = mean_W_sse
                    last_Y_sse = mean_Y_sse

                _,  j = kern_Y_sse[b,:,i].min(0)
                delta = kern_delta[b,j,i]
                start = delta - 2
                mean_W_sse = kern_W_sse[b,j,i]
                mean_Y_sse = kern_Y_sse[b,j,i]
                mean_Y_top = kern_Y_top[b,j,i]
                print('%s | layer: %03d/%03d, band %04d/%04d, delta: %+6.2f, '
                      'mse: %5.2e (%5.2e), top1: %5.2f, numel: %5.2e, rate: %4.1f, time: %5.2fs'\
                      % (archname, l, len(layers), i, layers[l].num_bands(),\
                         delta, mean_Y_sse, mean_W_sse, 100*mean_Y_top, coded, b, sec))

        io.savemat(('%s_%s_val_%03d_%04d_output_%s_kern.mat' % (archname,'idt',l,testsize,'inter')),\
                   {'kern_coded':kern_coded.cpu().numpy(),'kern_Y_sse':kern_Y_sse.cpu().numpy(),\
                    'kern_Y_top':kern_Y_top.cpu().numpy(),'kern_delta':kern_delta.cpu().numpy(),\
                    'kern_W_sse':kern_W_sse.cpu().numpy()})
