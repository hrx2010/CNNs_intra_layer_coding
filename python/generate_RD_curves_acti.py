import common
import header
import importlib
import network

from common import *
from header import *

archname = str(sys.argv[1])
testsize = int(sys.argv[2])

maxsteps = 32
maxrates = 16

neural, _, _, images, labels = loadnetwork(archname,testsize)
network.quantize_2d(neural)
neural = neural.to(common.device)

layers = findlayers(neural,(transconv.QAConv2d))
dimens = hooklayers(neural,(transconv.QAConv2d))
print(neural)
neural.eval()
Y = predict(neural,images)
mean_Y_top = (Y.topk(1,dim=1)[1] == labels[:,None]).double().mean()
dimens = [dimens[i].input for i in range(0,len(dimens))]

print('%s | top1: %5.2f' % (archname, 100*mean_Y_top))

for l in range(0,len(layers)):
    with torch.no_grad():
        acti_delta = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_coded = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_W_sse = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_Y_sse = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_Y_top = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf

        scale = 0
        coded = Inf
        start = scale + 2
        for b in range(0,maxrates):
            last_Y_sse = Inf
            last_W_sse = Inf
            for j in range(0,maxsteps):
                sec = time.time()
                delta = start + 0.25*j
                coded = int(dimens[l].prod())
                layers[l].is_quantized, layers[l].coded, layers[l].delta = True, [coded*b], [delta]
                Y_hats = predict(neural,images)
                Y_cats = gettop1(Y_hats)
                sec = time.time() - sec
                #acti_W_sse[b,j,0] = (delta_weights**2).mean()
                acti_Y_sse[b,j,0] = ((Y_hats - Y)**2).mean()
                acti_Y_top[b,j,0] = (Y_cats == labels).double().mean()
                acti_delta[b,j,0] = delta
                acti_coded[b,j,0] = coded*b
                mean_Y_sse = acti_Y_sse[b,j,0]
                mean_Y_top = acti_Y_top[b,j,0]
                #mean_W_sse = acti_W_sse[b,j,0]
                mean_coded = acti_coded[b,j,0]
                
                if mean_Y_sse >= last_Y_sse or\
                   b == 0:
                    break

                last_Y_sse = mean_Y_sse
                #last_W_sse = mean_W_sse

            _,  j = acti_Y_sse[b,:,0].min(0)
            delta = acti_delta[b,j,0]
            start = float(delta - 2)
            mean_Y_sse = acti_Y_sse[b,j,0]
            mean_Y_top = acti_Y_top[b,j,0]
            print('%s | layer: %03d/%03d, delta: %+6.2f, '
                  'mse: %5.2e (%5.2e), top1: %5.2f, numel: %5.2e, rate: %4.1f, time: %5.2fs'\
                  % (archname, l, len(layers), delta, mean_Y_sse, mean_Y_sse, 100*mean_Y_top,\
                     coded, b, sec))

        layers[l].is_quantized, layers[l].coded, layers[l].delta = False, [Inf], [Inf]

        io.savemat(('%s_%s_val_%03d_%04d_output_%s_acti.mat' % (archname,'idt',l,testsize,'inter')),\
                   {'acti_coded':acti_coded.cpu().numpy(),'acti_Y_sse':acti_Y_sse.cpu().numpy(),\
                    'acti_Y_top':acti_Y_top.cpu().numpy(),'acti_delta':acti_delta.cpu().numpy()})
