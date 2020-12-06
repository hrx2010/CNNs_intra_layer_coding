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
gpuid   = gpuid if len(sys.argv) < 6 else int(sys.argv[5])

maxsteps = 32
maxrates = 4

neural, images, labels = loadnetwork(archname,gpuid,testsize)

neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

neural = convert_qconv(neural)
layers = findconv(neural,False)

for l in range(0,len(layers)):
    with torch.no_grad():
        acti_delta = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_coded = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_W_sse = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_Y_sse = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf
        acti_Y_top = torch.ones(maxrates,maxsteps,1,device=getdevice()) * Inf

        scale = 0
        coded = Inf
        start = scale
        for b in range(0,maxrates):
            last_Y_sse = Inf
            last_W_sse = Inf
            for j in range(0,maxsteps):
                sec = time.time()
                delta = start + 0.25*j
                layers[l].quantized, layers[l].depth, layers[l].delta = True, b, delta
                Y_hats = predict(neural,images)
                Y_cats = gettop1(Y_hats)
                sec = time.time() - sec
                #acti_W_sse[b,j,0] = (delta_weights**2).mean()
                acti_Y_sse[b,j,0] = ((Y_hats - Y)**2).mean()
                acti_Y_top[b,j,0] = (Y_cats == labels).double().mean()
                acti_delta[b,j,0] = delta
                acti_coded[b,j,0] = b
                mean_Y_sse = acti_Y_sse[b,j,0]
                mean_Y_top = acti_Y_top[b,j,0]
                #mean_W_sse = acti_W_sse[b,j,0]
                mean_coded = acti_coded[b,j,0]
                
                print('%d, %d, %f' % (b, j, acti_Y_sse[b,j,0]))
                if mean_Y_sse > last_Y_sse or\
                   b == 0:
                    break

                last_Y_sse = mean_Y_sse
                #last_W_sse = mean_W_sse

            _,  j = acti_Y_sse[b,:,0].min(0)
            delta = acti_delta[b,j,0]
            start = delta - 2
            mean_Y_sse = acti_Y_sse[b,j,0]
            mean_Y_top = acti_Y_top[b,j,0]
            print('%s %s | layer: %03d/%03d, delta: %+6.2f, '
                  'mse: %5.2e (%5.2e), top1: %5.2f, rate: %4.1f, time: %5.2fs'\
                  % (archname, tranname, l, len(layers), delta, mean_Y_sse, mean_Y_sse, 100*mean_Y_top, b, sec))

        layers[l].quantized, layers[l].depth, layers[l].delta = False, 0, 0

        io.savemat(('%s_%s_val_%03d_%04d_output_%s_acti.mat' % (archname,tranname,l,testsize,trantype)),\
                   {'acti_coded':acti_coded.cpu().numpy(),'acti_Y_sse':acti_Y_sse.cpu().numpy(),\
                    'acti_Y_top':acti_Y_top.cpu().numpy(),'acti_delta':acti_delta.cpu().numpy()})
