import common
import header
import network
import importlib
importlib.reload(common)
importlib.reload(header)
importlib.reload(network)

from common import *
from header import *

tranname = str(sys.argv[1])
archname = str(sys.argv[2])
testsize = int(sys.argv[3])
gpuid   = gpuid if len(sys.argv) < 5 else int(sys.argv[4])

maxsteps = 32
maxrates = 17
maxparts = 8

neural, images, labels = loadnetwork(archname,gpuid,testsize)
neural = network.trans2d(neural,tranname,archname)

neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

layers = findconv(neural,False)

for l in range(0,len(layers)):
    with torch.no_grad():

        if layers[l].conv1 == None:
            continue

        n, m, p, _ = list(layers[l].conv2.weight.shape)
        s = min(int(np.ceil(m/maxparts)),int(np.ceil(n*p*p/maxparts)))
        
        base_delta = torch.ones(maxrates,maxsteps,m//s,device=getdevice()) * Inf
        base_coded = torch.ones(maxrates,maxsteps,m//s,device=getdevice()) * Inf
        base_W_sse = torch.ones(maxrates,maxsteps,m//s,device=getdevice()) * Inf
        base_Y_sse = torch.ones(maxrates,maxsteps,m//s,device=getdevice()) * Inf
        base_Y_top = torch.ones(maxrates,maxsteps,m//s,device=getdevice()) * Inf

        layer_weight = layers[l].conv1.weight.clone()
        for i in range(0,m,s):
            ra,rz = i, min(i+s,m)
            scale = (layers[l].conv2.weight[:,ra:rz].reshape(-1)**2).mean().sqrt().log2().floor()
            if scale < -20:
                continue
            scale = (layer_weight[ra:rz,:].reshape(-1)**2).mean().sqrt().log2().floor()
            
            coded = Inf
            start = scale + 2
            for b in range(0,maxrates):
                last_Y_sse = Inf
                last_W_sse = Inf
                for k in range(0,maxsteps):
                    sec = time.time()
                    delta = start + 0.25*k
                    coded = (rz-ra)*m
                    quant_weight = layer_weight.clone()
                    quant_weight[ra:rz,:] = quantize(quant_weight[ra:rz,:],2**delta,b)
                    layers[l].conv1.weight[:] = quant_weight[:]
                    Y_hats = predict(neural,images)
                    Y_cats = gettop1(Y_hats)
                    Y_exps = Y_hats.exp()/(Y_hats.exp().sum(1)).reshape(-1,1)
                    sec = time.time() - sec
                    base_W_sse[b,k,i//s] = ((quant_weight[ra:rz,:]-layer_weight[ra:rz,:])**2).mean()
                    base_Y_sse[b,k,i//s] = ((Y_hats - Y)**2).mean()
                    base_Y_top[b,k,i//s] = (Y_cats == labels).double().mean()
                    base_delta[b,k,i//s] = delta
                    base_coded[b,k,i//s] = coded*b
                    mean_Y_sse = base_Y_sse[b,k,i//s]
                    mean_Y_top = base_Y_top[b,k,i//s]
                    mean_W_sse = base_W_sse[b,k,i//s]
                    mean_coded = base_coded[b,k,i//s]
                    
                    if mean_Y_sse > last_Y_sse and \
                       mean_W_sse > last_W_sse or  \
                       b == 0:
                        break

                    last_Y_sse = mean_Y_sse
                    last_W_sse = mean_W_sse

                _,  k = base_Y_sse[b,:,i//s].min(0)
                delta = base_delta[b,k,i//s]
                start = delta - 2
                mean_Y_sse = base_Y_sse[b,k,i//s]
                mean_Y_top = base_Y_top[b,k,i//s]
                print('%s %s | layer: %03d/%03d, band %04d/%04d, delta: %+6.2f, '
                      'mse: %5.2e (%5.2e), top1: %5.2f, rate: %4.1f, time: %5.2fs'\
                      % (archname, tranname, l, len(layers), i, m, delta,\
                         mean_Y_sse, mean_W_sse, 100*mean_Y_top, b, sec))

        layers[l].conv1.weight[:] = layer_weight[:]

        io.savemat(('%s_%s_val_%03d_%04d_output_%s_base1.mat' % (archname,tranname,l,testsize,'2d')),\
                   {'base_coded':base_coded.cpu().numpy(),'base_Y_sse':base_Y_sse.cpu().numpy(),\
                    'base_Y_top':base_Y_top.cpu().numpy(),'base_delta':base_delta.cpu().numpy(),\
                    'base_W_sse':base_W_sse.cpu().numpy()})
