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

        m, n, p, _ = list(layers[l].conv2.weight.shape)
        
        s = int(np.ceil(m/maxparts)) if layers[l].conv3 == None else \
            min(int(np.ceil(m/maxparts)),int(np.ceil(n*p*p/maxparts)))

        t = int(np.ceil(n/maxparts)) if layers[l].conv1 == None else \
            min(int(np.ceil(n/maxparts)),int(np.ceil(m*p*p/maxparts)))
        
        kern_delta = torch.ones(maxrates,maxsteps,m//s,n//t,device=getdevice()) * Inf
        kern_coded = torch.ones(maxrates,maxsteps,m//s,n//t,device=getdevice()) * Inf
        kern_W_sse = torch.ones(maxrates,maxsteps,m//s,n//t,device=getdevice()) * Inf
        kern_Y_sse = torch.ones(maxrates,maxsteps,m//s,n//t,device=getdevice()) * Inf
        kern_Y_top = torch.ones(maxrates,maxsteps,m//s,n//t,device=getdevice()) * Inf

        layer_weight = layers[l].conv2.weight.clone()
        for i in range(0,m,s):
            for j in range(0,n,t):
                ra,rz = i, min(i+s,m)
                ca,cz = j, min(j+t,n)
                scale = (layer_weight[ra:rz,ca:cz].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -20:
                    continue
                scale = (layer_weight[ra:rz,ca:cz].reshape(-1)**2).mean().sqrt().log2().floor()

                coded = Inf
                start = scale + 2
                for b in range(0,maxrates):
                    last_Y_sse = Inf
                    last_W_sse = Inf
                    for k in range(0,maxsteps):
                        sec = time.time()
                        delta = start + 0.25*k
                        coded = (rz-ra)*(cz-ca)*p*p
                        quant_weight = layer_weight.clone()
                        quant_weight[ra:rz,ca:cz] = quantize(quant_weight[ra:rz,ca:cz],2**delta,b)
                        layers[l].conv2.weight[:] = quant_weight[:]
                        Y_hats = predict(neural,images)
                        Y_cats = gettop1(Y_hats)
                        Y_exps = Y_hats.exp()/(Y_hats.exp().sum(1)).reshape(-1,1)
                        sec = time.time() - sec
                        kern_W_sse[b,k,i//s,j//t] = ((quant_weight[ra:rz,ca:cz]-layer_weight[ra:rz,ca:cz])**2).mean()
                        kern_Y_sse[b,k,i//s,j//t] = ((Y_hats - Y)**2).mean()
                        kern_Y_top[b,k,i//s,j//t] = (Y_cats == labels).double().mean()
                        kern_delta[b,k,i//s,j//t] = delta
                        kern_coded[b,k,i//s,j//t] = coded*b
                        mean_Y_sse = kern_Y_sse[b,k,i//s,j//t]
                        mean_Y_top = kern_Y_top[b,k,i//s,j//t]
                        mean_W_sse = kern_W_sse[b,k,i//s,j//t]
                        mean_coded = kern_coded[b,k,i//s,j//t]

                        if mean_Y_sse > last_Y_sse and \
                           mean_W_sse > last_W_sse or  \
                           b == 0:
                            break

                        last_Y_sse = mean_Y_sse
                        last_W_sse = mean_W_sse

                    _,  k = kern_Y_sse[b,:,i//s,j//t].min(0)
                    delta = kern_delta[b,k,i//s,j//t]
                    start = delta - 2
                    mean_Y_sse = kern_Y_sse[b,k,i//s,j//t]
                    mean_Y_top = kern_Y_top[b,k,i//s,j//t]
                    print('%s %s | layer: %03d/%03d, band %04d/%04d, %04d/%04d, delta: %+6.2f, '
                          'mse: %5.2e (%5.2e), top1: %5.2f, rate: %4.1f, time: %5.2fs'\
                          % (archname, tranname, l, len(layers), i, m, j, n, delta,\
                             mean_Y_sse, mean_W_sse, 100*mean_Y_top, b, sec))

        layers[l].conv2.weight[:] = layer_weight[:]

        io.savemat(('%s_%s_val_%03d_%04d_output_%s_kern.mat' % (archname,tranname,l,testsize,'2d')),\
                   {'kern_coded':kern_coded.cpu().numpy(),'kern_Y_sse':kern_Y_sse.cpu().numpy(),\
                    'kern_Y_top':kern_Y_top.cpu().numpy(),'kern_delta':kern_delta.cpu().numpy(),\
                    'kern_W_sse':kern_W_sse.cpu().numpy()})
