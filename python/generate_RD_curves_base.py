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

maxsteps = 32
maxrates = 17

neural, images, labels = loadnetwork(archname,gpuid,testsize)

neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

perm, flip = getperm(trantype)
layers = findconv(neural,False)

for l in range(0,len(layers)):
    with torch.no_grad():
        basis_vectors = gettrans(archname,trantype,tranname,l,'')
        layer_weights = layers[l].weight.clone()
        layer_weights = layer_weights.flatten(2).permute(perm)
        dimen_weights = layer_weights.size()
        layer_weights = layer_weights.flatten(1).permute(flip)
        trans_weights = basis_vectors[:,:,0].flatten(1).mm(layer_weights)
        basis_vectors = basis_vectors[:,:,1].flatten(1)
        base_delta = torch.ones(maxrates,maxsteps,trans_weights.size(0),device=getdevice()) * Inf
        base_coded = torch.ones(maxrates,maxsteps,trans_weights.size(0),device=getdevice()) * Inf
        base_W_sse = torch.ones(maxrates,maxsteps,trans_weights.size(0),device=getdevice()) * Inf
        base_Y_sse = torch.ones(maxrates,maxsteps,trans_weights.size(0),device=getdevice()) * Inf
        base_Y_top = torch.ones(maxrates,maxsteps,trans_weights.size(0),device=getdevice()) * Inf
        s = min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(trans_weights.size(1)/8)))

        for i in range(0,trans_weights.shape[0],s):
            rs = range(i,min(i+s,trans_weights.shape[0]))
            scale = (trans_weights[rs,:].reshape(-1)**2).mean().sqrt().log2().floor()
            if scale < -24:
                continue
            scale = (basis_vectors[:,rs].reshape(-1)**2).mean().sqrt().log2().floor()

            coded = Inf
            start = scale - 2
            for b in range(0,maxrates):
                last_Y_sse = Inf
                last_W_sse = Inf
                for j in range(0,maxsteps):
                    sec = time.time()
                    delta = start + 0.25*j
                    coded = s*basis_vectors.shape[0]
                    quant_vectors = basis_vectors.clone()
                    quant_vectors[:,rs] = quantize(quant_vectors[:,rs],2**delta,b)
                    delta_weights = (quant_vectors[:,rs]-basis_vectors[:,rs]).mm(trans_weights[rs,:])
                    layers[l].weight[:] = (layer_weights + delta_weights).permute(inv(flip)).\
                                          reshape(dimen_weights).permute(inv(perm)).reshape(layers[l].weight.shape)
                    Y_hats = predict(neural,images)
                    Y_cats = gettop1(Y_hats)
                    Y_exps = Y_hats.exp()/(Y_hats.exp().sum(1)).reshape(-1,1)
                    sec = time.time() - sec
                    base_W_sse[b,j,i] = (delta_weights**2).mean()
                    #base_Y_log[b,j,i] = -Y_exps[:,0:1000:1000//testsize].diag().log().sum()
                    base_Y_sse[b,j,i] = ((Y_hats - Y)**2).mean()
                    base_Y_top[b,j,i] = (Y_cats == labels).double().mean()
                    base_delta[b,j,i] = delta
                    base_coded[b,j,i] = coded*b
                    mean_Y_sse = base_Y_sse[b,j,i]
                    mean_Y_top = base_Y_top[b,j,i]
                    mean_W_sse = base_W_sse[b,j,i]
                    mean_coded = base_coded[b,j,i]

                    if mean_Y_sse > last_Y_sse and \
                       mean_W_sse > last_W_sse or  \
                       b == 0:
                        break

                    last_Y_sse = mean_Y_sse
                    last_W_sse = mean_W_sse

                _,  j = base_Y_sse[b,:,i].min(0)
                delta = base_delta[b,j,i]
                start = delta - 2
                mean_Y_sse = base_Y_sse[b,j,i]
                mean_Y_top = base_Y_top[b,j,i]
                print('%s %s | layer: %03d/%03d, band %04d/%04d, delta: %+6.2f, '
                      'mse: %5.2e (%5.2e), top1: %5.2f, rate: %4.1f, time: %5.2fs'\
                      % (archname, tranname, l, len(layers), i, basis_vectors.shape[1],\
                         delta, mean_Y_sse, mean_W_sse, 100*mean_Y_top, b, sec))

        layers[l].weight[:] = layer_weights[:].permute(inv(flip)).reshape(dimen_weights).\
                              permute(inv(perm)).reshape(layers[l].weight.shape)
        io.savemat(('%s_%s_val_%03d_%04d_output_%s_base.mat' % (archname,tranname,l,testsize,trantype)),\
                   {'base_coded':base_coded.cpu().numpy(),'base_Y_sse':base_Y_sse.cpu().numpy(),\
                    'base_Y_top':base_Y_top.cpu().numpy(),'base_delta':base_delta.cpu().numpy(),\
                    'base_W_sse':base_W_sse.cpu().numpy()})
