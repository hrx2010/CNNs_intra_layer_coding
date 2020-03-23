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
gpuid   = gpuid if len(sys.argv) < 6 else int(sys.argv[5])


maxsteps = 32
maxrates = 17

neural, images, labels = loadnetwork(archname,gpuid,testsize)

neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

perm1, flip1 = getperm('inter')
perm2, flip2 = getperm('exter')
layers = findconv(neural,False)

for l in range(0,len(layers)):
    with torch.no_grad():
        inter_vectors = gettrans(archname,'inter',tranname,l,'')
        exter_vectors = gettrans(archname,'exter',tranname,l,'')
        layer_weights = layers[l].weight.clone()
        trans_weights = exter_vectors[:,:,0].matmul(layer_weights.permute([2,3,0,1])).\
                        matmul(inter_vectors[:,:,0].permute([1,0])).permute([2,3,0,1])
        count_weights = trans_weights.numel()
        [r,c] = [min(int(np.ceil(trans_weights.size(0)/8)),int(np.ceil(count_weights/trans_weights.size(0)/8))),\
                 min(int(np.ceil(trans_weights.size(1)/8)),int(np.ceil(count_weights/trans_weights.size(1)/8)))]
        dims_iters = [maxrates,maxsteps,np.ceil(trans_weights.size(0)/r),np.ceil(trans_weights.size(1)/c)]
        kern_delta = torch.ones(dims_iters,device=getdevice()) * Inf
        kern_coded = torch.ones(dims_iters,device=getdevice()) * Inf
        kern_W_sse = torch.ones(dims_iters,device=getdevice()) * Inf
        kern_Y_sse = torch.ones(dims_iters,device=getdevice()) * Inf
        kern_Y_top = torch.ones(dims_iters,device=getdevice()) * Inf

        for i in range(0,trans_weights.shape[0],r):
            for j in range(0,trans_weights.shape[1],c):
                rs = range(i,min(i+r,trans_weights.shape[0]))
                cs = range(j,min(j+c,trans_weights.shape[1]))
                scale = (trans_weights[rs,cs].reshape(-1)**2).mean().sqrt().log2().floor()
                if scale < -20:
                    continue
                scale = (trans_weights[rs,cs].reshape(-1)**2).mean().sqrt().log2().floor()
                coded = Inf
                start = scale + 2
                for b in range(0,maxrates):
                    last_Y_sse = Inf
                    last_W_sse = Inf
                    for k in range(0,maxsteps):
                        sec = time.time()
                        delta = start + 0.25*k
                        coded = trans_weights[rs,cs].numel()
                        quant_weights = trans_weights.clone()
                        quant_weights[rs,cs] = quantize(quant_weights[rs,cs],2**delta,b)
                        delta_weights = exter_vectors[:,rs].matmul((quant_weights[rs,cs]-trans_weights[rs,cs]).\
                                        permute([2,3,0,1])).matmul(inter_vectors[:,cs].permute([1,0])).permute([2,3,0,1])
                        layers[l].weight[:] = layer_weights + delta_weights
                        Y_hats = predict(neural,images)
                        Y_cats = gettop1(Y_hats)
                        Y_exps = Y_hats.exp()/(Y_hats.exp().sum(1)).reshape(-1,1)
                        sec = time.time() - sec
                        kern_W_sse[b,k,i//r,j//c] = (delta_weights**2).mean()

                        kern_Y_sse[b,k,i//r,j//c] = ((Y_hats - Y)**2).mean()
                        kern_Y_top[b,k,i//r,j//c] = (Y_cats == labels).double().mean()
                        kern_delta[b,k,i//r,j//c] = delta
                        kern_coded[b,k,i//r,j//c] = coded*b
                        mean_Y_sse = kern_Y_sse[b,k,i//r,j//c]
                        mean_Y_top = kern_Y_top[b,k,i//r,j//c]
                        mean_W_sse = kern_W_sse[b,k,i//r,j//c]
                        mean_coded = kern_coded[b,k,i//r,j//c]
                        
                        if mean_Y_sse > last_Y_sse and \
                           mean_W_sse > last_W_sse or  \
                           b == 0:
                            break
                        last_Y_sse = mean_Y_sse
                        last_W_sse = mean_W_sse

                    _,  k = kern_Y_sse[b,:,i//r,j//c].min(0)
                    delta = kern_delta[b,k,i//r,j//c]
                    start = delta - 2
                    mean_Y_sse = kern_Y_sse[b,k,i//r,j//c]
                    mean_Y_top = kern_Y_top[b,k,i//r,j//c]
                    print('%s %s | layer: %03d/%03d, band %04d/%04d,%04d/%04d, delta: %+6.2f, '
                          'mse: %5.2e (%5.2e), top1: %5.2f, rate: %4.1f, time: %5.2fs'\
                          % (archname, tranname, l, len(layers), i, trans_weights.shape[0], j, trans_weights.shape[1],\
                             delta, mean_Y_sse, mean_W_sse, 100*mean_Y_top, b, sec))

                layers[l].weight[:] = layer_weights[:].permute(inv(flip)).reshape(dimen_weights).\
                              permute(inv(perm)).reshape(layers[l].weight.shape)
        io.savemat(('%s_%s_val_%03d_%04d_output_%s_both.mat' % (archname,tranname,l,testsize,trantype)),\
                   {'kern_coded':kern_coded.cpu().numpy(),'kern_Y_sse':kern_Y_sse.cpu().numpy(),\
                    'kern_Y_top':kern_Y_top.cpu().numpy(),'kern_delta':kern_delta.cpu().numpy(),\
                    'kern_W_sse':kern_W_sse.cpu().numpy()})
