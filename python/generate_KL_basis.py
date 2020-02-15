import sys
import scipy
import scipy.io as io
import scipy.linalg as linalg
import common

from common import * 

archname = str(sys.argv[1])
trantype = str(sys.argv[2])
tranname = str(sys.argv[3])
testsize = int(sys.argv[4])

neural, _, _ = loadnetwork(archname,gpuid,1)
layers = common.findconv(neural,False)
covars = common.loadvarstat(archname,trantype,testsize)
perm, flip = getperm(trantype)

T = [None] * len(layers)
for l in range(0,len(layers)):
    with torch.no_grad():
        layer_weights = layers[l].weight
        m = layer_weights.size(0)
        n = layer_weights.size(1)
        layer_weights = layer_weights.reshape(m,n,-1).\
                        permute(perm).flatten(1).permute(flip)
        covH = np.array(layer_weights.mm(layer_weights.permute([1,0])).to('cpu'),dtype=np.float64)
        covG = np.array(covars[0,l],dtype=np.float64)
        if tranname == 'klt':
            _, U = linalg.eigh(covH)
        else:
            _, U = linalg.eigh(covH,covG)
        U = np.linalg.inv(U.transpose())
        S = U/np.sqrt(np.sum(U**2,0))
        A = np.linalg.inv(S)
        T[l] = np.stack((A,S),axis=-1)
    print(("%s %s | generated %s transform for layer %03d" %(archname, trantype, tranname, l)))

io.savemat(('%s_%s_%s_%d.mat' % (archname, trantype, tranname, testsize)),{'T':T})