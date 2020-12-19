import sys
import scipy
import scipy.io as io
import scipy.linalg as linalg
import common
import importlib

importlib.reload(common)
from common import * 

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])
gpuid   = gpuid if len(sys.argv) < 6 else int(sys.argv[5])

neural, _, _ = loadnetwork(archname,gpuid,1)
layers = common.findlayers(neural,(nn.Conv2d))
perm, flip = getperm(trantype)

T = [None] * len(layers)
for l in range(0,len(layers)):
    with torch.no_grad():
        layer_weights = layers[l].weight
        layer_weights = layer_weights.flatten(2).permute(perm).flatten(1).permute(flip)
        covH = np.array(layer_weights.mm(layer_weights.permute([1,0])).to('cpu'),dtype=np.float64)
        if tranname == 'idt' or layers[l].groups > 1:
            U = np.eye(covH.shape[0])
        elif tranname == 'klt':
            _, U = linalg.eigh(covH)
        elif tranname == 'elt':
            covars = common.loadvarstats(archname,trantype,testsize)
            covG = np.array(covars[0,l],dtype=np.float64)
            covG = np.linalg.inv(covG + (0.001*np.linalg.norm(covG))*np.eye(covG.shape[0]))
            _, U = linalg.eigh(covH,covG)
        U = np.flip(np.linalg.inv(U.transpose()),1)
        S = U/np.sqrt(np.sum(U**2,0))
        A = np.linalg.inv(S)
        T[l] = np.stack((A,S),axis=-1)
    print(("%s %s | generated %s transform for layer %03d" %(archname, trantype, tranname, l)))

io.savemat(('%s_%s_%s_%d.mat' % (archname, trantype, tranname, testsize)),{'T':T})
