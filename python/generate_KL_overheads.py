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
layers = common.findconv(neural,False)
perm, flip = getperm(trantype)

overhead = [None] * len(layers)
for l in range(0,len(layers)):
    with torch.no_grad():
        layer_weights = layers[l].weight
        dims = layer_weights.shape
        layer_weights = layer_weights.flatten(2).permute(perm).flatten(1).permute(flip)
        overhead[l] = min(layer_weights.shape[0],layer_weights.shape[1])**2
        overhead[l] = overhead[l]/layer_weights.numel()
        print('overhead for layer %3d (%4d x %4d x %4d x %4d) is %5.2f%%.' %\
              (l, dims[0], dims[1], dims[2], dims[3], 100.0*overhead[l]))
    
        
