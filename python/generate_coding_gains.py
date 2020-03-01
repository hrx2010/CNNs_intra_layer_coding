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
layers = findconv(neural,False)
perm, flip = getperm(trantype)

for l in range(0,len(layers)):
    G = np.array(loadvarstats(archname,trantype,testsize)[0,l],dtype=np.float64);
    G = G/G.max()
    W = layers[l].weight.flatten(2).permute(perm).flatten(1).permute(flip)
    W = W/W.max()
    U = gettrans(archname,trantype,tranname,l,'').flatten(2)
    U[:,:,0].mm(W.mm(W.permute([1,0]))).mm(U[:,:,1])

# for l = 1:6%length(layers)
#     G = cov{l};
#     G = G./max(G(:));
#     W = net.Layers(layers(l)).Weights;
#     W = permute(W,[3,1,2,4])./max(W(:));
#     W = W(:,:);
#     U = T{l};

#     Dklt = sort(diag(U(:,:,1)*(W*W')*U(:,:,2)).*diag(U(:,:,1)*G*U(:,:,2)),'desc');
#     Didt = sort(diag(W*W').*diag(G),'desc');
#     gains(l) = geomean(Didt)/geomean(Dklt);
# end

# semilogy(1:6,gains);
