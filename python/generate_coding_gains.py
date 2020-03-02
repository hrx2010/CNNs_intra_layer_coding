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

gainw = [None] * len(layers)
gaing = [None] * len(layers)
areaw = [None] * len(layers)
areag = [None] * len(layers)

for l in range(0,len(layers)):
    G = torch.tensor(loadvarstats(archname,trantype,testsize)[0,l]).to(common.device);
    G = G/G.max()
    W = layers[l].weight.flatten(2).permute(perm).flatten(1).permute(flip)
    W = W/W.max()
    U = gettrans(archname,trantype,tranname,l,'').flatten(2)

    Dwklt = U[:,:,0].mm(W.mm(W.permute([1,0]))).mm(U[:,:,1]).diag()
    Dgklt = U[:,:,0].mm(G).mm(U[:,:,1]).diag()
    Dwidt = W.mm(W.permute([1,0])).diag()
    Dgidt = G.diag()

    gainw[l] = (10*(Dwidt.log().mean().exp()/Dwklt.log().mean().exp()).log10().detach().cpu()).numpy()
    gaing[l] = (10*(Dgidt.log().mean().exp()/Dgklt.log().mean().exp()).log10().detach().cpu()).numpy()
    areaw[l] = (10*(Dwidt.log10()-Dwklt.log10()).sort(descending=True)[0]).detach().cpu().numpy()
    areag[l] = (10*(Dgidt.log10()-Dgklt.log10()).sort(descending=True)[0]).detach().cpu().numpy()

    print('%s %s | layer: %03d/%03d, coding gain: %5.2f %5.2f (%5.2f) dB' %\
          (archname, tranname, l, len(layers), gainw[l], gaing[l], gainw[l] + gaing[l]))

io.savemat(('%s_%s_%s_gain_%04d.mat' % (archname,trantype,tranname,testsize)),\
           {'gainw':gainw,'gaing':gaing,'areaw':areaw,'areag':areag})
