import common
import header
import network
import transconv
import importlib
importlib.reload(common)
importlib.reload(header)
importlib.reload(network)
importlib.reload(transconv)

from common import *
from header import *
from transconv import *

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])
rdlambda = float(sys.argv[5])
codebase = True if len(sys.argv) < 7 else int(sys.argv[6])
codekern = True if len(sys.argv) < 8 else int(sys.argv[7])
gpuid   = gpuid if len(sys.argv) < 9 else int(sys.argv[8])

neural, images, labels = loadnetwork(archname,gpuid,testsize)
neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | top1: %5.2f' % (archname, tranname, 100*mean_Y_top))

# quantize
neural = network.quantize(neural,trantype,tranname,archname,rdlambda,codekern,codebase)

epochs = 40
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(neural.parameters(),lr=0.001, weight_decay=0.0001, momentum=0.9)

neural.train()
for i in range(0,epochs):
    dataloader = torch.utils.data.DataLoader(images,batch_size=50)
    for x, y in dataloader:
        x = x.to(common.device)
        y = y.to(common.device)
        y_hat = neural(x)
        loss = criterion(y_hat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


neural.eval()
Y_hats = predict(neural.to(common.device),images)
Y_cats = gettop1(Y_hats)
hist_sum_Y_sse = ((Y_hats - Y)**2).mean()
hist_sum_Y_top = (Y_cats == labels).double().mean()

print('%s %s | slope: %+5.1f, top1: %5.2f' % (archname, tranname, rdlambda, 100*hist_sum_Y_top))
