import common
import header
import network
import importlib

from common import *
from header import *

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])
bitdepth = int(sys.argv[5])
rdlambda = float(sys.argv[6])
codebase = True if len(sys.argv) < 8 else bool(int(sys.argv[7]))
codekern = True if len(sys.argv) < 9 else bool(int(sys.argv[8]))
codeacti = True if len(sys.argv) < 10 else bool(int(sys.argv[9]))
gpuid   = gpuid if len(sys.argv) < 11 else int(sys.argv[10])

neural, images, labels = loadnetwork(archname,gpuid,testsize)
neural = network.transform(neural,trantype,tranname,archname,bitdepth,rdlambda,codekern,codebase,codeacti)
neural.eval()
Y = predict(neural,images)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels).double().mean()
print('%s %s | slope: %+5.1f, top1: %5.2f' % (archname, tranname, rdlambda, 100*mean_Y_top))

# quantize

epochs = 100
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
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
