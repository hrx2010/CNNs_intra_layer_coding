import common
import header
import network
import importlib
import os

from common import *
from header import *

trantype = str(sys.argv[1])
tranname = str(sys.argv[2])
archname = str(sys.argv[3])
testsize = int(sys.argv[4])
rdlambda = float(sys.argv[5])
codebase = True if len(sys.argv) <  7 else bool(int(sys.argv[6]))
codekern = True if len(sys.argv) <  8 else bool(int(sys.argv[7]))
codeacti = True if len(sys.argv) <  9 else bool(int(sys.argv[8]))

basic_lr = 1e-4 if len(sys.argv) < 10 else float(sys.argv[ 9])
batch_size = 64 if len(sys.argv) < 11 else   int(sys.argv[10])
nepoch = 10     if len(sys.argv) < 12 else   int(sys.argv[11])

neural, images, labels, images_val, labels_val= loadnetwork(archname,testsize=testsize)
neural = network.transform(neural,trantype,tranname,archname,rdlambda,codekern,codebase,codeacti)

neural.eval()
Y = predict(neural,images_val)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels_val).double().mean()
print('%s %s | slope: %+5.1f, top1: %5.2f' % (archname, tranname, rdlambda, 100*mean_Y_top))

epochs = nepoch
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(neural.parameters(),lr=basic_lr, weight_decay=0.0001, momentum=0.9)

neural.train()
dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4)

best_accuracy = 0
best_epochs = -1

for i in range(0, epochs):

    cnt_step = 0
    for x, y in iter(dataloader):
        x = x.to(common.device)
        y = y.to(common.device)
        y_hat = neural(x)
        loss = criterion(y_hat, y)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epochs %3d steps %4d loss %7.4f' % (i + 1, cnt_step + 1, loss))
        cnt_step = cnt_step + 1

    print('########## evaluate at epoch %d' % (i + 1))
    neural.eval()
    Y = predict(neural, images_val)
    Y_cats = gettop1(Y)
    mean_Y_top = (Y_cats == labels_val).double().mean()
    print('epochs %d: %s %s | top1: %5.2f' % (i + 1, archname, tranname, 100 * mean_Y_top))

    neural.train()
