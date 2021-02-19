import common
import header
import network

from common import *
from header import *

archname = str(sys.argv[1])
testsize = int(sys.argv[2])
rdlambda = float(sys.argv[3])
codekern = True if len(sys.argv) <  5 else bool(int(sys.argv[4]))
codeacti = True if len(sys.argv) <  6 else bool(int(sys.argv[5]))

basic_lr = 1e-4 if len(sys.argv) < 7 else float(sys.argv[6])
batch_size = 64 if len(sys.argv) < 8 else   int(sys.argv[7])
nepoch = 10     if len(sys.argv) < 9 else   int(sys.argv[8])

neural, images, labels, images_val, labels_val= loadnetwork(archname,testsize=testsize)

network.quantize_2d(neural)
neural = neural.to(common.device)

w_layers = findlayers(neural,transconv.QWConv2d)
a_layers = findlayers(neural,transconv.QAConv2d)
a_dimens = hooklayers(neural,transconv.QAConv2d)

neural.eval()
Y = predict(neural,images_val)
mean_Y_tp1 = (Y.topk(1,dim=1)[1] == labels_val[:,None]).double().sum(1).mean()
mean_Y_tp5 = (Y.topk(5,dim=1)[1] == labels_val[:,None]).double().sum(1).mean()
a_dimens = [a_dimens[i].input for i in range(0,len(a_dimens))]

print('%s | topk: %5.2f (%5.2f)' % (archname, 100*mean_Y_tp1, 100*mean_Y_tp5))

network.quantize_slope_2d(neural, archname, rdlambda,codekern, codeacti, a_dimens)

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
