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
bitdepth = int(sys.argv[5])
rdlambda = float(sys.argv[6])
codebase = True if len(sys.argv) < 8 else bool(int(sys.argv[7]))
codekern = True if len(sys.argv) < 9 else bool(int(sys.argv[8]))
codeacti = True if len(sys.argv) < 10 else bool(int(sys.argv[9]))
gpuid   = 0 if len(sys.argv) < 11 else int(sys.argv[10])
basic_lr = 0.0001 if len(sys.argv) < 12 else float(sys.argv[11])
batch_size = 64 if len(sys.argv) < 13 else int(sys.argv[12])

nepoch = 10
log_path = './logs'
isExists=os.path.exists(log_path)
if not isExists:
    os.makedirs(log_path)

filename_rst = ('%s/results_retraining_%s_%s_%s_%s' % (log_path , archname , sys.argv[2] , sys.argv[6]  , sys.argv[11]))
filename_log = ('%s/log_retraining_%s_%s_%s_%s' % (log_path , archname , sys.argv[2] , sys.argv[6] , sys.argv[11]))
file_results = open( filename_rst, "a+")
file_log = open( filename_log, "a+")

neural, images, labels, images_val,  labels_val= loadnetwork2(archname,gpuid)
neural = network.transform(neural,trantype,tranname,archname,bitdepth,rdlambda,codekern,codebase,codeacti)
neural.eval()
Y = predict(neural,images_val)
Y_cats = gettop1(Y)
mean_Y_top = (Y_cats == labels_val).double().mean()
print('%s %s | slope: %+5.1f, top1: %5.2f' % (archname, tranname, rdlambda, 100*mean_Y_top))
file_results.write("before retraining top_1 %f \n" % (100 * mean_Y_top))

print(neural)

# quantize

epochs = nepoch
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(neural.parameters(),lr=basic_lr, weight_decay=0.0001, momentum=0.9)

neural.train()
dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=16)

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
        print('epochs %d steps %d loss %f' % (i + 1, cnt_step + 1, loss))
        cnt_step = cnt_step + 1

        if (cnt_step % 100) == 0:
            file_log.write("epoch %d step %d loss %f\n" % (i + 1, cnt_step, loss))
            file_log.flush()

    print('########## evaluate at epoch %d' % (i + 1))
    neural.eval()
    Y = predict(neural, images_val)
    Y_cats = gettop1(Y)
    mean_Y_top = (Y_cats == labels_val).double().mean()
    print('epochs %d: %s %s | top1: %5.2f' % (i + 1, archname, tranname, 100 * mean_Y_top))

    file_results.write("epoch %d top_1 %f \n" % (i + 1, 100 * mean_Y_top))
    file_results.flush()

    neural.train()
