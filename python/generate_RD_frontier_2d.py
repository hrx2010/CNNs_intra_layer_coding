import common
import header
import network

from common import *
from header import *

archname = str(sys.argv[1])
testsize = int(sys.argv[2])
codekern = True if len(sys.argv) < 4 else bool(int(sys.argv[3]))
codeacti = True if len(sys.argv) < 5 else bool(int(sys.argv[4]))

maxsteps = 48
maxrates = 17

neural, _, _, images, labels = loadnetwork(archname,testsize)

network.quantize_2d(neural)
neural = neural.to(common.device)

w_layers = findlayers(neural,transconv.QWConv2d)
a_layers = findlayers(neural,transconv.QAConv2d)
a_dimens = hooklayers(neural,transconv.QAConv2d)

neural.eval()
Y = predict(neural,images)
mean_Y_tp1 = (Y.topk(1,dim=1)[1] == labels[:,None]).double().sum(1).mean()
mean_Y_tp5 = (Y.topk(5,dim=1)[1] == labels[:,None]).double().sum(1).mean()
a_dimens = [a_dimens[i].input for i in range(0,len(a_dimens))]

print('%s | topk: %5.2f (%5.2f)' % (archname, 100*mean_Y_tp1, 100*mean_Y_tp5))

hist_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
pred_sum_Y_sse = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_coded = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_denom = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_tp1 = torch.ones(maxsteps,device=getdevice()) * Inf
hist_sum_Y_tp5 = torch.ones(maxsteps,device=getdevice()) * Inf

for j in range(0,maxsteps):
    hist_sum_Y_sse[j] = pred_sum_Y_sse[j] = hist_sum_coded[j] = 0.0
    hist_sum_Y_tp1[j] = hist_sum_Y_tp5[j] = hist_sum_denom[j] = 0.0
    with torch.no_grad():
        slope = -30 + 0.5*j
        sec = time.time()
        # for l in range(0,max(len(w_layers),len(a_layers))):

        #     if codekern:
        #         hist_sum_denom[j] = hist_sum_denom[j] + w_layers[l].weight.numel()
        #         kern_Y_sse, kern_delta, kern_coded = loadrdcurves(archname,'idt','inter',l, 'kern')
        #         kern_Y_sse, kern_delta, kern_coded = findrdpoints(kern_Y_sse,kern_delta,kern_coded, 2**slope)
        #         stride = w_layers[l].get_bandwidth()
        #         for i in range(0,w_layers[l].num_bands(),stride):
        #             w_layers[l].is_quantized = True
        #             w_layers[l].delta[i] = kern_delta[i]
        #             w_layers[l].coded[i] = kern_coded[i]
        #             pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + kern_Y_sse[i]
        #             hist_sum_coded[j] = hist_sum_coded[j] + kern_coded[i]

        #     if codeacti:
        #         hist_sum_denom[j] = hist_sum_denom[j] + a_dimens[l].prod()
        #         acti_Y_sse, acti_delta, acti_coded = loadrdcurves(archname,'idt','inter',l, 'acti')
        #         acti_Y_sse, acti_delta, acti_coded = findrdpoints(acti_Y_sse,acti_delta,acti_coded, 2**slope)
        #         stride = a_layers[l].get_bandwidth()
        #         for i in range(0,a_layers[l].num_bands(),stride):
        #             a_layers[l].is_quantized = True
        #             a_layers[l].coded[i] = acti_coded[i]
        #             a_layers[l].delta[i] = acti_delta[i]
        #             pred_sum_Y_sse[j] = pred_sum_Y_sse[j] + acti_Y_sse[i]
        #             hist_sum_coded[j] = hist_sum_coded[j] + acti_coded[i]

        pred_sum_Y_sse[j], hist_sum_coded[j], hist_sum_denom[j] = \
            network.quantize_slope_2d(neural, archname, slope, codekern, codeacti)

        Y_hats = predict(neural,images)
        hist_sum_Y_sse[j] = ((Y_hats - Y)**2).mean()
        hist_sum_Y_tp1[j] = (Y_hats.topk(1,dim=1)[1] == labels[:,None]).double().sum(1).mean()
        hist_sum_Y_tp5[j] = (Y_hats.topk(5,dim=1)[1] == labels[:,None]).double().sum(1).mean()
        hist_sum_coded[j] = hist_sum_coded[j]/hist_sum_denom[j]
        sec = time.time() - sec

        print('%s | slope: %+5.1f, ymse: %5.2e (%5.2e), topk: %5.2f (%5.2f), rate: %5.2e' %\
              (archname, slope, hist_sum_Y_sse[j], pred_sum_Y_sse[j], \
               100*hist_sum_Y_tp1[j], 100*hist_sum_Y_tp5[j],  hist_sum_coded[j]))
        if hist_sum_coded[j] == 0.0 or \
           hist_sum_Y_tp1[j] <= 0.002:
            break

io.savemat(('%s_%s_sum_%d_output_%s.mat' % (archname,'idt',testsize,'inter')),\
           {'hist_sum_Y_sse':hist_sum_Y_sse.cpu().numpy(),'hist_sum_Y_tp1':hist_sum_Y_tp1.cpu().numpy(),\
            'pred_sum_Y_sse':pred_sum_Y_sse.cpu().numpy(),'hist_sum_coded':hist_sum_coded.cpu().numpy(),\
            'hist_sum_denom':hist_sum_denom.cpu().numpy(),'hist_sum_Y_tp5':hist_sum_Y_tp5.cpu().numpy()})
