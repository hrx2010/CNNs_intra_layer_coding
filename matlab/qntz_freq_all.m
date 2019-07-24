clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = './ILSVRC2012_val.txt';
tranname = 'dft2';
testsize = 1024;
maxsteps = 96;

[neural,images] = loadnetwork(archname,imagedir, labeldir, testsize);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);
trans = {str2func(tranname), str2func(['i',tranname])};

l_kernel = findconv(neural.Layers); % or specify the layer number directly
l_length = length(l_kernel);

hist_sum_Y_top = zeros(maxsteps,1,testsize)*NaN;
hist_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;
pred_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;
hist_sum_coded = zeros(maxsteps,1)*NaN;
hist_sum_W_sse = zeros(maxsteps,1)*NaN;

Y = pred(neural,nclass,images);
load(sprintf('%s_%s_val_%d',archname,tranname,testsize),'hist_coded','hist_Y_sse','hist_Y_top','hist_delta','hist_W_sse');
for j = 1:maxsteps
    slope = sqrt(2^j)/2^48;
    ydist = cell(l_length,1);
    coded = cell(l_length,1);
    delta = cell(l_length,1);
    wdist = cell(l_length,1);
    denom = cell(l_length,1);

    layers = neural.Layers;
    for l = 1:l_length
        l_ind = l_kernel(l);
        quant = layers(l_ind);
        quant.Weights = trans{1}(quant.Weights);
        [h,w,p,q] = size(quant.Weights);

        ydist{l} = lambda2points(hist_coded{l},mean(hist_Y_sse{l},3),hist_Y_sse{l},slope);
        coded{l} = lambda2points(hist_coded{l},mean(hist_Y_sse{l},3),hist_coded{l},slope);
        delta{l} = lambda2points(hist_coded{l},mean(hist_Y_sse{l},3),hist_delta{l},slope);
        denom{l} = ones(size(coded{l}))*(p*q);
        for i = 1:h*w
            [r,c] = ind2sub([h,w],i);
            % quantize for the given lambda
            quant.Weights(r,c,:) = quantize(quant.Weights(r,c,:),delta{l}(i));
            assert(qentropy(quant.Weights(r,c,:))*(p*q) == coded{l}(i));
        end
        quant.Weights = trans{2}(quant.Weights);
        layers(l_ind) = quant;
        wdist{l} = double(sum(reshape(quant.Weights - neural.Layers(l_ind).Weights,h*w,[]).^2,2));
    end
    ournet = replaceLayers(neural,layers);

    ydist = cell2mat(ydist);
    coded = cell2mat(coded);
    delta = cell2mat(delta);
    wdist = cell2mat(wdist);
    denom = cell2mat(denom);
    
    [Y_hats,Y_cats] = pred(ournet,nclass,images);
    hist_sum_Y_sse(j,1,:) = mean((Y_hats - Y).^2,1);
    hist_sum_Y_top(j,1,:) = images.Labels == Y_cats;
    pred_sum_Y_sse(j,1,:) = sum(ydist,1);
    hist_sum_W_sse(j,1,1) = sum(wdist(:))/sum(denom(:));
    hist_sum_coded(j,1,1) = sum(coded(:))/sum(denom(:));

    disp(sprintf('%s %s | slope: %+5.1f, ymse: %5.2e (%5.2e), wmse: %5.2e, top1: %4.1f, rate: %5.2e',...
         archname, tranname, log2(slope), mean(hist_sum_Y_sse(j,1,:)), mean(pred_sum_Y_sse(j,1,:)), ...
         mean(hist_sum_W_sse(j,1)), 100*mean(hist_sum_Y_top(j,1,:)), mean(hist_sum_coded(j,1))));
    if sum(coded(:)) == 0
        break;
    end
end

save(sprintf('%s_%s_sum_%d',archname,tranname,testsize),'hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_W_sse',...
     'hist_sum_Y_top');