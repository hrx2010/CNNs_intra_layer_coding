clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
imagedir = '~/Developer/ILSVRC2012_val/*.JPEG';
labeldir = '~/Developer/ILSVRC2012_val/ILSVRC2012_validation_ground_truth.txt';
tranname = 'idt2';
testsize = 1024;
maxsteps = 96;

[neural,imds] = loadnetwork(archname, imagedir, labeldir);
[layers,lclass] = removeLastLayer(neural);
neural = assembleNetwork(layers);
nclass = assembleNetwork(lclass);
trans = {str2func(tranname), str2func(['i',tranname])};

l_inds = findconv(layers); % or specify the layer number directly
l_length = length(l_inds);

hist_sum_Y_top = zeros(maxsteps,1,testsize)*NaN;
hist_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;
pred_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;
hist_sum_coded = zeros(maxsteps,1)*NaN;
hist_sum_W_sse = zeros(maxsteps,1)*NaN;

outputsize = layers(end-1).OutputSize;
Y = zeros(outputsize,testsize);
parfor f = 1:testsize
    X = imds.readimage(f);
    Y(:,f) = predict(neural,X);
end
load([archname,'_',tranname,'_val_',num2str(testsize)],'hist_coded','hist_Y_sse','hist_delta');

for j = 1:maxsteps
    slope = sqrt(2^j)/2^32;
    quant = neural.Layers;
    dists = cell(l_length,1);
    coded = cell(l_length,1);
    steps = cell(l_length,1);
    distw = cell(l_length,1);
    denom = cell(l_length,1);
    for l = 1:1%l_length
        l_ind = l_inds(l);
        quant(l_ind).Weights = trans{1}(quant(l_ind).Weights);
        [h,w,p,q] = size(quant(l_ind).Weights);

        mean_coded = mean(hist_coded{l},3);
        mean_Y_sse = mean(hist_Y_sse{l},3);
        mean_delta = mean(hist_delta{l},3);
        dists{l} = lambda2points(mean_coded,mean_Y_sse,hist_Y_sse{l},slope);
        coded{l} = lambda2points(mean_coded,mean_Y_sse,hist_coded{l},slope);
        steps{l} = lambda2points(mean_coded,mean_Y_sse,hist_delta{l},slope);
        denom{l} = ones(size(coded{l}))*(p*q);
        for i = 1:h*w % iterate over output channels
            [r,c] = ind2sub([h,w],i);
            % quantize for the given lambda
            delta = steps{l}(i);
            quant(l_ind).Weights(r,c,:,:) = quantize(quant(l_ind).Weights(r,c,:,:),delta);
            assert(qentropy(quant(l_ind).Weights(r,c,:,:))*(p*q) == coded{l}(i));
        end
        quant(l_ind).Weights = trans{2}(quant(l_ind).Weights);
        distw{l} = double(sum(reshape(quant(l_ind).Weights - neural.Layers(l_ind).Weights,h*w,[]).^2,2));
    end
    ournet = replaceLayers(neural,quant);

    dists = cell2mat(dists);
    coded = cell2mat(coded);
    steps = cell2mat(steps);
    distw = cell2mat(distw);
    denom = cell2mat(denom);
    parfor f = 1:testsize
        X = imds.readimage(f);
        Y_hat = predict(ournet,X);
        Y_cls = classify(nclass,Y_hat(:));
        hist_sum_Y_top(j,1,f) = Y_cls == imds.Labels(f);
        hist_sum_Y_sse(j,1,f) = mean((Y_hat(:) - Y(:,f)).^2);
        pred_sum_Y_sse(j,1,f) = sum(dists(:,f));
    end
    hist_sum_W_sse(j,1) = sum(distw(:));
    hist_sum_coded(j,1) = sum(coded(:))/sum(denom(:));

    disp(sprintf('%s %s | layer %03d, band sum, lambda: %5.1f, ymse: %5.2e (%5.2e), top1: %5.4e, wmse: %5.2e, rate: %5.2e',...
         archname, tranname, l, log2(slope), mean(hist_sum_Y_sse(j,1,:)), mean(pred_sum_Y_sse(j,1,:)), ...
                  mean(hist_sum_Y_top(j,1,:)), mean(hist_sum_W_sse(j,1)), mean(hist_sum_coded(j,1))));
    if sum(coded(:)) == 0
        break;
    end
end

save([archname,'_',tranname,'_sum_val_',num2str(testsize)],'hist_sum_coded','hist_sum_Y_sse','pred_sum_Y_sse','hist_sum_Y_top');