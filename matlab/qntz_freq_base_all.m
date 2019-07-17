clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 1024;
numslope = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

load([archname,'_freq_base_',num2str(testsize)],'hist_freq_base_coded','hist_freq_base_Y_sse','hist_freq_base_delta');

mean_freq_base_coded = mean(hist_freq_base_coded,3);
mean_freq_base_Y_sse = mean(hist_freq_base_Y_sse,3);
mean_freq_base_delta = mean(hist_freq_base_delta,3);

hist_freq_base_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
pred_freq_base_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
hist_freq_base_sum_coded = zeros(numslope,1,testsize)*NaN;

% layers(l).Weights = dft2(layers(l).Weights);

outputsize = layers(end-1).OutputSize;
Y = zeros(outputsize,testsize);
parfor f = 1:testsize
    X = imds.readimage(f);
    Y(:,f) = predict(neural,X);
end

for j = 1:numslope
    slope = sqrt(2^j)/2^24;
    dists = lambda2points(mean_freq_base_coded,mean_freq_base_Y_sse,hist_freq_base_Y_sse,slope);
    coded = lambda2points(mean_freq_base_coded,mean_freq_base_Y_sse,hist_freq_base_coded,slope);
    steps = lambda2points(mean_freq_base_coded,mean_freq_base_Y_sse,hist_freq_base_delta,slope);
    quant = layers;
    for i = 1:h*w % iterate over output channels
        [r,c] = ind2sub([h,w],i);
        % quantize for the given lambda
        delta = steps(i,1);
        quant(l).Weights(r,c,:,:) = quantize(quant(l).Weights(r,c,:,:),delta);
    end
    % quant(l).Weights = idft2(quant(l).Weights);
    neural = assembleNetwork(quant);
    parfor f = 1:testsize%
        X = imds.readimage(f);
        Y_hat = predict(neural,X);
        pred_freq_base_sum_Y_sse(j,1,f) = sum(dists(:,f));
        hist_freq_base_sum_Y_sse(j,1,f) = mean((Y_hat(:) - Y(:,f)).^2);
        hist_freq_base_sum_coded(j,1,f) = mean(coded(:,f));
    end
    disp(sprintf('%s | layer %03d, band sum, lambda: %5.1f, mse: %5.2e (%5.2e), rate: %5.2e',...
         archname, l, log2(slope), mean(hist_freq_base_sum_Y_sse(j,1,:)), mean(pred_freq_base_sum_Y_sse(j,1,:)), mean(coded(:))));
    if mean(coded(:)) == 0
        break;
    end
end

save([archname,'_freq_base_sum_',num2str(testsize)],'hist_freq_base_sum_coded','hist_freq_base_sum_Y_sse','pred_freq_base_sum_Y_sse');