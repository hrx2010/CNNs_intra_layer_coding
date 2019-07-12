clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG';
testsize = 128;
numslope = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

load([archname,'_freq_',num2str(testsize)],'hist_freq_coded','hist_freq_Y_sse','hist_freq_delta');

mean_freq_coded = mean(hist_freq_coded,3);
mean_freq_Y_sse = mean(hist_freq_Y_sse,3);
mean_freq_delta = mean(hist_freq_delta,3);

hist_freq_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
pred_freq_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
hist_freq_sum_coded = zeros(numslope,1,testsize)*NaN;

layers(l).Weights = fft2split(fftshift(fftshift(fft2(neural.Layers(l).Weights),1),2));

for j = 1:numslope
    slope = sqrt(2^j)/2^28;
    quant = layers;
    dists = lambda2points(mean_freq_coded,mean_freq_Y_sse,hist_freq_Y_sse,slope);
    coded = lambda2points(mean_freq_coded,mean_freq_Y_sse,hist_freq_coded,slope);
    steps = lambda2points(mean_freq_coded,mean_freq_Y_sse,hist_freq_delta,slope);
    for i = 1:h*w % iterate over output channels
        [r,c] = ind2sub([h,w],i);
        % quantize for the given lambda
        quant(l).Weights(r,c,:,:) = quantize(layers(l).Weights(r,c,:,:),steps(i,1));
    end
    quant(l).Weights = ifft2(ifftshift(ifftshift(ifft2split(quant(l).Weights),1),2));
    net = assembleNetwork(quant);
    parfor f = 1:testsize%
        X = imds.readimage(f);
        Y = predict(neural,X);
        Y_ssq = sum(Y(:).^2);
        Y_hat = predict(net,X);
        Y_sse = sum((Y_hat(:) - Y(:)).^2);
        hist_freq_sum_Y_sse(j,1,f) = Y_sse;
        pred_freq_sum_Y_sse(j,1,f) = sum(dists(:,f));
        hist_freq_sum_coded(j,1,f) = mean(coded(:,f));
        [~,filename] = fileparts(imds.Files{f});
        disp(sprintf('%s %s | band sum, lambda: %5.2e, relerr: %5.2e (%5.2e), rate: %5.2e',...
                     archname, filename, slope, sqrt(Y_sse/Y_ssq), sqrt(sum(dists(:,f))/Y_ssq), mean(coded(:,f))));
    end
    if mean(coded(:)) == 0
        break;
    end
end

save([archname,'_freq_sum_',num2str(testsize)],'hist_freq_sum_coded','hist_freq_sum_Y_sse','pred_freq_sum_Y_sse');