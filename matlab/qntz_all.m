clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/ILSVRC2012_test_00000*.JPEG';
testsize = 8;
numslope = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

load([archname,'_',num2str(testsize)],'hist_coded','hist_Y_sse','hist_delta');

mean_coded = mean(hist_coded,3);
mean_Y_sse = mean(hist_Y_sse,3);
mean_delta = mean(hist_delta,3);

hist_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
pred_sum_Y_sse = zeros(numslope,1,testsize)*NaN;
hist_sum_coded = zeros(numslope,1,testsize)*NaN;

for j = 1:numslope
    slope = sqrt(2^j)/2^16;
    quant = layers;
    dists = lambda2points(mean_coded,mean_Y_sse,hist_Y_sse,slope);
    coded = lambda2points(mean_coded,mean_Y_sse,hist_coded,slope);
    steps = lambda2points(mean_coded,mean_Y_sse,hist_delta,slope);
    for i = 1:q % iterate over output channels
        convw = quant(l).Weights(:,:,:,i);
        biasw = quant(l).Bias(i);
        % quantize for the given lambda
        delta = steps(i,1);
        % quantize each of the q slices
        convq = quantize(convw,delta);
        biasq = quantize(biasw,delta);
        % assemble the net using layers
        quant(l).Weights(:,:,:,i) = convq;
        quant(l).Bias(i) = biasq;
    end
    net = assembleNetwork(quant);
    for f = 1:testsize%
        X = imds.readimage(f);
        Y = predict(neural,X);
        Y_ssq = sum(Y(:).^2);
        Y_hat = predict(net,X);
        Y_sse = sum((Y_hat(:) - Y(:)).^2);
        hist_sum_Y_sse(j,1,f) = Y_sse;
        pred_sum_Y_sse(j,1,f) = sum(dists(:,f));
        hist_sum_coded(j,1,f) = mean(coded(:,f));
        [~,filename] = fileparts(imds.Files{f});
        disp(sprintf('%s %s | slice sum, lambda: %5.2e, relerr: %5.2e (%5.2e), rate: %5.2e',...
                     archname, filename, slope, sqrt(Y_sse/Y_ssq), sqrt(sum(dists(:,f))/Y_ssq), mean(coded(:,f))));
    end
    if mean(coded(:)) == 0
        break;
    end
end

save([archname,'_sum_',num2str(testsize)],'hist_sum_coded','hist_sum_Y_sse','hist_sum_coded','pred_sum_Y_sse');