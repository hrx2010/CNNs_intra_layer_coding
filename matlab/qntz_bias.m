clear all;
close all;

% Choose one of: 'alexnet', 'vgg16', 'densenet201', 'mobilenetv2' and
% 'resnet50', and specify the filepath for ILSVRC test images. Number
% of test files to predict can be set manually or set to 0 to predict
% all files in the datastore (not recommended)
archname = 'alexnet';
filepath = '~/Developer/ILSVRC2012/*.JPEG';
testsize = 64;
maxsteps = 64;

[neural,imds] = loadnetwork(archname, filepath);
layers = removeLastLayer(neural);
neural = assembleNetwork(layers);

l = findconv(layers); % or specify the layer number directly
[h,w,p,q] = size(layers(l).Weights);

hist_bias_coded = zeros(maxsteps,1)*NaN;
hist_bias_sum_coded = zeros(maxsteps,1,testsize)*NaN;
hist_bias_sum_Y_sse = zeros(maxsteps,1,testsize)*NaN;

scale = 2^floor(log2(sqrt(mean(layers(l).Bias(:).^2))/1024));
for j = 1:maxsteps
    delta = scale*sqrt(2^(j-1));
    quant = layers;
    biasw = quant(l).Bias;
    biasq = quantize(biasw,delta);
    coded = qentropy(biasq(:));
    quant(l).Bias = biasq;
    net = assembleNetwork(quant);
    for f = 1:testsize%
        X = imds.readimage(f);
        Y = predict(neural,X);
        Y_ssq = sum(Y(:).^2);
        Y_hat = predict(net,X);
        Y_sse = sum((Y_hat(:) - Y(:)).^2);
        hist_bias_sum_Y_sse(j,1,f) = Y_sse;
        hist_bias_sum_coded(j,1,f) = coded;
        [~,filename] = fileparts(imds.Files{f});
        disp(sprintf('%s %s | slice sum, delta: %5.2e, relerr: %5.2e, rate: %5.2e',...
                     archname, filename, delta, sqrt(Y_sse/Y_ssq), coded));
    end
    if hist_bias_sum_coded(j,1,1) == 0
        break;
    end
end

save([archname,'_bias_',num2str(testsize)],'hist_bias_coded','hist_bias_sum_coded','hist_bias_sum_Y_sse');